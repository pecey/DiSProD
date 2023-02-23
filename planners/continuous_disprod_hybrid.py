import jax
import jax.numpy as jnp
import time

from planners.disprod import Disprod
from functools import partial
from utils.common_utils import print_, random_argmax
from planners.utils import adam_with_projection
import numpy as np

class ContinuousDisprodHybrid(Disprod):
    def __init__(self, env, cfg, key):
        super(ContinuousDisprodHybrid, self).__init__(env, cfg, key)
        self.low_action = self.env.action_space.low
        self.high_action = self.env.action_space.high
        
        # Multiplicative factor used to transform free_action variables to the legal range.
        self.multiplicative_factor = self.high_action - self.low_action

        if not self.nn_model:
            self.first_partials_fn = jax.jacfwd(self.next_state, argnums=(0, 1))


        if self.nn_model:
            self.next_state_fn = self.next_state_for_nn
            self.partials_fn = self.partials_for_nn
        else:
            self.next_state_fn = self.next_state_for_exact_fn
            self.partials_fn = self.partials_for_exact_fn
            
        if cfg['disprod']['reward_fn_using_taylor']:
            self.reward_fn = self.reward_fn_taylor
        else:
            self.reward_fn = self.reward_fn_non_taylor
            
        self.converged_jit = jax.jit(lambda x , thresh : jnp.max(jnp.abs(x)) < thresh)

        self.choose_action_mean = cfg["disprod"]["choose_action_mean"]

        if cfg["disprod"]["taylor_expansion_mode"] == "complete":
            self.next_state_expectation_and_variance = self.next_state_expectation_and_variance_hybrid
        else:
            raise Exception(f"Unknown value for config taylor_expansion_mode. Got {cfg['taylor_expansion_mode']}")
        
        self.n_bin_var = cfg.get("n_bin_var", 0)
        self.uniform_noise_mean = 0
        self.uniform_noise_variance = 1/12

        self.reset()
            
    # Computes the second order derivatives of a vector
    def hessian(self, fn, wrt):
        return jax.jacfwd(jax.jacrev(fn, argnums=wrt), argnums=wrt)
        
    # Computes the diagonal of hessian.
    def diag_hessian_of_transition(self, s, a, wrt):
        if self.nn_model:
            stacked_hessian = self.hessian(self.next_state, wrt)(s, a)
        else:
            stacked_hessian = self.hessian(self.next_state, wrt)(s, a, self.env, self.alpha)
        return jax.numpy.diagonal(stacked_hessian, axis1=1, axis2=2)

    # TODO : Should be ideally part of env
    def set_goal(self, goal_position):
        self.env.goal_x, self.env.goal_y = goal_position

    def update_model(self, model):
        self.model = model

    def reset(self):
        self.key, subkey = jax.random.split(self.key)
        self.saved_restart = jax.random.uniform(subkey, shape=(self.depth, self.nA))

    # state: (nS - 1,)
    def choose_action(self, state):
        # Two noise variables - One for binary and one for real
        self.nS = state.shape[0] + 2
        self.n_real_var = state.shape[0] - self.n_bin_var
        
        print(f"State: {state}, n_real: {self.n_real_var}, n_bin: {self.n_bin_var}")

        # Create a vector of states corresponding to n_restarts
        stacked_state = jnp.tile(state, (self.n_restarts, 1)).astype('float32')

        # Initialize free_action_mean to [0,1) range and free_action_variance using a uniform distribution
        self.key, subkey1, subkey2, subkey3 = jax.random.split(self.key, 4)
        free_action_mean = self.initialize_action_means(subkey1)
        free_action_variance = self.initialize_action_variance(free_action_mean)

        # Check for expected shapes
        assert(stacked_state.shape == (self.n_restarts, self.nS - 2))
        assert(free_action_mean.shape == free_action_variance.shape == (self.n_restarts, self.depth, self.nA))

        opt_init_mean, self.opt_update_mean, self.get_params_mean = adam_with_projection(self.step_size)
        opt_state_mean = opt_init_mean(free_action_mean)

        opt_init_var, self.opt_update_var, self.get_params_var = adam_with_projection(self.step_size_var)
        opt_state_var = opt_init_var(free_action_variance)

        n_grad_steps = 0
        has_converged = False

        init_val = (free_action_mean, free_action_variance, n_grad_steps, has_converged, stacked_state , opt_state_mean, opt_state_var, jnp.zeros((self.max_grad_steps,)))
        # Iterate until max_grad_steps reached or both means and variance has not converged
        if self.run_mode != "production":
            free_action_mean, free_action_variance, n_grad_steps, _, _, _, _, tmp = self.update_actions(init_val)
        else:
            free_action_mean, free_action_variance, n_grad_steps , _, _, _,_, tmp  = jax.lax.while_loop(self.have_actions_converged , self.update_actions_optimised , init_val)

        assert (jnp.sum(jnp.where(free_action_variance > 1/12, jnp.ones_like(free_action_variance), jnp.zeros_like(free_action_variance))) == 0.0)

        action_means = self.transform_action_means(free_action_mean).block_until_ready()
        action_variance = self.transform_action_variance(free_action_variance).block_until_ready()

        if self.debug:
            print(f"Gradients steps taken: {n_grad_steps}. Resets per step: {tmp}")

        # Compute updated Q-value.
        q_value, trajectory = jax.vmap(self.q_optimised, in_axes=(0, 0, 0), out_axes=0)(stacked_state, action_means, action_variance)
        
        # TODO: If multiple restarts have the same q-value, should we choose the action with lowest variance?
        best_restart = random_argmax(subkey2, q_value)
        self.saved_restart = free_action_mean[best_restart]
        
        print(f"Action chosen: {action_means[best_restart][0]}, variance: {action_variance[best_restart][0]}")

        best_trajectory = trajectory[best_restart]

        if self.choose_action_mean:
            selected_action = action_means[best_restart][0]
        else:
            selected_action = action_means[best_restart][0] + jax.random.normal(subkey3, shape=(self.nA,)) * action_variance[best_restart][0]
        return jnp.clip(selected_action, self.low_action, self.high_action), best_trajectory

    def update_actions(self, val):
        free_action_mean , free_action_variance, n_grad_steps , has_converged , stacked_state , opt_state_mean, opt_state_var, tmp = val
        while n_grad_steps < self.max_grad_steps and not has_converged:
            # free_action_mean_old = free_action_mean.copy()
            # free_action_variance_old = free_action_variance.copy()

            # Transform action means and variance from (0,1) to permissible action ranges
            action_means = self.transform_action_means(free_action_mean)
            action_variance = self.transform_action_variance(free_action_variance)

            (reward , _), (grad_mean, grad_var) = jax.vmap(jax.value_and_grad(self.q, argnums=(1,2), has_aux = True), in_axes=(0, 0, 0), out_axes=0)(stacked_state, action_means, action_variance)
            
            # Loss is negative of Q-value.
            opt_state_mean = self.opt_update_mean(n_grad_steps, -grad_mean, opt_state_mean, 0, 1)
            free_action_mean_  = self.get_params_mean(opt_state_mean)
        
            wiggle_room = jnp.minimum(free_action_mean_ - 0, 1 - free_action_mean_)
            opt_state_var = self.opt_update_var(n_grad_steps, -grad_var, opt_state_var, 0, jnp.minimum(1/12, jnp.square(wiggle_room)/12))
            free_action_variance_ = self.get_params_var(opt_state_var)

            updated_action_means = self.transform_action_means(free_action_mean_)
            updated_action_variance = self.transform_action_variance(free_action_variance_)
            
            updated_reward , _ = jax.vmap(self.q, in_axes=(0, 0, 0), out_axes=0)(stacked_state, updated_action_means, updated_action_variance)

            restarts_to_reset = jnp.where(updated_reward < reward, jnp.ones(self.n_restarts, dtype=jnp.int32), jnp.zeros(self.n_restarts, dtype=jnp.int32))
            mask = jnp.tile(restarts_to_reset, (self.depth, self.nA, 1)).transpose(2, 0, 1)
            free_action_mean_final = free_action_mean * mask + free_action_mean_ * (1-mask)
            free_action_variance_final = free_action_variance * mask + free_action_variance_ * (1-mask)

            tmp.at[n_grad_steps].set(jnp.sum(restarts_to_reset))

            mean_epsilon = free_action_mean_final - free_action_mean
            variance_epsilon = free_action_variance_final - free_action_variance

            free_action_mean = free_action_mean_final
            free_action_variance = free_action_variance_final
            
            has_converged = jnp.max(jnp.abs(mean_epsilon)) < self.convergance_threshold and jnp.max(jnp.abs(variance_epsilon)) < self.convergance_threshold
            n_grad_steps += 1
        return free_action_mean_final, free_action_variance_final, n_grad_steps, None, None, None, tmp

    def have_actions_converged(self , val):
        _ , _ , n_grad_steps , has_converged , _ , _, _, _= val
        return jnp.logical_and(n_grad_steps < self.max_grad_steps, jnp.logical_not(has_converged))

    def update_actions_optimised(self, val):
        free_action_mean, free_action_variance, n_grad_steps, has_converged, stacked_state, opt_state_mean, opt_state_var, tmp = val

        # # Transform action means and variance from (0,1) to permissible action ranges
        action_means = self.transform_action_means(free_action_mean)
        action_variance = self.transform_action_variance(free_action_variance)

        (reward , _), (grad_mean, grad_var) = jax.vmap(jax.value_and_grad(self.q_optimised, argnums=(1,2), has_aux = True), in_axes=(0, 0, 0), out_axes=0)(stacked_state, action_means, action_variance)

        # Loss is negative of Q-value.
        opt_state_mean = self.opt_update_mean(n_grad_steps, -grad_mean, opt_state_mean, 0, 1)
        free_action_mean_  = self.get_params_mean(opt_state_mean)
        
        wiggle_room = jnp.minimum(free_action_mean_ - 0, 1 - free_action_mean_)
        opt_state_var = self.opt_update_var(n_grad_steps, -grad_var, opt_state_var, 0, jnp.minimum(1/12, jnp.square(wiggle_room)/12))
        free_action_variance_ = self.get_params_var(opt_state_var)

        updated_action_means = self.transform_action_means(free_action_mean_)
        updated_action_variance = self.transform_action_variance(free_action_variance_)
        updated_reward , _ = jax.vmap(self.q_optimised, in_axes=(0, 0, 0), out_axes=0)(stacked_state, updated_action_means, updated_action_variance)

        restarts_to_reset = jnp.where(updated_reward < reward, jnp.ones(self.n_restarts, dtype=jnp.int32), jnp.zeros(self.n_restarts, dtype=jnp.int32))
        mask = jnp.tile(restarts_to_reset, (self.depth, self.nA, 1)).transpose(2, 0, 1)
        free_action_mean_final = free_action_mean * mask + free_action_mean_ * (1-mask)
        free_action_variance_final = free_action_variance * mask + free_action_variance_ * (1-mask)

        # Check for convergence of action means and variance. Changed from OR to AND
        mean_epsilon = free_action_mean_final - free_action_mean
        variance_epsilon = free_action_variance_final - free_action_variance

        has_converged = jnp.logical_and(self.converged_jit(mean_epsilon, self.convergance_threshold), self.converged_jit(variance_epsilon, self.convergance_threshold/10))
        return free_action_mean_final, free_action_variance_final, n_grad_steps + 1, has_converged, stacked_state , opt_state_mean, opt_state_var, tmp.at[n_grad_steps].set(jnp.sum(restarts_to_reset))

    def q_optimised(self, state, action_means, action_variance):
        # augment state by adding variable for noise
        state_expectation = jnp.concatenate((state, jnp.array([self.noise_mean, self.uniform_noise_mean])), 0)
        # state_variance = jnp.concatenate((jnp.zeros_like(state), jnp.array([1])), 0)
        # Workaround to make state_variance batched
        state_variance = jnp.concatenate((state * 0, jnp.array([self.noise_var, self.uniform_noise_variance])), 0)

        init_reward = jnp.array(0.0)
        assert(state_expectation.shape == state_variance.shape == (self.nS, ))
        trajectory_placeholder = jnp.zeros((2, self.depth , self.nS))
        init_params = (init_reward, state_expectation, state_variance, action_means, action_variance, trajectory_placeholder)
        aggregate_reward, _, _, _, _ ,  trajectory= jax.lax.fori_loop(0, self.depth, self.rollout_graph, init_params)
        return aggregate_reward.sum() , trajectory

    @partial(jax.jit, static_argnums=(0, ))
    def rollout_graph(self, d, params):
        agg_reward, state_expectation, state_variance, action_means, action_variance, trajectory = params
        
        # Compute immediate reward
        reward = self.reward_fn(state_expectation, state_variance, action_means[d, :], action_variance[d, :], self.env)
        
        # Compute next state expectation
        state_expectation, state_variance = self.next_state_expectation_and_variance(
                state_expectation, state_variance,
                action_means[d, :], action_variance[d, :])
        
        # Update trajectory placeholder
        trajectory = trajectory.at[0, d].set(state_expectation) 
        trajectory = trajectory.at[1, d].set(state_variance)
        
        return agg_reward+reward, state_expectation, state_variance, action_means, action_variance, trajectory

    def q(self, state, action_means, action_variance):
         # augment state by adding variable for noise
        state_expectation = jnp.concatenate((state, jnp.array([0])), 0)
        # state_variance = jnp.concatenate((jnp.zeros_like(state), jnp.array([1])), 0)
        # Workaround to make state_variance batched
        state_variance = jnp.concatenate((state * 0, jnp.array([1])), 0)
        aggregate_reward = jnp.array(0.0)
        trajectory = jnp.zeros((2, self.depth , self.nS))
        for d in range(self.depth):
            reward = self.reward_fn(state_expectation, state_variance, action_means[d, :], action_variance[d, :], self.env)
            aggregate_reward += reward
            state_expectation, state_variance = jax.jit(self.next_state_expectation_and_variance)(
                state_expectation, state_variance,
                action_means[d, :], action_variance[d, :])
             
            # Update trajectory
            trajectory = trajectory.at[0, d].set(state_expectation) 
            trajectory = trajectory.at[1, d].set(state_variance)
        return aggregate_reward.sum() , trajectory

    # Clamp each action between 0 and 1.
    @partial(jax.jit, static_argnums=(0,))
    def project_mean(self, free_action_mean):
        return jnp.clip(free_action_mean, 0, 1)

    # Prevent variance from becoming negative
    @partial(jax.jit, static_argnums=(0,))
    def project_variance(self, free_action_variance):
        return jnp.clip(free_action_variance, 0, 1/12)
    
    
    @partial(jax.jit, static_argnums=(0,))
    def next_state_expectation_and_variance_hybrid(self, state_means, state_variance, actions_means, action_variance):
        operands = state_means, actions_means
        next_state = self.next_state_fn(operands)
    
        (fop_wrt_state, fop_wrt_action), (sop_wrt_state, sop_wrt_action) = self.partials_fn(operands)

        # Taylor's expansion
        next_state_expectation = next_state + 0.5*(jnp.multiply(sop_wrt_action, action_variance).sum(axis = 1) + jnp.multiply(sop_wrt_state, state_variance).sum(axis=1))
        next_state_variance = jnp.multiply(jnp.square(fop_wrt_action), action_variance).sum(axis = 1) + jnp.multiply(jnp.square(fop_wrt_state), state_variance).sum(axis = 1)
        
        # Reset the variance for binary variables
        mu_for_bin = jnp.clip(next_state_expectation[self.n_real_var:self.n_real_var + self.n_bin_var], 0, 1)
        next_state_expectation = next_state_expectation.at[self.n_real_var:self.n_real_var + self.n_bin_var].set(mu_for_bin)
        
        var_for_bin = mu_for_bin*(1-mu_for_bin)
        next_state_variance = next_state_variance.at[self.n_real_var:self.n_real_var + self.n_bin_var].set(var_for_bin)
        
        next_state_expectation = jnp.concatenate([next_state_expectation, jnp.array([self.noise_mean, self.uniform_noise_mean])], axis=0)
        next_state_variance = jnp.concatenate([next_state_variance, jnp.array([self.noise_var, self.uniform_noise_variance])], axis=0)
        
        return next_state_expectation, next_state_variance
    
    # Shape of FOP and SOP: (nS-1, nA), (nS-1, nS)
    @partial(jax.jit, static_argnums=(0,))
    def next_state_expectation_and_variance_complete(self, state_means, state_variance, actions_means, action_variance):
        operands = state_means, actions_means
        next_state = self.next_state_fn(operands)
    
        (fop_wrt_state, fop_wrt_action), (sop_wrt_state, sop_wrt_action) = self.partials_fn(operands)

        # Taylor's expansion
        next_state_expectation = next_state + 0.5*(jnp.multiply(sop_wrt_action, action_variance).sum(axis = 1) + jnp.multiply(sop_wrt_state, state_variance).sum(axis=1))
        next_state_variance = jnp.multiply(jnp.square(fop_wrt_action), action_variance).sum(axis = 1) + jnp.multiply(jnp.square(fop_wrt_state), state_variance).sum(axis = 1)
        
        next_state_expectation = jnp.concatenate([next_state_expectation, jnp.array([self.noise_mean])], axis=0)
        next_state_variance = jnp.concatenate([next_state_variance, jnp.array([self.noise_var])], axis=0)
        
        return next_state_expectation, next_state_variance

    # Ignore the variance terms
    @partial(jax.jit, static_argnums=(0,))
    def next_state_no_var(self, state_means, state_variance, actions_means, action_variance):
        operands = state_means, actions_means
        next_state = self.next_state_fn(operands)

        next_state_expectation = next_state 
        next_state_variance = jnp.zeros((self.nS-1, ))
        
        next_state_expectation = jnp.concatenate([next_state_expectation, jnp.array([self.noise_mean])], axis=0)
        next_state_variance = jnp.concatenate([next_state_variance, jnp.array([self.noise_var])], axis=0)
        
        return next_state_expectation, next_state_variance

    # Use only state variance
    @partial(jax.jit, static_argnums=(0,))
    def next_state_state_var_only(self, state_means, state_variance, actions_means, action_variance):
        operands = state_means, actions_means
        next_state = self.next_state_fn(operands)

        (fop_wrt_state, _), (sop_wrt_state, _) = self.partials_fn(operands)

        next_state_expectation = next_state + 0.5*(jnp.multiply(sop_wrt_state, state_variance).sum(axis=1))
        next_state_variance =  jnp.multiply(jnp.square(fop_wrt_state), state_variance).sum(axis = 1)
        
        next_state_expectation = jnp.concatenate([next_state_expectation, jnp.array([self.noise_mean])], axis=0)
        next_state_variance = jnp.concatenate([next_state_variance, jnp.array([self.noise_var])], axis=0)
        
        return next_state_expectation, next_state_variance
    
    # Use only action variance
    @partial(jax.jit, static_argnums=(0,))
    def next_state_action_var_only(self, state_means, state_variance, actions_means, action_variance):
        operands = state_means, actions_means
        next_state = self.next_state_fn(operands)

        (_, fop_wrt_action), (_, sop_wrt_action) = self.partials_fn(operands)

        next_state_expectation = next_state + 0.5*(jnp.multiply(sop_wrt_action, action_variance).sum(axis=1))
        next_state_variance =  jnp.multiply(jnp.square(fop_wrt_action), action_variance).sum(axis = 1)
        
        next_state_expectation = jnp.concatenate([next_state_expectation, jnp.array([self.noise_mean])], axis=0)
        next_state_variance = jnp.concatenate([next_state_variance, jnp.array([self.noise_var])], axis=0)
        
        return next_state_expectation, next_state_variance


    # Functions for getting next state
    @partial(jax.jit, static_argnums=(0,))
    def next_state_for_nn(self, operands):
        state, actions = operands
        final_actions = self.postprocessing_fn(self.env, state, actions)
        return self.model.next_state(state, final_actions)

    @partial(jax.jit, static_argnums=(0,))
    def next_state_for_exact_fn(self, operands):
        state, actions = operands
        return self.next_state(state, actions, self.env, self.alpha)

    
    # Functions for computing partials
    @partial(jax.jit, static_argnums=(0,))
    def partials_for_nn(self, operands):
        state, actions = operands
        return self.model.get_partials(state, actions)

    @partial(jax.jit, static_argnums=(0,))
    def partials_for_exact_fn(self, operands):
        fop_wrt_state, fop_wrt_action = self.first_order_partials_for_exact_fn(operands)
        sop_wrt_state, sop_wrt_action = self.second_order_partials_for_exact_fn(operands)
        return (fop_wrt_state, fop_wrt_action), (sop_wrt_state, sop_wrt_action)
    
    def first_order_partials_for_exact_fn(self, operands):
        state_means, action_means = operands
        return self.first_partials_fn(state_means, action_means, self.env, self.alpha)


    def second_order_partials_for_exact_fn(self, operands):
        state_means, action_means = operands
        sop_wrt_state  = self.diag_hessian_of_transition(state_means, action_means, 0) 
        sop_wrt_action = self.diag_hessian_of_transition(state_means, action_means, 1)
        return sop_wrt_state, sop_wrt_action 

    # actions ~ Uniform(0,1).
    # If saved, then replace the corresponding action in the sample
    # This has not been JIT intentionally
    def initialize_action_means(self, key):
        actions = jax.random.uniform(key, shape=(self.n_restarts, self.depth, self.nA))
        actions = actions.at[0, : self.depth-1,:].set(self.saved_restart[1:, :])
        return actions

    # Uniform distribution variance : (1/12)(b-a)^2
    @partial(jax.jit, static_argnums=(0,))
    def initialize_action_variance(self, action_means, low_action=0, high_action=1):
        lower_limit = jnp.abs(action_means - low_action)
        higher_limit = jnp.abs(action_means - high_action)
        closer_limit = jnp.minimum(lower_limit, higher_limit)
        variance = jnp.square(2 * closer_limit) / 12
        return variance

    @partial(jax.jit, static_argnums=(0,))
    def transform_action_means(self, free_action_means):
        transformed_actions = self.low_action + self.multiplicative_factor * free_action_means
        return transformed_actions

    @partial(jax.jit, static_argnums=(0,))
    def transform_action_variance(self, free_action_variance):
        return jnp.square(self.multiplicative_factor) * free_action_variance
    
    
    # Computes the diagonal of hessian. Need to add sparsity multiplier.
    def diag_hessian_of_reward(self, s, a, wrt):
        stacked_hessian = self.hessian(self.reward_marginal, wrt)(s, a, self.env)
        return jax.numpy.diagonal(stacked_hessian, axis1=0, axis2=1)

    def second_order_partials_for_reward(self, operands):
        state_means, action_means = operands
        sop_wrt_state  = self.diag_hessian_of_reward(state_means, action_means, 0) 
        sop_wrt_action = self.diag_hessian_of_reward(state_means, action_means, 1)
        return sop_wrt_state, sop_wrt_action 

    def reward_fn_non_taylor(self, s_mean, s_var, a_mean, a_var, env):
        return self.reward_marginal(s_mean, a_mean, env)

    def reward_fn_taylor(self, s_mean, s_var, a_mean, a_var, env):
        reward_for_mean_tau = self.reward_marginal(s_mean, a_mean, env)
        
        (sop_wrt_state, sop_wrt_action) = self.second_order_partials_for_reward((s_mean, a_mean))

        # Taylor's expansion
        return reward_for_mean_tau + 0.5*(jnp.multiply(sop_wrt_action, a_var).sum(axis=0) + jnp.multiply(sop_wrt_state, s_var).sum(axis=0))
