import numpy as np
from utils.common_utils import random_argmax
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from planners.disprod import Disprod


class DiscreteDisprod(Disprod):

    def __init__(self, env, cfg, key):
        super(DiscreteDisprod, self).__init__(env, cfg, key)

        if not self.nn_model:
            self.first_partials_fn = jax.jacfwd(self.next_state, argnums=(0, 1))

        self.q_jit = jax.jit(self.q_for_a_restart)
        self.init_action_jit = jax.jit(self.initialize_conformant_actions)
        self.projection_jit = jax.jit(jax.vmap(self.projection, in_axes=0, out_axes=0))

        self.converged_jit = jax.jit(lambda x , thresh : jnp.max(jnp.abs(x)) < thresh)

     # Computes the second order derivatives of a vector
    def hessian(self, fn, wrt):
        return jax.jacfwd(jax.jacrev(fn, argnums=wrt), argnums=wrt)
        
    # Computes the diagonal of hessian.
    def diag_of_hessian(self, s, a, wrt):
        if self.nn_model:
            stacked_hessian = self.hessian(self.next_state, wrt)(s, a)
        else:
            stacked_hessian = self.hessian(self.next_state, wrt)(s, a, self.env)
        return jax.numpy.diagonal(stacked_hessian, axis1=1, axis2=2)

    def set_goal(self, goal_position):
        self.env.goal_x, self.env.goal_y = goal_position

    # actions: (n_restarts, depth, nA)
    def initialize_conformant_actions(self):
        n_batches = self.n_restarts // self.nA
        extra_restarts = self.n_restarts - (n_batches * self.nA)
        batches_of_actions = [self.initialize_batch_of_conformant_actions(self.nA, self.nA) for _ in range(n_batches)]
        if extra_restarts > 0:
            batches_of_actions.append(self.initialize_batch_of_conformant_actions(extra_restarts, self.nA))
        actions = jnp.vstack(batches_of_actions)

        # Shifting:
        # restore the actions of the last saved restart and action apart from the d=first and d=last.

        if self.saved_restart_action is not None:
            marginals_for_last_chosen_action = self.saved_restart_action[2:, self.last_chosen_action]
            actions = actions.at[self.promising_restart, 1:self.depth - 1, self.last_chosen_action].set(
                marginals_for_last_chosen_action[0])
        # return self.projection(actions)
        return actions

    # Initialize a batch of conformant actions
    def initialize_batch_of_conformant_actions(self, n_restarts, nA):
        first_layer_actions = np.expand_dims(self.initialize_first_layer_actions(n_restarts, nA), axis=1)
        subsequent_layer_actions = np.expand_dims(np.random.dirichlet([1] * nA, size=(self.depth - 1)), 0)
        subsequent_layer_actions = np.repeat(subsequent_layer_actions, n_restarts, 0)
        actions = np.concatenate((first_layer_actions, subsequent_layer_actions), axis=1)
        return actions

    def initialize_first_layer_actions(self, n_restarts, nA):
        if nA == n_restarts:
            return np.eye(nA)
        return np.random.randint(0, nA, size=(n_restarts, nA))

    #TODO: Need to return trajectory
    def choose_action(self, obs):
        state = self.preprocessing_fn(obs)
        self.nS = state.shape[0] + 1

        # Create a vector of states corresponding to n_restarts
        initial_state = jnp.tile(state, (self.n_restarts, 1)).astype('float32')

        # Initialize conformant action variables.
        actions = self.initialize_conformant_actions()

        opt_init, self.opt_update, self.get_params = optimizers.adam(self.step_size)
        opt_state = opt_init(actions)

        n_grad_steps = 0
        has_converged = False

        if self.run_mode != "production":
            while n_grad_steps < self.max_grad_steps and not has_converged:
                actions_old = actions.copy()

                (reward, _), grad = jax.vmap(jax.value_and_grad(self.q_jit, argnums=1, has_aux=True), in_axes=(0, 0), out_axes=0)(initial_state, actions)
                # Loss is negative of Q-value
                opt_state = self.opt_update(n_grad_steps, -grad, opt_state)
                updated_actions = self.get_params(opt_state)

                updated_actions = jax.vmap(self.projection, in_axes=0, out_axes=0)(updated_actions)

                updated_reward , _ = jax.vmap(self.q_jit, in_axes=(0, 0), out_axes=0)(initial_state, updated_actions)

                # Find indices where the gradient step increased the loss
                restarts_to_reset = jnp.where(updated_reward < reward)[0]
                if len(restarts_to_reset) > 0:
                    actions = actions.at[restarts_to_reset, :, :].set(actions_old[restarts_to_reset, :, :])
        
                epsilon = actions - actions_old
                has_converged = jnp.max(jnp.abs(epsilon)) < 0.1
                n_grad_steps += 1
        else:
            init_val = (actions, n_grad_steps, has_converged, initial_state, opt_state)
            actions, n_grad_steps , _ , _ , _  = jax.lax.while_loop(self.have_actions_converged , self.iterate_build_graph , init_val)
            
        if self.debug:
            print(f"Gradient steps taken to select action: {n_grad_steps}")
        
       
        q_value, trajectory = jax.vmap(self.q_jit, in_axes=(0, 0), out_axes=0)(initial_state, actions)

        self.promising_restart = random_argmax(q_value)
        best_trajectory = trajectory[self.promising_restart]
        if self.save_actions:
            self.saved_restart_action = actions[self.promising_restart]
        best_initial_state = actions[self.promising_restart][0]
        self.last_chosen_action = jnp.argmax(best_initial_state).item()
        if self.debug:
            print(f"Choosing action {self.last_chosen_action}, Q: {q_value[self.promising_restart]}")
        return self.last_chosen_action, best_trajectory

    def have_actions_converged(self , val):
        _ , n_grad_steps, has_converged , _, _ = val
        return jnp.logical_and(n_grad_steps < 10, jnp.logical_not(has_converged))

    def iterate_build_graph(self, val):
        actions, n_grad_steps, has_converged, initial_state, opt_state = val

        actions_old = actions.at[:].set(actions)

        (reward , _), grad = jax.vmap(jax.value_and_grad(self.q_jit, argnums=1, has_aux = True), in_axes=(0, 0), out_axes=0)(initial_state, actions)
        # grad = jax.jacfwd(self.q)(action_means, action_variance, stacked_state)
        # print(reward)
        # Loss is negative of Q-value.
        opt_state = self.opt_update(n_grad_steps, -grad, opt_state)
        updated_actions = self.get_params(opt_state)

        updated_actions = self.projection_jit(updated_actions)

        updated_reward , _ = jax.vmap(self.q_jit, in_axes=(0, 0), out_axes=0)(initial_state, updated_actions)

        restarts_to_reset = jnp.where(updated_reward < reward, jnp.ones(self.n_restarts, dtype=jnp.int32), jnp.zeros(self.n_restarts, dtype=jnp.int32))
        mask = jnp.tile(restarts_to_reset, (self.depth, self.nA, 1)).transpose(2, 0, 1)
        actions = actions_old * mask + actions * (1-mask)

        # previous_loss = loss

        # Check for convergence of action means and variance. Changed from OR to AND
        mean_epsilon = actions - actions_old
        has_converged = self.converged_jit(mean_epsilon , self.convergance_threshold)
        
        return actions, n_grad_steps+1, has_converged, initial_state, opt_state

    def q_for_a_restart(self, state, actions):
        # augment state by adding variable for noise
        state_expectation = jnp.concatenate((state, jnp.array([0])), 0)
        state_variance = jnp.concatenate((state * 0, jnp.array([1])), 0)

        init_reward = jnp.array(0.0)
        done = 0
        model_params = None if self.nn_model is False else self.model.get_params()

        trajectory_placeholder = jnp.zeros((self.depth , self.nS))
        init_params = (init_reward, state_expectation, state_variance, actions, trajectory_placeholder, model_params)
        aggregate_reward, _, _, _, trajectory, _= jax.lax.fori_loop(0, self.depth, self.computation_graph, init_params)

        return aggregate_reward.sum() , trajectory 


    def computation_graph(self, d, params):
        agg_reward, state_expectation, state_variance, actions, state_expectation_list, model_params = params
        reward = self.reward_marginal(state_expectation, actions[d, :], self.env)
        state_expectation, state_variance = self.next_state_expectation_and_variance(
                state_expectation, state_variance,
                actions[d, :], model_params)

        state_expectation_list = state_expectation_list.at[d].set(state_expectation) 
        
        return agg_reward + reward, state_expectation, state_variance, actions, state_expectation_list, model_params
    
    # https://arxiv.org/pdf/1309.1541.pdf
    def projection(self, actions):
        return jax.vmap(self.project_action, in_axes=0, out_axes=0)(actions)

    def project_action(self, action):
        sorted_actions = jnp.sort(action)[::-1]
        sorted_actions_cumsum = jnp.cumsum(sorted_actions)
        rho_candidates = sorted_actions + (1-sorted_actions_cumsum)/(jnp.arange(1, self.nA+1))
        mask = jnp.where(rho_candidates > 0, jnp.arange(self.nA), -jnp.ones(self.nA, dtype=int))
        rho = jnp.max(mask)
        contribution = (1-sorted_actions_cumsum[rho])/(rho + 1)
        return jax.nn.relu(action + contribution)

    def next_state_expectation_and_variance(self, state_means, state_variance, actions, model_params):
        # assert (state_expectation.shape == (self.n_restarts, self.nS))
        # assert (state_variance.shape == (self.n_restarts, self.nS))
        # assert (actions.shape == (self.n_restarts, self.nA))

        # noise is the last member of transposed state variable. Hence nS+1
        # transposed_state_variables = [state_expectation[:, idx] for idx in range(self.nS + 1)]
        next_state = self.get_next_state(state_means, actions, model_params)

        next_state_expectation, next_state_variance = self.expectation_and_variance(state_means,
                                                                                    state_variance, actions,
                                                                                    next_state, model_params)
        next_state_expectation = jnp.concatenate([next_state_expectation, jnp.array([0])], axis=0)
        next_state_variance = jnp.concatenate([next_state_variance, jnp.array([1])], axis=0)

        return next_state_expectation, next_state_variance

    # state: (nS, n_restart), actions: (nA, n_restart)
    def expectation_and_variance(self, state_means, state_variance, actions, next_state, model_params):

        fop_wrt_state, fop_wrt_action = self.get_first_order_partials(state_means, actions, model_params)
        sop_wrt_state, sop_wrt_action = self.get_second_order_partials(state_means, actions, model_params)

        # Shape of partials : (n_state_variables_wo_noise, n_state_variables_w_noise, n_restarts)
        # Shape of variance : (n_restarts, n_state_variables_w_noise)
        
        next_state_expectations = next_state + 0.5*(jnp.multiply(sop_wrt_state, state_variance).sum(axis=1))
        next_state_variances =  jnp.multiply(jnp.square(fop_wrt_state), state_variance).sum(axis = 1)

        return next_state_expectations, next_state_variances

    # state: (nS, n_restart), actions: (nA, n_restart)
    def get_next_state(self, state, actions, model_params):
        operands = (state, actions, model_params)
        if self.nn_model:
            return self.next_state_for_nn(operands)
        else:
            return self.next_state_for_exact_fn(operands)

    def next_state_for_nn(self, operands):
        state, actions, model_params = operands
        final_actions = self.postprocessing_fn(self.env, state, actions)
        return self.next_state(state, final_actions, model_params)

    def next_state_for_exact_fn(self, operands):
        state, actions, _ = operands
        return self.next_state(state, actions, self.env)
    

    def get_first_order_partials(self, state_means, action_means, model_params = None):
        operands = (state_means, action_means, model_params)
        # return lax.cond(self.nn_model, self.first_order_partials_for_nn, self.first_order_partials_for_exact_fn, operands)

        if self.nn_model:
            return self.first_order_partials_for_nn(operands)
        else:
            return self.first_order_partials_for_exact_fn(operands)

    def first_order_partials_for_nn(self, operands):
        state_means, action_means, model_params = operands
        final_actions = self.postprocessing_fn(self.env, state_means, action_means)
        return self.model.first_order_partials(state_means, final_actions, model_params)

    def first_order_partials_for_exact_fn(self, operands):
        state_means, action_means, _ = operands
        return self.first_partials_fn(state_means, action_means, self.env)

    def get_second_order_partials(self, state_means, action_means, model_params = None):
        operands = (state_means, action_means, model_params)
        # return lax.cond(self.nn_model, self.second_order_partials_for_nn, self.second_order_partials_for_exact_fn, operands)
        if self.nn_model:
            return self.second_order_partials_for_nn(operands)
        else:
            return self.second_order_partials_for_exact_fn(operands)

    def second_order_partials_for_nn(self, operands):
        state_means, action_means, model_params = operands
        final_actions = self.postprocessing_fn(self.env , state_means , action_means)
        return self.model.second_order_partials(state_means, final_actions, model_params)

    def second_order_partials_for_exact_fn(self, operands):
        state_means, action_means, _ = operands
        sop_wrt_state  = self.diag_of_hessian(state_means, action_means, 0) 
        sop_wrt_action = self.diag_of_hessian(state_means, action_means, 1)
        return sop_wrt_state, sop_wrt_action 
