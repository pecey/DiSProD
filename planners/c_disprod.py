import jax
import jax.numpy as jnp

from functools import partial
from planners.utils import adam_with_projection
import numpy as np
from jax import random as jax_random
from planners.disprod import Disprod


class ContinuousDisprod(Disprod):
    def __init__(self, env, cfg, key):

        super(ContinuousDisprod, self).__init__(env, cfg, key)
        self.ac_lb = env.action_space.low
        self.ac_ub = env.action_space.high
        
        # self.nA = cfg["nA"]
        # self.nS = cfg["nS"]
        self.depth = cfg.get("depth")

        self.n_res = cfg["disprod"]["n_restarts"]
        self.max_grad_steps = cfg["disprod"]["max_grad_steps"]
        self.step_size = cfg["disprod"]["step_size"]
        self.step_size_var = cfg["disprod"]["step_size_var"]
        self.conv_thresh = cfg["disprod"]["convergance_threshold"]        
            
        self.log_file = cfg["log_file"]

        # Multiplicative factor used to transform free_action variables to the legal range.
        self.scale_fac = self.ac_ub - self.ac_lb
        self.converged_jit = jax.jit(lambda x, thresh: jnp.max(jnp.abs(x)) < thresh)
        

        # Support for normal noise.
        self.noise_dist = { "norm_mu"   : 0,
                            "norm_var"  : 1}
        self.n_noise = 1
        
        # Partial function to initialize action distribution
        self.ac_dist_init_fn = init_ac_dist(self.n_res, self.depth, self.nA, low_ac=0, high_ac=1)
        
        # Partial function to scale action distribution
        self.ac_dist_trans_fn = trans_ac_dist(self.ac_lb, self.scale_fac)
        
        # Dynamics distribution function
        if cfg["disprod"]["taylor_expansion_mode"] == "complete":
            fop_fn = fop_analytic(self.ns_fn, env)
            sop_fn = sop_analytic(self.ns_fn, env)
            self.dynamics_dist_fn = dynamics_comp(self.ns_fn, env, fop_fn, sop_fn, self.noise_dist)
        elif cfg["disprod"]["taylor_expansion_mode"] == "no_var":
            self.dynamics_dist_fn = dynamics_nv(self.ns_fn, env, self.noise_dist)
        else:
            raise Exception(
                f"Unknown value for config taylor_expansion_mode. Got {cfg['taylor_expansion_mode']}")
        
        # Reward distribution function
        if cfg['disprod']['reward_fn_using_taylor']:
            self.reward_dist_fn = reward_comp(self.reward_fn, self.env)
        else:
            self.reward_dist_fn = reward_mean(self.reward_fn, self.env)

        # Action selector
        if cfg["disprod"]["choose_action_mean"]:
            self.ac_selector = lambda m,v,key: m
        else:
            self.ac_selector = lambda m,v,key: m + jnp.sqrt(v) * jax.random.normal(key, shape=(self.nA,)) 
            
        self.rollout_fn = rollout_graph(self.dynamics_dist_fn, self.reward_dist_fn)
        self.q_fn = q(self.noise_dist, self.depth, self.rollout_fn)
        self.batch_q_fn = jax.vmap(self.q_fn, in_axes=(0, 0, 0, None), out_axes=(0, None))
        
        self.batch_grad_q_fn = jax.vmap(grad_q(self.q_fn), in_axes=(0, 0, 0, None), out_axes=(0, 0))


    def reset(self, key):
        key_1, key_2 = jax.random.split(key)
        ac_seq = jax.random.uniform(key_1, shape=(self.depth, self.nA))
        return ac_seq, key_2

    @partial(jax.jit, static_argnums=(0,))
    def choose_action(self, obs, prev_ac_seq, key):
        ac_seq = prev_ac_seq

        # Create a vector of obs corresponding to n_restarts
        state = jnp.tile(obs, (self.n_res, 1)).astype('float32')

        # key: returned
        # subkey1: for action distribution initialization
        # subkey2: for random argmax
        # subkey3: for sampling action from action distribution
        # subkey4: for sampling noise in encapsulate mode
        key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)

        # Initialize the action distribution
        ac_mean, ac_var = self.ac_dist_init_fn(subkey1, ac_seq)

        opt_init_mean, opt_update_mean, get_params_mean = adam_with_projection(self.step_size)
        opt_state_mean = opt_init_mean(ac_mean)

        opt_init_var, opt_update_var, get_params_var = adam_with_projection(self.step_size_var)
        opt_state_var = opt_init_var(ac_var)

        n_grad_steps = 0
        has_converged = False

        def _update_ac(val):
            """
            Update loop for all the restarts.
            """
            ac_mu_init, ac_var_init, n_grad_steps, has_converged, state, opt_state_mean, opt_state_var, tmp, _key = val

            # Scale action means and variance from (0,1) to permissible action ranges
            scaled_ac_mu, scaled_ac_var = self.ac_dist_trans_fn(ac_mu_init, ac_var_init)

            # Compute Q-value function for all restarts
            reward, _subkey1 = self.batch_q_fn(state, scaled_ac_mu, scaled_ac_var, _key)

            # Compute gradients with respect to action means and action variance.
            grad_mean, grad_var = self.batch_grad_q_fn(state, scaled_ac_mu, scaled_ac_var, _key)

            # Update action distribution based on gradients
            opt_state_mean = opt_update_mean(n_grad_steps, -grad_mean, opt_state_mean, 0, 1)
            ac_mu_upd = get_params_mean(opt_state_mean)

            wiggle_room = jnp.minimum(ac_mu_upd - 0, 1 - ac_mu_upd)
            opt_state_var = opt_update_var(n_grad_steps, -grad_var, opt_state_var, 0, jnp.minimum(1/12, jnp.square(wiggle_room)/12))
            ac_var_upd = get_params_var(opt_state_var)

            # Scale updated action means and variance from (0,1) to permissible action ranges
            scaled_ac_mu_upd, scaled_ac_var_upd = self.ac_dist_trans_fn(ac_mu_upd, ac_var_upd)

            # Compute updated reward
            updated_reward, _subkey2 = self.batch_q_fn(state, scaled_ac_mu_upd, scaled_ac_var_upd, _subkey1)

            # Reset the restarts in which updates led to a poor reward
            restarts_to_reset = jnp.where(updated_reward < reward, jnp.ones(self.n_res, dtype=jnp.int32), jnp.zeros(self.n_res, dtype=jnp.int32))
            mask = jnp.tile(restarts_to_reset, (self.depth, self.nA, 1)).transpose(2, 0, 1)
            ac_mu = ac_mu_init * mask + ac_mu_upd * (1-mask)
            ac_var = ac_var_init * mask + ac_var_upd * (1-mask)

            # Compute action mean and variance epsilon
            ac_mu_eps = ac_mu - ac_mu_init
            ac_var_eps = ac_var - ac_var_init

            # Check for convergence
            has_converged = jnp.logical_and(jnp.max(jnp.abs(ac_mu_eps)) < self.conv_thresh, jnp.max(jnp.abs(ac_var_eps)) < self.conv_thresh/10)

            return ac_mu, ac_var, n_grad_steps + 1, has_converged, state, opt_state_mean, opt_state_var, tmp.at[n_grad_steps].set(jnp.sum(restarts_to_reset)), _subkey2

        def _check_conv(val):
            _, _, n_grad_steps, has_converged, _, _, _, _, _ = val
            return jnp.logical_and(n_grad_steps < self.max_grad_steps, jnp.logical_not(has_converged))

        # Iterate until max_grad_steps reached or both means and variance has not converged

        init_val = (ac_mean, ac_var, n_grad_steps, has_converged, state, opt_state_mean, opt_state_var, jnp.zeros((self.max_grad_steps,)), subkey4)        
        ac_mean, ac_var, n_grad_steps, _, _, _, _, tmp, subkey5 = jax.lax.while_loop(_check_conv, _update_ac, init_val)

        scaled_ac_mean, scaled_ac_var = self.ac_dist_trans_fn(ac_mean, ac_var)

        # if self.debug:
        #       print(f"Gradients steps taken: {n_grad_steps}. Resets per step: {tmp}")

        q_value, _ = self.batch_q_fn(state, scaled_ac_mean, scaled_ac_var, subkey5)

        # TODO: Figure out a JAX version of random_argmax
        # best_restart = random_argmax(subkey2, q_value)
        best_restart = jnp.nanargmax(q_value)
        ac_seq = ac_mean[best_restart]

        ac = self.ac_selector(scaled_ac_mean[best_restart][0], scaled_ac_var[best_restart][0], subkey3)
        
        return jnp.clip(ac, self.ac_lb, self.ac_ub), ac_seq, key


# def random_argmax(key, x, pref_idx=0):
#     options = jnp.where(x == jnp.nanmax(x))[0]
#     val = 0 if 0 in options else jax_random.choice(key, options)
#     return val


#########
# Action Distribution initialization and transformation
#########

# actions ~ Uniform(0,1).
def init_ac_dist(n_res, depth, nA, low_ac, high_ac):
    def _init_ac_dist(key, ac_seq):
        ac_mean = jax.random.uniform(key, shape=(n_res, depth, nA))
        ac_mean = ac_mean.at[0, : depth -1, :].set(ac_seq[1:, :])

        lower_limit = jnp.abs(ac_mean - low_ac)
        higher_limit = jnp.abs(ac_mean - high_ac)
        closer_limit = jnp.minimum(lower_limit, higher_limit)
        ac_var = jnp.square(2 * closer_limit) / 12
        return ac_mean, ac_var
    return _init_ac_dist

def trans_ac_dist(ac_lb, scale_fac):
    def _trans_ac_dict(ac_mean, ac_var):
        scaled_ac_mean = ac_lb + scale_fac * ac_mean
        scaled_ac_var = jnp.square(scale_fac) * ac_var
        return scaled_ac_mean, scaled_ac_var
    return _trans_ac_dict


#####################################
# Q-function computation graph
#################################

def rollout_graph(dynamics_dist_fn, reward_dist_fn):
    def _rollout_graph(d, params):
        agg_reward, s_mu, s_var, a_mu, a_var, key = params
        reward = reward_dist_fn(s_mu, s_var, a_mu[d, :], a_var[d, :])
        ns_mu, ns_var, key = dynamics_dist_fn(s_mu, s_var, a_mu[d, :], a_var[d, :], key)            
        return agg_reward+reward, ns_mu, ns_var, a_mu, a_var, key
    return _rollout_graph

def q(noise_dist, depth, rollout_fn):
    def _q(s, a_mu, a_var, key):
        """
        Compute the Q-function for a single restart
        """
        # augment state by adding variable for noise    
        s_mu = jnp.concatenate([s, jnp.array([noise_dist["norm_mu"]])], axis=0)
        s_var = jnp.concatenate([s * 0, jnp.array([noise_dist["norm_var"]])], axis=0)

        init_rew = jnp.array([0.0])
        init_params = (init_rew, s_mu, s_var, a_mu, a_var, key)
        agg_rew, _, _, _, _, key = jax.lax.fori_loop( 0, depth, rollout_fn, init_params)
        return agg_rew.sum(), key
    return _q
    
def grad_q(q):    
    def _grad_q(s, ac_mu, ac_var, key):
        """
        Compute the gradient of Q-function for a single restart
        """
        grads = jax.grad(q, argnums=(1,2), has_aux=True)(s, ac_mu, ac_var, key)[0]
        return grads[0], grads[1]
    return _grad_q

#####################################
# Dynamics Distribution Fn
#####################################

# No variance mode
def dynamics_nv(ns_fn, env, noise_dist):
    def _dynamics_nv(s_mu, s_var, a_mu, a_var, key):
        ns, reward = ns_fn(s_mu, a_mu, env)
        
        ns_mu = jnp.concatenate([ns, jnp.array([noise_dist["norm_mu"]])], axis=0)
        ns_var = jnp.concatenate([jnp.zeros_like(ns), jnp.array([noise_dist["norm_var"]])], axis=0)

        return ns_mu, ns_var, key
    return _dynamics_nv

# Complete Mode
def dynamics_comp(ns_fn, env, fop_fn, sop_fn, noise_dist):
    def _dynamics_comp(s_mu, s_var, a_mu, a_var, key):
        ns = ns_fn(s_mu, a_mu, env)
        
        fop_wrt_s, fop_wrt_ac = fop_fn(s_mu, a_mu)
        sop_wrt_s, sop_wrt_ac = sop_fn(s_mu, a_mu)

        # Taylor's expansion
        ns_mu = ns + 0.5*(jnp.multiply(sop_wrt_ac, a_var).sum(axis=1) + jnp.multiply(sop_wrt_s, s_var).sum(axis=1))
        ns_var = jnp.multiply(jnp.square(fop_wrt_ac), a_var).sum(axis=1) + jnp.multiply(jnp.square(fop_wrt_s), s_var).sum(axis=1)

        ns_mu = jnp.concatenate([ns_mu, jnp.array([noise_dist["norm_mu"]])], axis=0)
        ns_var = jnp.concatenate([ns_var, jnp.array([noise_dist["norm_var"]])], axis=0)

        return ns_mu, ns_var, key
    return _dynamics_comp

#####################################
# Reward distribution fn
#####################################

def reward_mean(reward_fn, env):
    def _reward_mean(s_mu, s_var, ac_mu, ac_var):
        return reward_fn(s_mu, ac_mu, env)
    return _reward_mean

def reward_comp(reward_fn, env):
    def _reward_comp(s_mu, s_var, ac_mu, ac_var):
        reward_mu = reward_fn(s_mu, ac_mu, env)
        def _diag_hessian(wrt):
            hess = jax.hessian(reward_fn, wrt)(s_mu, ac_mu, env)
            return jax.numpy.diagonal(hess, axis1=1, axis2=2)
        sop_wrt_s = _diag_hessian(0)
        sop_wrt_ac = _diag_hessian(1)

        return reward_mu + 0.5*(jnp.multiply(sop_wrt_ac, ac_var).sum(axis=0) + 
                                jnp.multiply(sop_wrt_s, s_var).sum(axis=0))
    return _reward_comp

#########################################
# Functions for computing partials - Analytic
###########################################

def fop_analytic(ns_fn, env):
    def _fop_analytic(s_mu, ac_mu):
        return jax.jacfwd(ns_fn, argnums=(0, 1))(s_mu, ac_mu, env)
    return _fop_analytic

def sop_analytic(ns_fn, env):
    def _sop_analytic(s_mu, ac_mu):
        def _diag_hessian(wrt):
            hess = jax.hessian(ns_fn, wrt)(s_mu, ac_mu, env)
            return jax.numpy.diagonal(hess, axis1=1, axis2=2)
        # TODO: Compute in one call
        sop_wrt_s = _diag_hessian(0)
        sop_wrt_ac = _diag_hessian(1)
        return sop_wrt_s, sop_wrt_ac
    return _sop_analytic