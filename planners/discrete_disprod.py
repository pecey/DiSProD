import jax
import jax.numpy as jnp

from functools import partial
from planners.utils import adam_with_projection
from planners.disprod import Disprod

class DiscreteDisprod(Disprod):
    def __init__(self, env, cfg):

        super(DiscreteDisprod, self).__init__(env, cfg)

        self.ac_lb = 0
        self.ac_ub = env.action_space.n - 1
        
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
        
        self.n_bin_var = cfg.get("n_bin_var", 0)
        self.n_real_var = self.nS - self.n_bin_var
        
        # Partial function to initialize action distribution
        self.ac_dist_init_fn = init_ac_dist(self.n_res, self.depth, self.nA, low_ac=0, high_ac=1)    
        
        norm_mu, norm_var = 0, 1
        
        noise_dist = (jnp.array([norm_mu]), jnp.array([norm_var]))

        projection = projection_fn(self.nA)
        self.batch_projection = jax.vmap(projection, in_axes=(0), out_axes=(0))
       
        
        # Dynamics distribution function
        if cfg["disprod"]["taylor_expansion_mode"] == "complete":
            fop_fn = fop_analytic(self.ns_fn, env)
            sop_fn = sop_analytic(self.ns_fn, env)
            self.dynamics_dist_fn = dynamics_comp(self.ns_fn, env, fop_fn, sop_fn, noise_dist, self.n_real_var, self.n_bin_var)
        elif cfg["disprod"]["taylor_expansion_mode"] == "no_var":
            self.dynamics_dist_fn = dynamics_nv(self.ns_fn, env, noise_dist, self.n_real_var, self.n_bin_var)
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
        self.q_fn = q(noise_dist, self.depth, self.rollout_fn)
        self.batch_q_fn = jax.vmap(self.q_fn, in_axes=(0, 0), out_axes=(0))
        
        self.batch_grad_q_fn = jax.vmap(grad_q(self.q_fn), in_axes=(0, 0), out_axes=(0))


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
        key, subkey1 = jax.random.split(key, 2)

        # Initialize the action distribution
        ac = self.ac_dist_init_fn(subkey1, ac_seq)

        opt_init_mean, opt_update_mean, get_params_mean = adam_with_projection(self.step_size, proj_fn=self.batch_projection)
        opt_state_mean = opt_init_mean(ac)


        n_grad_steps = 0
        has_converged = False

        def _update_ac(val):
            """
            Update loop for all the restarts.
            """
            ac_init, n_grad_steps, has_converged, state, opt_state_mean, tmp = val

            # Compute Q-value function for all restarts
            reward = self.batch_q_fn(state, ac_init)

            # Compute gradients with respect to action marginals.
            grads = self.batch_grad_q_fn(state, ac_init)

            # Update action distribution based on gradients
            opt_state_mean = opt_update_mean(n_grad_steps, -grads, opt_state_mean)
            ac_mu_upd = get_params_mean(opt_state_mean)

            # Compute updated reward
            updated_reward = self.batch_q_fn(state, ac_mu_upd)

            # Reset the restarts in which updates led to a poor reward
            restarts_to_reset = jnp.where(updated_reward < reward, jnp.ones(self.n_res, dtype=jnp.int32), jnp.zeros(self.n_res, dtype=jnp.int32))
            mask = jnp.tile(restarts_to_reset, (self.depth, self.nA, 1)).transpose(2, 0, 1)
            ac_mu = ac_init * mask + ac_mu_upd * (1-mask)

            # Compute action mean and variance epsilon
            ac_mu_eps = ac_mu - ac_init

            # Check for convergence
            has_converged = jnp.max(jnp.abs(ac_mu_eps)) < self.conv_thresh

            return ac_mu, n_grad_steps + 1, has_converged, state, opt_state_mean, tmp.at[n_grad_steps].set(jnp.sum(restarts_to_reset))

        def _check_conv(val):
            _, n_grad_steps, has_converged, _, _, _ = val
            return jnp.logical_and(n_grad_steps < self.max_grad_steps, jnp.logical_not(has_converged))

        # Iterate until max_grad_steps reached or both means and variance has not converged

        init_val = (ac, n_grad_steps, has_converged, state, opt_state_mean, jnp.zeros((self.max_grad_steps,)))        
        ac, n_grad_steps, _, _, _, tmp = jax.lax.while_loop(_check_conv, _update_ac, init_val)


        # if self.debug:
        #       print(f"Gradients steps taken: {n_grad_steps}. Resets per step: {tmp}")

        q_value = self.batch_q_fn(state, ac)

        # TODO: Figure out a JAX version of random_argmax
        # best_restart = random_argmax(subkey2, q_value)
        best_restart = jnp.nanargmax(q_value)
        ac_seq = ac[best_restart]

        ac = jnp.argmax(ac_seq[0])
        
        return ac, ac_seq, key


# def random_argmax(key, x, pref_idx=0):
#     options = jnp.where(x == jnp.nanmax(x))[0]
#     val = 0 if 0 in options else jax_random.choice(key, options)
#     return val


#########
# Action Distribution initialization and transformation
#########

def init_ac_dist(n_res, depth, nA, low_ac, high_ac):
    def _init_ac_dist(key, ac_seq):
        key1, key2 = jax.random.split(key)
        # Layer 1 actions are concrete
        ac_l1 = jax.random.randint(key1, shape=(n_res, 1, nA), minval=0, maxval=1)
        # Rest actions are marginals
        ac_l2 = jax.random.dirichlet(key2, jnp.ones(nA), (n_res, depth-1))

        ac = jnp.concatenate([ac_l1, ac_l2], axis=1)
        return ac
    return _init_ac_dist

# https://arxiv.org/pdf/1309.1541.pdf
def projection_fn(nA):
    def _projection_fn(ac):
        ac_sort = jnp.sort(ac)[::-1]
        ac_sort_cumsum = jnp.cumsum(ac_sort)
        rho_candidates = ac_sort + (1 - ac_sort_cumsum)/jnp.arange(1, nA+1)
        mask = jnp.where(rho_candidates > 0, jnp.arange(nA, dtype=jnp.int32), -jnp.ones(nA, dtype=jnp.int32))
        rho = jnp.max(mask)
        contrib = (1 - ac_sort_cumsum[rho])/(rho + 1)
        return jax.nn.relu(ac + contrib)
    return jax.vmap(_projection_fn, in_axes=0, out_axes=0)

#####################################
# Q-function computation graph
#################################

def rollout_graph(dynamics_dist_fn, reward_dist_fn):
    def _rollout_graph(d, params):
        agg_reward, s_mu, s_var, a_mu = params
        reward = reward_dist_fn(s_mu, s_var, a_mu[d, :])
        ns_mu, ns_var = dynamics_dist_fn(s_mu, s_var, a_mu[d, :])            
        return agg_reward+reward, ns_mu, ns_var, a_mu
    return _rollout_graph

def q(noise_dist, depth, rollout_fn):
    def _q(s, a_mu):
        """
        Compute the Q-function for a single restart
        """
        noise_mean, noise_var = noise_dist
        # augment state by adding variable for noise    
        s_mu = jnp.concatenate([s, noise_mean], axis=0)
        s_var = jnp.concatenate([s*0, noise_var], axis=0)

        init_rew = jnp.array([0.0])
        init_params = (init_rew, s_mu, s_var, a_mu)
        agg_rew, _, _, _ = jax.lax.fori_loop( 0, depth, rollout_fn, init_params)
        return agg_rew.sum()
    return _q
    
def grad_q(q):    
    def _grad_q(s, ac_mu):
        """
        Compute the gradient of Q-function for a single restart with actions
        """
        return jax.grad(q, argnums=(1), allow_int=True)(s, ac_mu)
    return _grad_q

#####################################
# Dynamics Distribution Fn
#####################################

# No variance mode
def dynamics_nv(ns_fn, env, noise_dist, n_real_var, n_bin_var):
    def _dynamics_nv(s_mu, s_var, a_mu):
        ns = ns_fn(s_mu, a_mu, env)
        
        noise_mean, noise_var = noise_dist
        ns_mu = jnp.concatenate([ns, noise_mean], axis=0)
        ns_var = jnp.concatenate([jnp.zeros_like(ns), noise_var], axis=0)

        return ns_mu, ns_var
    return _dynamics_nv

# Complete Mode
def dynamics_comp(ns_fn, env, fop_fn, sop_fn, noise_dist, n_real_var, n_bin_var):
    def _dynamics_comp(s_mu, s_var, a_mu):
        ns = ns_fn(s_mu, a_mu, env)
        
        fop_wrt_s, fop_wrt_ac = fop_fn(s_mu, a_mu)
        sop_wrt_s, sop_wrt_ac = sop_fn(s_mu, a_mu)

        # Taylor's expansion
        ns_mu = ns + 0.5*(jnp.multiply(sop_wrt_s, s_var).sum(axis=1))
        ns_var = jnp.multiply(jnp.square(fop_wrt_s), s_var).sum(axis=1)

        noise_mean, noise_var = noise_dist
        ns_mu = jnp.concatenate([ns_mu, noise_mean], axis=0)
        ns_var = jnp.concatenate([ns_var, noise_var], axis=0)

        return ns_mu, ns_var
    return _dynamics_comp

#####################################
# Reward distribution fn
#####################################

def reward_mean(reward_fn, env):
    def _reward_mean(s_mu, s_var, ac_mu):
        return reward_fn(s_mu, ac_mu, env)
    return _reward_mean

def reward_comp(reward_fn, env):
    def _reward_comp(s_mu, s_var, ac_mu):
        reward_mu = reward_fn(s_mu, ac_mu, env)
        def _diag_hessian(wrt):
            hess = jax.hessian(reward_fn, wrt)(s_mu, ac_mu, env)
            return jax.numpy.diagonal(hess, axis1=0, axis2=1)
        sop_wrt_s = _diag_hessian(0)
        sop_wrt_ac = _diag_hessian(1)

        return reward_mu + 0.5*(jnp.multiply(sop_wrt_s, s_var).sum(axis=0))
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