from tensorflow_probability.substrates import jax as tfp
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from utils.common_utils import print_, load_method


tfd = tfp.distributions

# Adapted from: https://github.com/zchuning/latco/blob/6aab525b66efb8c99e55d6e0587a7bd31a599809/planners/shooting_cem.py


class ShootingCEM():
    def __init__(self, env, cfg):
        self.env = env
        self.ns_fn = load_method(cfg.get('transition_fn'))
        if cfg["env_name"] in ["sparse-continuous-mountain-car-v1", "simple-env-v1"]:
            self.reward_fn = partial(load_method(cfg.get('reward_fn')), sparsity_multiplier = cfg["reward_sparsity"])
        else:
            self.reward_fn = load_method(cfg.get('reward_fn'))

        self.nA = env.action_space.n if cfg["discrete"] else env.action_space.shape[0]
        self.nS = cfg.get("nS_out", env.observation_space.shape[0])


        self.batch_dynamics = jax.vmap(self.ns_fn, in_axes=(0, 0, None), out_axes=(0))
        self.batch_rewards = jax.vmap(self.reward_fn, in_axes=(0, 0, None), out_axes=(0))
        self.batched_weighted_sample_fn = jax.vmap(lambda weight,sample: weight*sample, in_axes=(0, 0), out_axes=(0)) 

        self.alg = cfg['alg']
        self.cfg = cfg
                   
        self.plan_horizon = cfg["depth"]
        self.pop_size = cfg[self.alg]["n_samples"]
        self.opt_steps = cfg[self.alg]['optimization_steps']
        self.mppi_gamma = cfg["mppi"]["gamma"]
        self._float = jnp.float32


        self.elite_size = cfg["cem"]["elite_size"]

        # This is only required for exp_samples
        if self.pop_size < self.elite_size:
            self.elite_size = self.pop_size

        self.alpha = cfg.get('alpha', 0)

        self.ac_lb = env.action_space.low
        self.ac_ub = env.action_space.high


        self.n_bin_var = cfg.get("n_bin_var", 0)

        if self.n_bin_var == 0:
            noise_gen_fn = gen_norm_noise(self.plan_horizon, self.pop_size)
        else:
            noise_gen_fn = gen_norm_uni_noise(self.plan_horizon, self.pop_size)    

        a_dist_fn = a_dist(self.ac_lb, self.ac_ub, self.pop_size)
        eval_fitness_step_fn = eval_fitness_step(self.batch_dynamics, self.batch_rewards, env)
        eval_fitness_fn = eval_fitness(eval_fitness_step_fn, noise_gen_fn, self.pop_size, self.plan_horizon)

        if self.alg == "cem":
            self.plan_fn = evaluate_cem(a_dist_fn, eval_fitness_fn, self.elite_size, self.opt_steps)
        elif self.alg == "mppi":
            self.plan_fn = evaluate_mppi(a_dist_fn, eval_fitness_fn, self.batched_weighted_sample_fn, self.mppi_gamma, self.opt_steps)
        else:
            raise Exception(f"Unexpected value received for --alg. Got {cfg['alg']}, was expecting cem or mppi")
        

    def reset(self, key):
       ac_seq = jnp.tile((self.ac_lb + self.ac_ub)/2, [self.plan_horizon, 1])
       return ac_seq, key
    

    @partial(jax.jit, static_argnums=(0,))
    def choose_action(self, obs, ac_seq, key):        
        # Shape: (plan_horizon, nA)
        init_mean = ac_seq
        init_var = jnp.tile(jnp.square(self.ac_ub - self.ac_lb)/16, [self.plan_horizon, 1])
        _, mean, _, key = self.plan_fn(obs, init_mean, init_var, key)
        ac, ac_seq = mean[0], jnp.concatenate((mean[1:], jnp.zeros((1, self.nA))), axis=0)
        return ac, ac_seq, key


# Function to generate action sequences
def a_dist(ac_lb, ac_ub, pop_size):
    def _a_dist(a_mean, a_var, key):
        lb_dist, ub_dist = a_mean - ac_lb, ac_ub - a_mean
        a_var = jnp.minimum(jnp.minimum(jnp.square(lb_dist/2), jnp.square(ub_dist/2)), a_var)
        a_std = jnp.sqrt(a_var)

        # Shape: (pop_size, plan_horizon, nA)
        noise = tfd.TruncatedNormal(loc=jnp.zeros_like(a_mean), scale=jnp.ones_like(a_var), low=[-2.0], high=[2.0])
        noise = noise.sample(sample_shape=[pop_size], seed=key)

        return a_mean, a_std, noise
    return _a_dist

def gen_norm_noise(plan_horizon, pop_size):
    def _gen_norm_noise(key):
        return jax.random.normal(key, [plan_horizon, pop_size, 1])
    return _gen_norm_noise

def gen_norm_uni_noise(plan_horizon, pop_size):
    def _gen_norm_uni_noise(key):
        key1, key2 = jax.random.split(key)
        noise_norm = jax.random.normal(key1, [plan_horizon, pop_size, 1])
        noise_uni = jax.random.uniform(key2, [plan_horizon, pop_size, 1])
        return jnp.concatenate([noise_norm, noise_uni], axis = 2)
    return _gen_norm_uni_noise
        

# Function to evaluate next state and reward for a batch of actions.
def eval_fitness_step(batch_dynamics_fn, batch_rewards_fn, env):
    def _eval_fitness_step(d, val):
        feats, actions, noise, agg_rewards = val
        current_actions = actions[d]
        current_noise = noise[d]
        feats_ = jnp.concatenate((feats, current_noise), 1)

        next_feats = batch_dynamics_fn(feats_, current_actions, env)
        rewards = batch_rewards_fn(feats_, current_actions, env)
        return next_feats, actions, noise, agg_rewards + rewards
    return _eval_fitness_step

# Function to evaluate the fitness of a batch of actions.
def eval_fitness(eval_fitness_step_fn, noise_gen_fn, pop_size, plan_horizon):    
    def _eval_fitness(obs, actions, key):
        """
        obs: Vector of observed state variables
        actions: Array of action sequences. Shape: (pop_size, plan_horizon, nA)
        key: PRNG key for sampling noise
        """
        feats = jnp.tile(obs, [pop_size, 1])
        agg_rewards = jnp.zeros([pop_size], dtype=jnp.float32)

        # Generate noise variables for the "sampling" from the transition fn
        noise = noise_gen_fn(key)
        
        # From (pop_size, plan_horizon, nA) to (plan_horizon, pop_size, nA)
        actions = actions.transpose(1, 0, 2)
        init_val = (feats, actions, noise, agg_rewards)

        feats, _, _, agg_rewards = jax.lax.fori_loop(0, plan_horizon, eval_fitness_step_fn, init_val)
        return agg_rewards, feats
    return _eval_fitness
    

def evaluate_cem(a_dist_fn, eval_fitness, elite_size, opt_steps):
    def _evaluate_cem(obs, a_mean, a_var, key):
        def _cem(d, val):
            obs, a_mean, a_var, key = val

            # Generate candidate action sequences
            new_key, sub_key1, sub_key2 , sub_key3 = jax.random.split(key, 4)
            a_mean, a_std, noise = a_dist_fn(a_mean, a_var, sub_key1)
            samples = a_mean + a_std * noise

            # Compute fitness
            fitness, _ = eval_fitness(obs, samples, sub_key2)

            # Choose elite samples and compute new means and vars
            elite_values, elite_inds = jax.lax.top_k(jnp.squeeze(fitness), elite_size)
            elite_samples = samples[elite_inds]
            new_a_mean = jnp.mean(elite_samples, axis=0)
            new_a_var = jnp.var(elite_samples, axis=0)

            return obs, new_a_mean, new_a_var, new_key
        
        init_val = (obs, a_mean, a_var, key)
        return jax.lax.fori_loop(0, opt_steps, _cem, init_val)
    return _evaluate_cem

def evaluate_mppi(a_dist_fn, eval_fitness, batched_weighted_sample_fn, gamma, opt_steps):
    def _evaluate_mppi(obs, a_mean, a_var, key):
        def _mppi(d, val):
            obs, a_mean, a_var, key = val

            # Generate candidate action sequences
            new_key, sub_key1, sub_key2 , sub_key3 = jax.random.split(key, 4)
            a_mean, a_std, noise = a_dist_fn(a_mean, a_var, sub_key1)
            samples = a_mean + noise * a_std

            # Compute fitness
            fitness, _ = eval_fitness(obs, samples, sub_key2)

            # Compute weighted average
            weights = jax.nn.softmax(gamma * fitness)
            new_a_mean = jnp.sum(batched_weighted_sample_fn(weights, samples), axis=0)
            new_a_var = jnp.sum(batched_weighted_sample_fn(weights, jnp.square(samples - new_a_mean)), axis=0)

            return obs, new_a_mean, new_a_var, new_key
        
        init_val = (obs, a_mean, a_var, key)
        return jax.lax.fori_loop(0, opt_steps, _mppi, init_val)
    
    return _evaluate_mppi

    
