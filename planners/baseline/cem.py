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

        self.alg = cfg['alg']
        self.cfg = cfg
                   
        self.plan_horizon = cfg["depth"]
        self.pop_size = cfg[self.alg]["n_samples"]
        self.optimization_steps = cfg[self.alg]['optimization_steps']
        self.mppi_gamma = cfg["mppi"]["gamma"]
        self._float = jnp.float32


        self.elite_size = cfg["cem"]["elite_size"]

        # This is only required for exp_samples
        if self.pop_size < self.elite_size:
            self.elite_size = self.pop_size

        self.alpha = cfg.get('alpha', 0)

        self.ac_lb = env.action_space.low
        self.ac_ub = env.action_space.high

        self.a_dist_fn = self.a_dist
    

        self.n_noise_var = 1

        self.batch_dynamics = jax.vmap(self.ns_fn, in_axes=(0, 0, None), out_axes=(0))
        self.batch_rewards = jax.vmap(self.reward_fn, in_axes=(0, 0, None), out_axes=(0))
        self.batched_weighted_sample_fn = jax.vmap(lambda weight,sample: weight*sample, in_axes=(0, 0), out_axes=(0)) 


    def reset(self, key):
       ac_seq = jnp.tile((self.ac_lb + self.ac_ub)/2, [self.plan_horizon, 1])
       return ac_seq, key


    def a_dist(self, a_mean, a_var, key):
        lb_dist, ub_dist = a_mean - self.ac_lb, self.ac_ub - a_mean
        a_var = jnp.minimum(jnp.minimum(jnp.square(lb_dist/2), jnp.square(ub_dist/2)), a_var)
        a_std = jnp.sqrt(a_var)

        # Shape: (pop_size, plan_horizon, nA)
        noise = tfd.TruncatedNormal(loc=jnp.zeros_like(a_mean), scale=jnp.ones_like(a_var), low=[-2.0], high=[2.0])
        noise = noise.sample(sample_shape=[self.pop_size], seed=key)

        return a_mean, a_std, noise
    

    @partial(jax.jit, static_argnums=(0,))
    def choose_action(self, obs, ac_seq, key):        
        # Shape: (plan_horizon, nA)
        init_mean = ac_seq
        init_var = jnp.tile(jnp.square(self.ac_ub - self.ac_lb)/16, [self.plan_horizon, 1])
        plan_horizon = self.plan_horizon
        pop_size = self.pop_size
        opt_steps = self.optimization_steps
        elite_size = self.elite_size
        a_dist_fn = self.a_dist_fn
        batch_dynamics_fn = self.batch_dynamics
        batch_rewards_fn = self.batch_rewards

        def _eval_fitness_step(d, val):
            feats, actions, noise, agg_rewards = val
            current_actions = actions[d]
            current_noise = noise[d]
            feats_ = jnp.concatenate((feats, current_noise), 1)

            next_feats = batch_dynamics_fn(feats_, current_actions, self.env)
            rewards = batch_rewards_fn(feats_, current_actions, self.env)
            return next_feats, actions, noise, agg_rewards + rewards
        
        def _eval_fitness(obs, actions, key):
            """
            obs: Vector of observed state variables
            actions: Array of action sequences. Shape: (pop_size, plan_horizon, nA)
            key: PRNG key for sampling noise
            """
            feats = jnp.tile(obs, [pop_size, 1])
            agg_rewards = jnp.zeros([pop_size], dtype=jnp.float32)
            noise_norm = jax.random.normal(key, [plan_horizon, pop_size, 1])
            # noise_uni = jax.random.uniform(key, [plan_horizon, pop_size, 1])
            noise = jnp.concatenate([noise_norm], axis = 2)

            # From (pop_size, plan_horizon, nA) to (plan_horizon, pop_size, nA)
            actions = actions.transpose(1, 0, 2)
            init_val = (feats, actions, noise, agg_rewards)

            feats, _, _, agg_rewards = jax.lax.fori_loop(0, plan_horizon, _eval_fitness_step, init_val)
            return agg_rewards, feats
        
        def _cem(d, val):
            obs, a_mean, a_var, key = val

            # Bound variance
            new_key, sub_key1, sub_key2 , sub_key3 = jax.random.split(key, 4)
            a_mean, a_std, noise = a_dist_fn(a_mean, a_var, sub_key1)

            samples = a_mean + a_std * noise
            fitness, _ = _eval_fitness(obs, samples, sub_key2)

            # Choose elite samples and compute new means and vars
            elite_values, elite_inds = jax.lax.top_k(jnp.squeeze(fitness), elite_size)
            elite_samples = samples[elite_inds]
            new_a_mean = jnp.mean(elite_samples, axis=0)
            new_a_var = jnp.var(elite_samples, axis=0)

            return obs, new_a_mean, new_a_var, new_key

        def _evaluate_cem(obs, a_mean, a_var, key):
            init_val = (obs, a_mean, a_var, key)
            return jax.lax.fori_loop(0, opt_steps, _cem, init_val)

        _, mean, _, key = _evaluate_cem(obs, init_mean, init_var, key)

        ac, ac_seq = mean[0], jnp.concatenate((mean[1:], jnp.zeros((1, self.nA))), axis=0)
        return ac, ac_seq, key