from tensorflow_probability.substrates import jax as tfp
import jax
import jax.numpy as jnp
from functools import partial
from utils.common_utils import load_method

tfd = tfp.distributions

# Adapted from: https://github.com/zchuning/latco/blob/6aab525b66efb8c99e55d6e0587a7bd31a599809/planners/shooting_cem.py
class ShootingCEMHybrid():
    def __init__(self, env, cfg, key):
        self.env = env
        self.alg = cfg['alg'].replace("hybrid_","")
        self.nA = cfg.get("nA", env.action_space.shape[0])
        if cfg["env_name"] in ["sparse-continuous-mountain-car-v1", "sparse-pusher-v2", "sparse-reacher-3d-v1"]:
            self.reward_fn = partial(load_method(cfg.get('reward_fn')), sparsity_multiplier = cfg["reward_sparsity"])
        else:
            self.reward_fn = load_method(cfg.get('reward_fn'))
        self.plan_fn = self.evaluate_mppi if self.alg == "mppi" else self.evaluate_cem
        self.plan_horizon = cfg["depth"]
        self.pop_size = cfg[self.alg]["n_samples"]
        self.optimization_steps = cfg[self.alg]['optimization_steps']
        self.mppi_gamma = cfg["mppi"]["gamma"]
        self._float = jnp.float32
        self.key = key
        self.elite_size = cfg["cem"]["elite_size"] 

        # This is only required for exp_samples
        if self.pop_size < self.elite_size:
            self.elite_size = self.pop_size

        self.alpha = cfg['alpha']
        if not cfg['nn_model']:
            self.dynamics_fn = load_method(cfg.get('transition_fn'))
            self.nS = cfg.get("nS", env.observation_space.shape[0])
            self.n_noise_var = 1
            self.preprocessing_fn = lambda x : x
        else:
            self.nS = cfg.get("nS_out")
            self.dynamics_fn = self.next_state_for_nn 
            self.n_noise_var = self.nS
            self.preprocessing_fn = load_method(cfg.get('preprocessing_fn'))

        self.batch_dynamics = jax.vmap(self.dynamics_fn, in_axes=(0, 0, None, None), out_axes=0)
        self.batch_rewards = jax.vmap(self.reward_fn, in_axes=(0, 0, None), out_axes=(0))
        self.batched_weighted_sample_fn = jax.vmap(lambda weight,sample: weight*sample, in_axes=(0, 0), out_axes=(0)) 

        self.ac_lb = env.action_space.low
        self.ac_ub = env.action_space.high
    
    def update_model(self, model):
        self.model = model
        
    def reset(self):
        self.prev_a_mean = jnp.tile((self.ac_lb + self.ac_ub)/2 , [self.plan_horizon, 1]) 
    
    def next_state_for_nn(self, obs, action, env, alpha):
        obs_wo_noise = obs[: self.nS]
        state_wo_noise = self.preprocessing_fn(obs_wo_noise)
        noise = obs[self.nS: ] * 0
        return self.model.next_state_(state_wo_noise,action, noise)  

    def find_next_obs(self, d , val):
        feats, actions, imagined_trajectory = val 
        feats_ = jnp.concatenate((feats, jnp.array([0])), 0)
        next_feats = self.dynamics_fn(feats_, actions[d, :], self.env, self.alpha)
        imagined_trajectory = imagined_trajectory.at[d].set(next_feats.squeeze())
        return next_feats , actions , imagined_trajectory   
   
    def eval_fitness_step(self, d, val):
        feats, actions, normal_noise_vec, uniform_noise_vec, agg_rewards = val
        current_actions = actions[d]
        normal_noise = normal_noise_vec[d]
        uniform_noise = uniform_noise_vec[d]
        feats_ = jnp.concatenate((feats, normal_noise, uniform_noise), 1)
        next_feats = self.batch_dynamics(feats_, current_actions, self.env, self.alpha)
        rewards = self.batch_rewards(next_feats, current_actions, self.env)
        return next_feats, actions, normal_noise_vec, uniform_noise_vec, agg_rewards + rewards


    def eval_fitness(self, obs, actions, key1, key2):
        feats = jnp.tile(obs, [self.pop_size, 1])
        agg_rewards = jnp.zeros([self.pop_size, ], dtype=self._float)
        noise = jax.random.normal(key1, [self.plan_horizon, self.pop_size, self.n_noise_var])
        
        # Sample uniform noise in [-0.5, 0.5]
        uniform_noise = jax.random.uniform(key2, [self.plan_horizon, self.pop_size, self.n_noise_var]) - 0.5

        # From (pop_size, plan_horizon, nA) to (plan_horizon, pop_size, nA)
        actions = actions.transpose(1, 0, 2)
        init_val = (feats, actions, noise, uniform_noise,agg_rewards)
        # for d in range(self.plan_horizon):
        # #    next_feats = self.batch_dynamics(feats, actions[:, idx, :], self.env)
        # #    assert next_feats.shape == feats.shape , (next_feats.shape , feats.shape)
        # #    # feats = jnp.expand_dims(feats, 2)
        # #    reward = self.batch_rewards(feats, actions[:, idx, :], None, self.env)[0]
        # #    agg_rewards += reward
        # #    feats = next_feats
        #     current_actions = actions[d]
        #     current_noise = noise[d]
        #     uniform_noise_mean = uniform_noise[d]
        #     feats_ = jnp.concatenate((feats, current_noise , uniform_noise_mean), 1)
        #     next_feats = self.batch_dynamics(feats_, current_actions, self.env, self.alpha)
        #     reward = self.batch_rewards(next_feats, current_actions, self.env)
        #     agg_rewards += reward
        #     feats = next_feats
            

        feats, _, _,_, agg_rewards = jax.lax.fori_loop(0, self.plan_horizon, self.eval_fitness_step, init_val)
        return agg_rewards, feats, uniform_noise

    # samples: (batch_size, horizon, nA)
    # fitness: (batch_size,)
    @partial(jax.jit, static_argnums=(0,))
    def mppi(self, d, val):
        obs, a_mean, a_var, key, _, _ = val

         # Bound variance
        lb_dist, ub_dist = a_mean - self.ac_lb, self.ac_ub - a_mean
        a_var = jnp.minimum(jnp.minimum(jnp.square(lb_dist/2), jnp.square(ub_dist/2)), a_var)
        a_std = jnp.sqrt(a_var)

        new_key, sub_key1, sub_key2, sub_key3 = jax.random.split(key, 4)
        # Sample action sequences and evaluate fitness
        noise = tfd.TruncatedNormal(jnp.zeros_like(a_mean), jnp.ones_like(a_var), -2, 2).sample(sample_shape=[self.pop_size], seed=sub_key1)
        samples = a_mean + noise * a_std

        fitness, tau, uniform_noise = self.eval_fitness(obs, samples, sub_key2, sub_key3)

        weights = jax.nn.softmax(self.mppi_gamma * fitness)
        new_a_mean = jnp.sum(self.batched_weighted_sample_fn(weights, samples), axis=0)
        new_a_var = jnp.sum(self.batched_weighted_sample_fn(weights, jnp.square(samples - new_a_mean)), axis=0)

        return obs, new_a_mean, new_a_var, new_key, tau, uniform_noise
    
    # samples: (batch_size, horizon, nA)
    @partial(jax.jit, static_argnums=(0,))
    def cem(self, d, val):
        obs, a_mean, a_var, key, _, _ = val

        # Bound variance
        lb_dist, ub_dist = a_mean - self.ac_lb, self.ac_ub - a_mean
        a_var = jnp.minimum(jnp.minimum(jnp.square(lb_dist/2), jnp.square(ub_dist/2)), a_var)
        a_std = jnp.sqrt(a_var)

        new_key, sub_key1, sub_key2, sub_key3 = jax.random.split(key, 4)

        # Shape: (pop_size, plan_horizon, nA)
        noise = tfd.TruncatedNormal(jnp.zeros_like(a_mean), jnp.ones_like(a_var), -2, 2).sample(sample_shape=[self.pop_size], seed=sub_key1)
        # samples = jnp.tile(a_mean, [self.pop_size, 1, 1]) + noise * jnp.tile(a_std, [self.pop_size, 1, 1])
        samples = a_mean + noise * a_std
        # samples_ = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[self.pop_size], seed=sub_key1)
        # samples = jnp.clip(samples_, self.ac_lb, self.ac_ub)
        fitness, tau, uniform_noise = self.eval_fitness(obs, samples, sub_key2, sub_key3)
        # Choose elite samples and compute new means and vars
        elite_values, elite_inds = jax.lax.top_k(jnp.squeeze(fitness), self.elite_size)
        elite_samples = samples[elite_inds]
        new_a_mean = jnp.mean(elite_samples, axis=0)
        new_a_var = jnp.var(elite_samples, axis=0)

        return obs, new_a_mean, new_a_var, new_key, tau, uniform_noise

    def evaluate_cem(self, obs, a_mean, a_var, key):
        init_val = (obs, a_mean, a_var, key, jnp.zeros([self.pop_size, self.nS])*0.0, jnp.zeros([self.plan_horizon, self.pop_size, self.n_noise_var])*0.0)
        return jax.lax.fori_loop(0, self.optimization_steps, self.cem, init_val)

    def evaluate_mppi(self, obs, a_mean, a_var, key):
        init_val = (obs, a_mean, a_var, key)
        return jax.lax.fori_loop(0, self.optimization_steps, self.mppi, init_val)

    def choose_action(self, obs):
        obs = (obs.astype(jnp.float32))

        # Shape: (plan_horizon, nA)
        init_mean = self.prev_a_mean
        init_var = jnp.tile(jnp.square(self.ac_ub - self.ac_lb)/16, [self.plan_horizon, 1])

        # tau is the vector of state variables computed by CEM at t=planning horizon (Just the last time step)
        # uniform_noise is the vector of uniform noise variables sampled for t=planning horizon
        _, mean, _, self.key, tau, uniform_noise = self.plan_fn(obs, init_mean, init_var, self.key)
        
        action, self.prev_a_mean = mean[0], jnp.concatenate((mean[1:], jnp.zeros((1, self.nA))), axis=0)
                
        assert(action.shape == (self.nA,))
        act_pred = mean
        
        imagined_trajectory = jnp.zeros((self.plan_horizon , obs.shape[0]))
        init_val = [obs.reshape(-1), act_pred, imagined_trajectory]
        for i in range(len(act_pred)):
            feat_pred , _ , imagined_trajectory = self.find_next_obs(i , init_val)            
            init_val = [feat_pred.reshape(-1) , act_pred , imagined_trajectory]

        action = mean[0]
        assert(action.shape == (self.nA,))


        # imagined_trajectory_w_variance = jnp.zeros((2 , self.plan_horizon , obs.shape[0]))
        # imagined_trajectory_w_variance.at[0].set(imagined_trajectory[:])
        return action, imagined_trajectory