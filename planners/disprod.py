from utils.common_utils import print_, load_method
import jax
import jax.numpy as jnp
from functools import partial



class Disprod:
    def __init__(self, env, cfg, key):
        self.env = env
        self.ns_fn = load_method(cfg.get('transition_fn'))
        if cfg["env_name"] in ["sparse-continuous-mountain-car-v1", "simple-env-v1"]:
            self.reward_fn = partial(load_method(cfg.get('reward_fn')), sparsity_multiplier = cfg["reward_sparsity"])
        else:
            self.reward_fn = load_method(cfg.get('reward_fn'))

        self.nA = env.action_space.n if cfg["discrete"] else env.action_space.shape[0]
        self.nS = cfg.get("nS_out", env.observation_space.shape[0])
        
        self.max_grad_steps = cfg["disprod"]["max_grad_steps"]

        self.n_restarts = cfg["disprod"]["n_restarts"]
        self.depth = cfg.get("depth")
        self.step_size = cfg["disprod"]["step_size"]
        self.step_size_var = cfg["disprod"]["step_size_var"]
        self.debug = cfg.get("debug_planner")
        self.convergance_threshold = cfg.get("convergance_threshold")

        self.alpha = cfg.get("alpha")
            
        self.saved_restart_action = None
        self.last_chosen_action = None
        self.promising_restart = None

        self.log_file = cfg["log_file"]


    def set_goal(self, goal_position):
        self.env.goal_x, self.env.goal_y = goal_position