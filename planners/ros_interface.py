import numpy as np
from planners.discrete_disprod import DiscreteDisprod
from planners.continuous_disprod import ContinuousDisprod
from planners.baseline import cem

DEGREE_TO_RADIAN_MULTIPLIER = np.pi/180

def setup_planner(env, env_cfg):
    if env_cfg["alg"] in ["cem", "mppi"]:
        agent = cem.ShootingCEM(env, env_cfg)    
    elif env_cfg["alg"] == "disprod":
        if env_cfg["discrete"]:
            agent = DiscreteDisprod(env, env_cfg)
        else:
            agent = ContinuousDisprod(env, env_cfg)
    return agent

# In this model, the action is delta linear velocity and delta angular velocity.
# But we have to send linear velocity and angular velocity to the bot.
def plan_one_step_dubins_car(agent, env, obs, delta_ac_seq, key):
    x, y, theta, old_velocity, old_angular_velocity = obs
    
    obs = np.array([round(x, 2), round(y, 2), round(theta, 2), old_velocity, old_angular_velocity])
    delta_ac, delta_ac_seq, state_seq, key = agent.choose_action(obs, delta_ac_seq, key)
    
    delta_linear_velocity, delta_angular_velocity = delta_ac
    linear_velocity =  np.clip(old_velocity + delta_linear_velocity, env.min_velocity, env.max_velocity)
    angular_velocity = np.clip(old_angular_velocity + delta_angular_velocity * env.delta_angular_velocity_multiplier * DEGREE_TO_RADIAN_MULTIPLIER, env.min_angular_velocity, env.max_angular_velocity)

    ac = np.array([linear_velocity, angular_velocity])
    return delta_ac, ac, delta_ac_seq, state_seq, key











