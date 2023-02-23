import numpy as np
from planners.discrete_disprod import DiscreteDisprod
from planners.continuous_disprod import ContinuousDisprod
from planners.continuous_disprod_hybrid import ContinuousSogbofaHybrid

from planners.baseline import shooting_cem
DEGREE_TO_RADIAN_MULTIPLIER = np.pi/180


import rospy



def setup_planner(env, env_cfg , key):
    if env_cfg["alg"] in ["cem", "mppi"]:
        agent = shooting_cem.ShootingCEM(env, env_cfg , key)    
    elif env_cfg["alg"] == "sogbofa":
        if env_cfg["discrete"]:
            agent = DiscreteDisprod(env, env_cfg , key)
        else:
            agent = ContinuousSogbofaHybrid(env, env_cfg , key)
    return agent


def sogbofa_plan_one_step_dubins_car(agent , env , state , goal):
    x , y , theta = state[:3] 
    agent.set_goal(goal_position=(goal[0], goal[1]))
    action, imagined_trajectory = agent.choose_action(np.array([x , y , theta]))
    logstring = f"Moving velocity,theta by {action}"
    return action, imagined_trajectory

def baseline_plan_one_step_dubins_car(agent , env , state , goal):
    x , y , theta = state[:3] 
    action , imagined_trajectory = agent.choose_action(np.array([x , y , theta]).astype('float64').reshape(1, -1, 1))
    action = action.reshape(-1, 1)
    logstring = f"Moving velocity,theta by {action}"
    return action , imagined_trajectory

def plan_one_step_dubins_car_w_velocity(agent , env , obs , goal):
    x , y, theta , old_velocity, old_angular_velocity = obs
    
    obs = np.array([round(x , 2) , round(y , 2), round(theta , 2) , old_velocity, old_angular_velocity])
    true_action , imagined_trajectory = agent.choose_action(obs)

    print(obs, true_action)
    
    delta_linear_velocity, delta_angular_velocity = true_action
    linear_velocity =  np.clip(old_velocity + delta_linear_velocity, env.min_velocity, env.max_velocity)
    angular_velocity = np.clip(old_angular_velocity + delta_angular_velocity * env.delta_angular_velocity_multiplier * DEGREE_TO_RADIAN_MULTIPLIER , env.min_angular_velocity , env.max_angular_velocity)


    action = np.array([linear_velocity, angular_velocity])
    return true_action , action , imagined_trajectory











