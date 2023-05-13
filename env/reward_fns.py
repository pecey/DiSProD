import math
import jax
import jax.numpy as jnp
import numpy as np

from utils.reward_utils import angle_normalize, euclidean_distance, get_number_of_collisions, greater_than_or_eq_to

#######################################
# Simple Env
########################################

def simple_env(state, action, env, sparsity_multiplier):
    x, y, noise = state
    return 1-2 * jax.nn.sigmoid(sparsity_multiplier * (euclidean_distance((x,y), (env.goal_x, env.goal_y))-env.goal_boundary))

#############################################
# Dubins Car
###############################################

# Reward = 100 if goal, 0 otherwise
# Reward = Reward - number of collisions
# Reward function for Continuous Dubins Car
def continuous_dubins_car(state, action, env):
    goal_x, goal_y, goal_boundary = env.goal_x, env.goal_y, env.goal_boundary
    x, y = state[0], state[1]
    distance_from_goal = euclidean_distance((x, y), (goal_x, goal_y))
    agent_outside_boundary = jnp.tanh(1 * jax.nn.relu(distance_from_goal - goal_boundary))
    not_done = agent_outside_boundary
    done = 1 - not_done
    return done - 100 * get_number_of_collisions(x, y, env)


# Reward function for Continuous Dubins Car
def continuous_dubins_car_w_velocity(state, action, env):
    goal_x, goal_y, goal_boundary = env.goal_x, env.goal_y, env.goal_boundary
    x, y = state[0] , state[1]
    distance_from_goal = euclidean_distance((x, y), (goal_x, goal_y))
    agent_outside_boundary = jnp.tanh(1 * jax.nn.relu(distance_from_goal - goal_boundary))
    not_done = agent_outside_boundary
    done = 1 - not_done
    return done - 100 * get_number_of_collisions(x, y, env)



# Leaky reward function for Continuous Dubins Car
def continuous_dubins_car_w_velocity_wo_obstacles(state, action, env):
    goal_x, goal_y, goal_boundary = env.goal_x, env.goal_y, env.goal_boundary
    x, y = state[0] , state[1]
    distance_from_goal = euclidean_distance((x, y), (goal_x, goal_y))
    #distance_from_goal = jnp.linalg.norm(jnp.array([x,y])-jnp.array([goal_x, goal_y]))
    agent_outside_boundary = jnp.tanh(1 * jax.nn.relu(distance_from_goal - goal_boundary))
    not_done = agent_outside_boundary
    done = 1 - not_done
    return done



def goal_reward(goal, x , y):
    goal_x, goal_y, goal_boundary = goal
    distance_from_goal = euclidean_distance((x, y), (goal_x, goal_y))
    agent_outside_boundary = jnp.tanh(1 * jax.nn.relu(distance_from_goal - goal_boundary))
    # If agent_outside_boundary = 1, or already_done = 0, then not_done = 1
    return 1 - agent_outside_boundary

###################################################
# Cartpole
###################################################

def cartpole(state, action, env):
    x, theta = state[0], state[2]
    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    part_a = jax.nn.relu(- x - x_threshold)
    part_b = jax.nn.relu(x - x_threshold)
    part_c = jax.nn.relu(- theta - theta_threshold_radians)
    part_d = jax.nn.relu(theta - theta_threshold_radians)
    done = part_a + part_b + part_c + part_d
    return 1.0-jnp.tanh(1 * done)

def cartpole_hybrid(state, action, env):
    x, theta, left_of_marker = state[0], state[2], state[4]
    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    part_a = jax.nn.relu(- x - x_threshold)
    part_b = jax.nn.relu(x - x_threshold)
    part_c = jax.nn.relu(- theta - theta_threshold_radians)
    part_d = jax.nn.relu(theta - theta_threshold_radians)
    done = part_a + part_b + part_c + part_d
    # reward = 1 * jax.nn.sigmoid(2.0*(env.unstable_left-x)) + 3 * jax.nn.sigmoid(2.0*(x-env.unstable_right))
    # If left_of_marker=1, then reward_multiplier = 1. If left_of_marker=0, then reward_multiplier = 3
    return (1.0-jnp.tanh(1 * done)) * (3.0 - 2.0 * left_of_marker) 

##########################################################
# Mountain Car
###########################################################

def mountain_car(state, action, env):
    beta = 1
    position, velocity = state[0], state[1]
    position_greater_than_goal = greater_than_or_eq_to(position, env.goal_position, sparsity_multiplier=beta)
    velocity_greater_than_goal = greater_than_or_eq_to(velocity, env.goal_velocity, sparsity_multiplier=beta)
    done = jnp.tanh(1 * position_greater_than_goal * velocity_greater_than_goal)
    reward = -1
    return ((1 - done) * reward)


# Reward function for continuous mountain car
def continuous_mountain_car(state, action, env):
    position, velocity = state[0], state[1]
    position_greater_than_goal = greater_than_or_eq_to(position, env.goal_position, sparsity_multiplier=1)
    velocity_grater_than_goal = greater_than_or_eq_to(velocity, env.goal_velocity, sparsity_multiplier=1)
    done = position_greater_than_goal * velocity_grater_than_goal
    reward =  ((done * 100) + (-jnp.square(action[0]) * 0.1))
    return reward

# Reward function for continuous mountain car with high dimensional action space.
def continuous_mountain_car_high_dim(state, action, env):
    position, velocity = state[0], state[1]
    position_greater_than_goal = greater_than_or_eq_to(position, env.goal_position, sparsity_multiplier=1)
    velocity_greater_than_goal = greater_than_or_eq_to(velocity, env.goal_velocity, sparsity_multiplier=1)
    done = position_greater_than_goal * velocity_greater_than_goal
    # Action penalty considers all the action
    reward_action = env.action_cost * jnp.sum(jnp.square(action))
    reward =  (done * 100) - reward_action
    return reward

# def continuous_mountain_car(state, action, env):
#     return continuous_mountain_car_sparse(state, action, env, sparsity_multiplier=1)

# Reward function for continuous mountain car where sparsity multiplier is controlled from the outside.
def continuous_mountain_car_sparse(state, action, env, sparsity_multiplier):
    position, velocity = state[0], state[1]
    position_greater_than_goal = greater_than_or_eq_to(position, env.goal_position, sparsity_multiplier)
    velocity_greater_than_goal = greater_than_or_eq_to(velocity, env.goal_velocity, sparsity_multiplier)
    done = position_greater_than_goal * velocity_greater_than_goal
    reward =  ((done * 100) + (-jnp.square(action[0]) * 0.1))
    return reward


# # Reward : 100 on done + -0.1 * action^2 for every timestep
# def continuous_mountain_car_strict(state, action, env):
#     position, velocity = state[0], state[1]
#     position_greater_than_goal = greater_than_or_eq_to(position, env.goal_position)
#     velocity_greater_than_goal = greater_than_or_eq_to(velocity, env.goal_velocity)
#     done = jnp.tanh(1e7 * (position_greater_than_goal * velocity_greater_than_goal))
#     reward = done * 100 + (-jnp.square(action[0]) * 0.1)
#     return reward


###############################################
# Pendulum
###############################################

# Reward function for pendulum
def pendulum(state, action, env):
    theta, thdot = state[0], state[3]  # th := theta
    u = jnp.clip(action[0], -env.max_torque, env.max_torque)
    costs = angle_normalize(theta) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
    return -costs