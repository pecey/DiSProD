import env.reward_fns, env.transition_fns
import torch as T
import numpy as np


class FakeDubinsEnv:
    def __init__(self, goal_x, goal_y, goal_boundary):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_boundary = goal_boundary
        self.obstacles = []
        self.time_interval = 0.5

        self.default_velocity = 1
        self.turning_velocity = 0.2
        self.angular_velocity = 30

        self.min_y_position = 0
        self.min_x_position = 0
        self.max_y_position = 10
        self.max_x_position = 10


class FakeMountainCarEnv:
    def __init__(self, goal_position, goal_velocity):
        self.goal_position = goal_position
        self.goal_velocity = goal_velocity

class FakeCartPoleEnv:
    def __init__(self, x_threshold, theta_threshold):
        self.x_threshold = x_threshold
        self.theta_threshold_radians = theta_threshold

# Tests for utility methods
def test_position_should_be_geq_target():
    position = T.tensor([0.3, 0.5, 0.7])
    target = 0.5
    result = env.reward_fns.greater_than_or_eq_to(position, target)
    expected_result = T.tensor([0, 1, 1])

    assert result.flatten().shape == expected_result.flatten().shape
    assert T.all(T.eq(result, expected_result))


def test_position_should_be_within_limits():
    position = T.tensor([0.3, 0.5, 0.7])
    lower_limit = 0.4
    upper_limit = 0.6
    result = env.reward_fns.is_between_limits(position, lower_limit, upper_limit)
    expected_result = T.tensor([0, 1, 0])

    assert result.flatten().shape == expected_result.flatten().shape
    assert T.all(T.eq(result, expected_result))


def test_point_is_within_the_rectangular_patch():
    obstacle = {"x": 0.4, "y": 0, "width": 0.2, "height": 1}
    x = T.tensor([0.3, 0.5, 0.7])
    y = T.tensor([0, 0, 0])
    result = env.reward_fns.detect_collision_with_rectangular_obstacle(x, y, obstacle)
    expected_result = T.tensor([0, 1, 0])

    assert result.flatten().shape == expected_result.flatten().shape
    assert T.all(T.eq(result, expected_result))


# Tests for reward functions
############################################################################
# Env: CartPole
# state: x, _, theta, _
def test_cartpole_should_not_get_reward_if_pole_is_out_of_x_threshold():
    state = T.tensor([[2.41, 0, 0, 0],
                      [2.4, 0, 0, 0],
                      [2.39, 0, 0, 0],
                      [-2.39, 0, 0, 0],
                      [-2.4, 0, 0, 0],
                      [-2.41, 0, 0, 0]
                      ])
    already_done = T.zeros((len(state), 1))
    fake_env = FakeCartPoleEnv(x_threshold=2.4, theta_threshold=10)
    reward, _ = env.reward_fns.cartpole_strict(state, None, already_done, fake_env)
    expected_reward = T.tensor([[0], [1], [1], [1], [1], [0]])
    assert reward.shape == expected_reward.shape
    assert T.all(T.eq(reward, expected_reward))

############################################################################
# Env: Dubins Car
def test_dubins_should_get_reward_if_current_state_within_the_goal_boundary():
    goal = (5, 5)
    goal_boundary = 1
    state = T.tensor([[5, 5, 0, 0],
                      [5, 6, 0, 0],
                      [5, 4, 0, 0],
                      [4, 5, 0, 0],
                      [6, 5, 0, 0]])
    already_done = T.zeros((len(state), 1))

    fake_env = FakeDubinsEnv(goal_x=goal[0], goal_y=goal[1], goal_boundary=goal_boundary)
    reward, _ = env.reward_fns.dubins_car_strict(state, None, already_done, fake_env)
    expected_reward = 100 * T.ones_like(already_done)

    assert reward.shape == expected_reward.shape
    assert T.all(T.eq(reward, expected_reward))


# All the states are outside the boundary
def test_dubins_should_not_get_reward_if_current_state_outside_the_goal_boundary():
    goal = (5, 5)
    goal_boundary = 1
    state = T.tensor([[5, 6.1, 0, 0],
                      [5, 3.9, 0, 0],
                      [3.9, 5, 0, 0],
                      [6.1, 5, 0, 0]])
    already_done = T.zeros((len(state), 1))

    fake_env = FakeDubinsEnv(goal_x=goal[0], goal_y=goal[1], goal_boundary=goal_boundary)
    reward, _ = env.reward_fns.dubins_car_strict(state, None, already_done, fake_env)
    expected_reward = T.zeros_like(already_done)

    assert reward.shape == expected_reward.shape
    assert T.all(T.eq(reward, expected_reward))


# All the states get a reward of 100 irrespective of the fact if they are within the boundary or not, since already_done is 1
def test_dubins_should_continue_to_get_reward_if_already_done():
    goal = (5, 5)
    goal_boundary = 1
    states_within_goal = T.tensor([[5, 5, 0, 0],
                                   [5, 6, 0, 0],
                                   [5, 4, 0, 0],
                                   [4, 5, 0, 0],
                                   [6, 5, 0, 0]])
    states_outside_goal = T.tensor([[5, 6.1, 0, 0],
                                    [5, 3.9, 0, 0],
                                    [3.9, 5, 0, 0],
                                    [6.1, 5, 0, 0]])
    state = T.cat((states_within_goal, states_outside_goal))
    already_done = T.ones((len(state), 1))

    fake_env = FakeDubinsEnv(goal_x=goal[0], goal_y=goal[1], goal_boundary=goal_boundary)
    reward, _ = env.reward_fns.dubins_car_strict(state, None, already_done, fake_env)
    expected_reward = 100 * T.ones_like(already_done)

    assert reward.shape == expected_reward.shape
    assert T.all(T.eq(reward, expected_reward))


############################################################################
# Env: Continuous Dubins
# All the states are within the boundary
def test_cont_dubins_should_get_reward_if_current_state_within_the_goal_boundary():
    goal = (5, 5)
    goal_boundary = 1
    state = T.tensor([[5, 5, 0, 0],
                      [5, 6, 0, 0],
                      [5, 4, 0, 0],
                      [4, 5, 0, 0],
                      [6, 5, 0, 0]])
    already_done = T.zeros((len(state), 1))

    fake_env = FakeDubinsEnv(goal_x=goal[0], goal_y=goal[1], goal_boundary=goal_boundary)
    reward, _ = env.reward_fns.continuous_dubins_car_strict(state, None, already_done, fake_env)
    expected_reward = 100 * T.ones_like(already_done)

    assert reward.shape == expected_reward.shape
    assert T.all(T.eq(reward, expected_reward))


# All the states are outside the boundary
def test_cont_dubins_current_state_is_outside_the_goal_boundary():
    goal = (5, 5)
    goal_boundary = 1
    state = T.tensor([[5, 6.1, 0, 0],
                      [5, 3.9, 0, 0],
                      [3.9, 5, 0, 0],
                      [6.1, 5, 0, 0]])
    already_done = T.zeros((len(state), 1))

    fake_env = FakeDubinsEnv(goal_x=goal[0], goal_y=goal[1], goal_boundary=goal_boundary)
    reward, _ = env.reward_fns.continuous_dubins_car_strict(state, None, already_done, fake_env)
    expected_reward = T.tensor([0, 0, 0, 0])

    assert reward.flatten().shape == expected_reward.flatten().shape
    assert T.all(T.eq(reward, expected_reward))


# All the states get a reward of 100 irrespective of the fact if they are within the boundary or not, since already_done is 1
def test_cont_dubins_should_continue_to_get_reward_if_already_done():
    goal = (5, 5)
    goal_boundary = 1
    states_within_goal = T.tensor([[5, 5, 0, 0],
                                   [5, 6, 0, 0],
                                   [5, 4, 0, 0],
                                   [4, 5, 0, 0],
                                   [6, 5, 0, 0]])
    states_outside_goal = T.tensor([[5, 6.1, 0, 0],
                                    [5, 3.9, 0, 0],
                                    [3.9, 5, 0, 0],
                                    [6.1, 5, 0, 0]])
    state = T.cat((states_within_goal, states_outside_goal))
    already_done = T.ones((len(state), 1))

    fake_env = FakeDubinsEnv(goal_x=goal[0], goal_y=goal[1], goal_boundary=goal_boundary)
    reward, _ = env.reward_fns.continuous_dubins_car_strict(state, None, already_done, fake_env)
    expected_reward = 100 * T.ones_like(already_done)

    assert reward.shape == expected_reward.shape
    assert T.all(T.eq(reward, expected_reward))


############################################################################
# Env: Mountain Car
def test_mount_car_should_get_reward_only_if_position_and_velocity_greater_than_goal():
    position = T.tensor([0, -0.5, 0.5, -0.2])
    velocity = T.tensor([0, 0, -0.2, -0.2])
    state = T.stack((position, velocity, T.ones_like(position))).T
    fake_env = FakeMountainCarEnv(goal_position=0, goal_velocity=0)
    reward, _ = env.reward_fns.mountain_car(state, None, T.zeros((len(state), 1)), fake_env)
    expected_reward = T.tensor([0, -1, -1, -1]).view(-1, 1)

    assert reward.shape == expected_reward.shape
    assert T.all(T.eq(reward, expected_reward))


def test_mount_car_should_get_reward_irrespective_of_position_and_velocity_if_already_done():
    position = T.tensor([0, -0.5, 0.5, -0.2])
    velocity = T.tensor([0, 0, -0.2, -0.2])
    state = T.stack((position, velocity, T.ones_like(position))).T
    fake_env = FakeMountainCarEnv(goal_position=0, goal_velocity=0)
    reward, _ = env.reward_fns.mountain_car(state, None, T.ones((len(state), 1)), fake_env)
    expected_reward = T.tensor([0, 0, 0, 0]).view(-1, 1)

    assert reward.shape == expected_reward.shape
    assert T.all(T.eq(reward, expected_reward))


############################################################################
# Env: Continuous Mountain Car
def test_cont_mount_car_should_get_reward_only_if_position_and_velocity_greater_than_goal():
    position = T.tensor([0, -0.5, 0.5, -0.2])
    velocity = T.tensor([0, 0, -0.2, -0.2])
    state = T.stack((position, velocity, T.ones_like(position))).T
    action = T.tensor([0, 0.2, 0.4, -0.2]).view(-1, 1)
    fake_env = FakeMountainCarEnv(goal_position=0, goal_velocity=0)
    reward, _ = env.reward_fns.continuous_mountain_car(state, action, T.zeros((len(state), 1)), fake_env)
    expected_reward = T.tensor([100, 0, 0, 0]).view(-1, 1) + -T.pow(action, 2) * 0.1

    assert reward.shape == expected_reward.shape
    assert T.all(T.eq(reward, expected_reward))


def test_cont_mount_car_should_get_reward_irrespective_of_position_and_velocity_if_already_done():
    position = T.tensor([0, -0.5, 0.5, -0.2])
    velocity = T.tensor([0, 0, -0.2, -0.2])
    state = T.stack((position, velocity, T.ones_like(position))).T
    action = T.tensor([0, 0.2, 0.4, -0.2]).view(-1, 1)
    fake_env = FakeMountainCarEnv(goal_position=0, goal_velocity=0)
    reward, _ = env.reward_fns.continuous_mountain_car(state, action, T.ones((len(state), 1)), fake_env)
    expected_reward = (T.tensor([100, 100, 100, 100]).view(-1, 1) + -T.pow(action, 2) * 0.1).view(-1, 1)

    assert reward.shape == expected_reward.shape
    assert T.all(T.eq(reward, expected_reward))
