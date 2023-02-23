"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R

Continuous version copied from https://github.com/facebookresearch/mbrl-lib/blob/master/mbrl/env/cartpole_continuous.py
"""

import math

import gym
import numpy as np
from gym import logger, spaces
from gym.utils import seeding
from utils.reward_utils import is_between_smooth
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))


class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": [
        "human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, alpha=0.0, ignore_shaky_in_planner=False):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.theta_shaky_threshold = 2 * math.pi / 360

        self.x_threshold = 2.4

        # Region to the left of this is low reward region
        self.reward_marker = 0.5
        self.alpha = alpha
        
        self.ignore_bin_var_in_planner = ignore_shaky_in_planner
        
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        low = np.array(
            [
                -self.x_threshold * 2,
                -np.finfo(np.float32).max,
                -self.theta_threshold_radians * 2,
                -np.finfo(np.float32).max,
                0
            ],
            dtype=np.float32,
        )

        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                1
            ],
            dtype=np.float32,
        )

        act_high = np.array((1,), dtype=np.float32)
        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = action.squeeze()
        x, x_dot, theta, theta_dot, left_of_marker = self.state

        # Add noise from 0-1 normal
        noise = np.random.normal(loc=0, scale=1)

        shaky_action = action 
        force = shaky_action * self.force_mag
        force += self.alpha*noise

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole *
                           costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        uniform_noise = np.random.uniform(-0.5, 0.5)

        # If x is to left of reward marker, then is_left_of_marker = 1, else 0
        is_left_of_marker = x < self.reward_marker #or x > self.unstable_right
        p_left_of_marker = is_left_of_marker
        left_of_marker = 1 if uniform_noise <= (p_left_of_marker - 0.5) else 0

        self.state = (x, x_dot, theta, theta_dot, left_of_marker)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
             reward = 1.0 * left_of_marker + 3.0 * (1-left_of_marker)
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0 * left_of_marker + 3.0 * (1-left_of_marker)
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        state_real = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.state = np.append(state_real, 1)
        return self.state

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.track.add_attr(rendering.LineWidth(5))
            self.viewer.add_geom(self.track)

            self.low_reward_region = rendering.Line(
                ((-2.4) * scale + screen_width/2.0, carty), ((self.reward_marker) * scale + screen_width/2.0, carty))
            self.low_reward_region.add_attr(rendering.LineWidth(5))
            self.low_reward_region.set_color(255, 0, 0)
            self.viewer.add_geom(self.low_reward_region)
            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def set_imagined_trajectory_data(self, imaginary_data):
        if len(imaginary_data.shape) == 3:
            expectation_x = imaginary_data[0][:, 0]
            variance_x = imaginary_data[1][:, 0]
        else:
            expectation_x = imaginary_data[0]
            variance_x = np.zeros(expectation_x.shape)

        plt.clf()
        plt.axvline(self.reward_marker, color="red")
        # plt.axvline(self.unstable_right, color="red")
        plt.xlabel("X")
        plt.ylabel("Depth")

        plt.fill_betweenx(range(len(expectation_x)), expectation_x -
                          variance_x, expectation_x + variance_x, color="red")
        #plt.plot(expectation_x , )

        plt.pause(0.01)
