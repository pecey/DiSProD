#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
import rospy
import numpy as np
import sys
import os
from omegaconf import OmegaConf
import argparse
import json

DISPROD_PATH = os.getenv("DISPROD_PATH")
sys.path.append(DISPROD_PATH)
sys.path.append(os.path.join(DISPROD_PATH, "ros1-turtlesim"))
DISPROD_CONF_PATH = os.path.join(DISPROD_PATH, "config")

from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point

# note: this fails for local import so I moved it to global Python path
# from .planalg import planalg
from env.dubins_env_properties import EnvironmentProperties
from utils.common_utils import set_global_seeds
from planners.ros_interface import setup_planner
from planners.ros_interface import disprod_plan_one_step_dubins_car

DEGREE_TO_RADIAN_MULTIPLIER = np.pi / 180


class turtlewrapper():
    def __init__(self, debug=False, ser=None):
        rospy.init_node('turtlewrapper', anonymous=True)
        # Subscribers
        rospy.Subscriber('/turtle1/pose', Pose, self.pose_listener_callback)
        rospy.Subscriber('/turtlewrapper/GoalLocation', Point, self.goal_listener_callback)

        # Publishers
        self.cmd_pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)

        self.rate = rospy.Rate(1)  # hz ie 0.2 sec
        # self.timer = self.create_timer(timer_period, self.planner_callback)
        self.is_goal_set = False
        self.current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        self.agent = None
        self.env = None
        self.ac_seq = None
        self.key = None

    def goal_listener_callback(self, msg):
        rospy.loginfo('Goal Received: ("%f","%f")' % (msg.x, msg.y))
        self.goal_x = msg.x
        self.goal_y = msg.y
        self.is_goal_set = True
        self.ac_seq, self.key = self.agent.reset()

    def pose_listener_callback(self, msg):
        # rospy.loginfo('Pose Received: ("%f","%f")' %(msg.x,msg.y))
        self.current_pose = np.array([msg.x, msg.y, msg.theta, msg.linear_velocity, msg.angular_velocity])

    def planner_callback(self):
        # Either goal is not set or the current pose of bot is not set
        if not self.is_goal_set or self.current_pose is None:
            rospy.loginfo('Either goal not set or current pose is missing, nothing to do')
            return

        current_location, current_theta = self.current_pose[:2], self.current_pose[2]
        current_goal = np.array([self.goal_x, self.goal_y])

        dist = np.linalg.norm(current_goal - current_location)
        rospy.loginfo(f"Current goal : {current_goal}, Distance from goal : {dist}")

        if dist <= self.env.goal_boundary:
            rospy.loginfo('Arrived at Goal')
            rospy.loginfo(f"Goal: {self.goal_x, self.goal_y}, Location: {self.current_pose[0], self.current_pose[1]}")
            self.is_goal_set = False
            # If reached at goal, then send a no-op command
            cmd = generate_command_message([0, 0])
            self.cmd_pub.publish(cmd)
            return

        # Generate state using x,y,theta
        state = np.zeros(3)
        state[:2] = current_location
        state[2] = current_theta

        action, self.ac_seq, self.key, logstring = disprod_plan_one_step_dubins_car(self.agent, self.env, state, current_goal, self.ac_seq, self.key)
        action[1] *= DEGREE_TO_RADIAN_MULTIPLIER

        rospy.loginfo(logstring)
        cmd = generate_command_message(action)
        self.cmd_pub.publish(cmd)


def generate_command_message(action):
    cmd = Twist()
    cmd.linear.x = cmd.linear.y = cmd.linear.z = 0.0
    cmd.angular.x = cmd.angular.y = cmd.angular.z = 0.0
    cmd.linear.x = action[0]
    cmd.angular.z = action[1]
    return cmd


# Read configuration from file
def get_configuration(config_path, map_name):
    if config_path is not None:
        with open(config_path, 'r') as f:
            config_data = f.read()
        config_json = json.loads(config_data)
        maps = config_json['maps']
        selected_config = np.random.choice(maps) if map_name == "random" else maps[map_name]
        boundary = np.random.choice(config_json['boundary'])
        return selected_config, boundary
    return None


def update_with_args(env_cfg, args):
    keys_to_update = ["seed", "log_file", "naive"]
    for key in keys_to_update:
        if args.__contains__(key):
            if key == "naive":
                env_cfg[key] = getattr(args, key).lower() == "true"
            else:
                env_cfg[key] = getattr(args, key)

    env_cfg["map_name"] = args.map_name
    return env_cfg


def update_with_goal_and_obstacle_config(env_cfg, args):
    obstacles_config_path = f"{DISPROD_PATH}/env/assets/dubins.json"
    config, boundary = get_configuration(obstacles_config_path, args.map_name)
    env_cfg["config"] = config
    env_cfg["boundary"] = boundary
    return env_cfg


def prepare_config(planner, cfg_path=None):
    if planner == "naive":
        return dict()
    env_name = "continuous_dubins_car"
    planner_default_cfg = OmegaConf.load(f"{cfg_path}/planning/default.yaml")
    planner_env_cfg = OmegaConf.load(f"{cfg_path}/planning/{env_name}.yaml")
    return OmegaConf.merge(planner_default_cfg, planner_env_cfg, OmegaConf.load(f"{cfg_path}/{env_name}.yaml"))


def main(args):
    env_cfg = prepare_config(args.planner, DISPROD_CONF_PATH)
    env_cfg = update_with_args(env_cfg, args)
    env_cfg = update_with_goal_and_obstacle_config(env_cfg, args)
    env_cfg["naive"] = True if args.planner.lower() == "naive" else False
    env_cfg["map_name"] = args.map_name

    # If goal passed from command line, then overwrite
    if args.goal:
        env_cfg["config"]["goal_x"] = args.goal[0]
        env_cfg["config"]["goal_y"] = args.goal[1]

    # The turtlesim spawns at 5,5 by default.
    env_cfg["config"]["x"] = 5.544445
    env_cfg["config"]["y"] = 5.544445
    print(f"Running using the following config: {env_cfg}")
    set_global_seeds(env_cfg['seed'])

    tw = turtlewrapper()
    tw.env = EnvironmentProperties(env_cfg)
    tw.agent = setup_planner(tw.env, env_cfg)

    # Sync wrapper goal to config
    tw.goal_x = env_cfg["config"]["goal_x"]
    tw.goal_y = env_cfg["config"]["goal_y"]

    tw.is_goal_set = True
    print("Setup complete")

    while not rospy.is_shutdown():
        tw.planner_callback()
        tw.rate.sleep()


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--log_file', type=str, default=None)
        parser.add_argument('--seed', type=int, help='Seed for PRNG', default=42)
        parser.add_argument('--obstacles_config_file', type=str, help="Config filename without the JSON extension",
                            default="dubins")
        parser.add_argument('--planner', type=str, default="disprod")
        parser.add_argument('--goal', nargs=2, type=int, default=None)
        parser.add_argument('--map_name', type=str, help="Specify the map name to use", default="random")
        args = parser.parse_args()
        main(args)
    except rospy.ROSInterruptException:
        pass
