#!/usr/bin/env python3
import argparse
import tf

import rospy
import numpy as np
import sys
from omegaconf import OmegaConf
import os
import json

DISPROD_PATH = os.getenv("DISPROD_PATH")
sys.path.append(DISPROD_PATH)
sys.path.append(os.path.join(DISPROD_PATH, "ros1-sogbofa-turtlebot"))
sys.path.append(os.path.join(DISPROD_PATH, "ros1-sogbofa-turtlebot/catkin_ws/src"))
DISPROD_CONF_PATH = os.path.join(DISPROD_PATH, "config")
DISPROD_MOD_PATH = os.path.join(DISPROD_PATH, "ros1-sogbofa-turtlebot/catkin_ws/sdf_models")

from visualization_helpers.marker_array_rviz import PoseArrayRviz
from geometry_msgs.msg import Twist, Point, TwistStamped
from nav_msgs.msg import Odometry
from math import atan2, asin
from std_srvs.srv import Empty

from visualization_msgs.msg import Marker, MarkerArray

# note: this fails for local import so I moved it to global Python path
# from .planalg import planalg
from planners.ros_interface import setupAgent
from planners.ros_interface import sogbofa_plan_one_step_dubins,  pidOneStep

from visualization_msgs.msg import Marker, MarkerArray


DEGREE_TO_RADIAN_MULTIPLIER = np.pi / 180
DEFAULT_CONFIG = {"boundary": [
    {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10, "boundary_width": 0.5}
],
    "config": [
        {
            "x": 5, "y": 5,
            "goal_x": 3, "goal_y": 4,
            "obstacles": [{"x": 2, "y": 4, "width": 1, "height": 0.5},
                          {"x": 4, "y": 4, "width": 1, "height": 0.5},
                          {"x": 6, "y": 4, "width": 1, "height": 0.5}]
        }]
}


class TurtleBotWrapper:
    def __init__(self, debug=False, ser=None):
        rospy.init_node('turtlebotwrapper', anonymous=True)
        # Subscribers
        rospy.Subscriber('/odom', Odometry, self.pose_listener_callback)

        self.track_pub_1 = rospy.Publisher('/track1', Marker, queue_size=1)
        self.track_pub_2 = rospy.Publisher('/track2', Marker, queue_size=1)

        # Publishers
        self.goal_pub = rospy.Publisher('/goal_marker', MarkerArray, queue_size=2)
        self.goal_markers = MarkerArray()
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_cmd_pub = rospy.Publisher('/odom_cmd_pub' , Odometry , queue_size = 100)
        self.obstacle_pub = rospy.Publisher('/obstacles', MarkerArray, queue_size=10)
        self.obstacle_markers = MarkerArray()
        

        self.myrate = 5
        self.rate = rospy.Rate(self.myrate)  # hz ie 0.2 sec
        # self.timer = self.create_timer(timer_period, self.planner_callback)
        self.use_sogbofa = False
        self.use_feedback_control = False
        self.use_pid = False
        self.is_goal_set = False

        self.nS = 4
        self.nA = 2
        self.current_pose = None
        self.action_time = 0
        self.count = 1
        self.last_linear_vel , self.last_angular_vel = 0,0

        self.agent = None
        self.env = None

        self.goal_xs = []
        self.goal_ys = []

        #self.goal_xs = [ 1.59 , 1.53 , 1.54 , -0.10 , -0.01, -1.5 , -2.4 , -4.9 , -8.3 , -11,2]
        #self.goal_ys = [ -3.16 , -4.47 , -5.9 , -7.6 , -8.7 ,-9.4 , -8.6 , -8.7 , -7.5 , -5.25]
        self.goal_no = 1

    def divide_ellipse(self , N = 10):
        def distance(x1,y1,x2,y2):
            return np.sqrt((x2-x1)**2 + (y2-y1)**2)
        x0 , y0 = -7.5 , 0
        a , b = (13+10.60)/2 , (7.90 + 4.60)/2
        from math import radians
        angle = 0
        d = 0
        while angle <= 360:
            x = a * np.cos(radians(angle))
            y = b * np.sin(radians(angle))
            d += distance(x0,y0,x,y)
            x0 = x
            y0 = y
            angle += 0.25

        print("Circumference is ", d)
        alpha = 1.57 + 0.78

        rotation = np.array([[np.cos(alpha) , -np.sin(alpha)], [np.sin(alpha) , np.cos(alpha)]])


        divide = d/N
        angle = 0
        centre_x , centre_y = -7.5 , 0
        x0 , y0 = 0 , 0
        angle0 = 0
        for i in range(N):
            dist = 0
            while(dist<divide):
                x = a * np.cos(radians(angle))
                y = b * np.sin(radians(angle))
                dist += distance(x0,y0,x,y)
                x0 = x
                y0 = y
                angle += 0.25

            rotated = np.dot(rotation , np.array([x , y]))
            self.goal_xs.append(rotated[0] + centre_x)
            self.goal_ys.append(rotated[1] + centre_y)

        self.goal_xs = self.goal_xs[::-1][N//2:]
        self.goal_ys = self.goal_ys[::-1][N//2:]

        print(self.goal_xs)
        print(self.goal_ys)
    

    # Publish goal marker
    def publish_goal_marker(self):
        goals = [self.create_goal_marker(gx , gy,idx) for idx,(gx , gy) in enumerate(zip(self.goal_xs , self.goal_ys))]
        self.goal_markers.markers.extend(goals)
        self.goal_pub.publish(self.goal_markers)

    # Publish the obstacles
    def publish_obstacle_markers(self, mode=0):
        obstacles = self.create_obstacle_markers(mode)
        for obstacle in obstacles:
            self.obstacle_markers.markers.append(obstacle)
        self.obstacle_pub.publish(self.obstacle_markers)

    # Create the goal marker
    def create_goal_marker(self, gx , gy , idx):
        goal = Marker()
        goal.id = idx
        goal.type = Marker.CYLINDER
        goal.header.frame_id = "odom"
        goal.action = Marker.ADD
        goal.ns = "Goal"
        goal.pose.orientation.w = 1.0
        goal.scale.x, goal.scale.y, goal.scale.z = 0.2, 0.2, 0.1
        goal.color.g = 1
        goal.color.a = 0.5
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.position.z = 0
        return goal

    def render_model(self, x, y , yaw):
         
        os.system(
            'rosrun gazebo_ros spawn_model -urdf -model turtlebot3_burger -x {} -y {} -Y {} -param robot_description'.format(
                x, y, yaw))

    
    def pose_listener_callback(self, msg):
        state = {'x_pos': msg.pose.pose.position.x, 'y_pos': msg.pose.pose.position.y,
                 'z_pos': msg.pose.pose.position.z}

        self.current_odom_msg = msg

        q0 = msg.pose.pose.orientation.w
        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z

        state['roll'] = atan2(2 * q2 * q3 + 2 * q0 * q1, q3 * q3 - q2 * q2 - q1 * q1 + q0 * q0)
        state['pitch'] = -asin(2 * q1 * q3 - 2 * q0 * q2)
        state['yaw'] = atan2(2 * q1 * q2 + 2 * q0 * q3, q1 * q1 + q0 * q0 - q3 * q3 - q2 * q2)

        state['q0'] = q0
        state['q1'] = q1
        state['q2'] = q2
        state['q3'] = q3

        # Update the world frame velocity
        state['x_vel'] = msg.twist.twist.linear.x
        state['y_vel'] = msg.twist.twist.linear.y
        state['z_vel'] = msg.twist.twist.linear.z

        self.current_pose = state

    def create_track_markers(self , env):

        outer_dim_a = 13
        outer_dim_b = 7.90
        inner_dim_a = 10.60
        inner_dim_b = 4.60
        mid_offset = 0.3
    
        centre_x , centre_y, alpha = -7.5 , 0 , 1.57 + 0.78


        inner_track = create_track_marker(centre_x , centre_y , alpha, inner_dim_a , inner_dim_b , self.track_pub_1, c = [0,0,1 , 1])
        
        outer_track = create_track_marker(centre_x , centre_y , alpha , outer_dim_a , outer_dim_b , self.track_pub_2, c = [0,1,0 , 0.2])

    def planner_callback(self):
        # Either goal is not set or the current pose of bot is not set
        if not self.is_goal_set or self.current_pose is None:
            rospy.loginfo('Either goal not set or current pose is missing, nothing to do')
            return

        current_state = np.array([self.current_pose['x_pos'], self.current_pose['y_pos'], self.current_pose['yaw'] , self.last_linear_vel , self.last_angular_vel])
        current_goal = np.array([self.env.goal_x, self.env.goal_y])

        dist = np.linalg.norm(current_goal - current_state[:2])

        print("Distance to intermediate goal is ", dist)
        print(f"Current goal {current_goal} current state {current_state[:2]}")
    

        if dist <= 1:
            if self.goal_no < len(self.goal_xs):
                print("changing goal")
                input()
                self.env.goal_x = self.goal_xs[self.goal_no]
                self.env.goal_y = self.goal_ys[self.goal_no]

                print(self.env.goal_x , self.env.goal_y)
            self.goal_no += 1
            

        if self.use_sogbofa:
            
            time1 = rospy.Time().now().to_sec()

            log_string, action = sogbofa_plan_one_step_dubins(self.agent, self.env, current_state , current_goal)

            self.last_linear_vel , self.last_angular_vel = action
            
            time2 = rospy.Time().now().to_sec()
            rospy.loginfo("action {} is generated after {} secs".format(action , time2 - time1))
            action[1] *= DEGREE_TO_RADIAN_MULTIPLIER
            
            cmd = generate_command_message(action)

            current_odom_msg = self.current_odom_msg
            
        

        self.cmd_pub.publish(cmd)
        self.odom_cmd_pub.publish(current_odom_msg)

        
def generate_stamped_command_message(cmd,count):
    # for visualization
    stamped_cmd = TwistStamped()
    stamped_cmd.twist = cmd
    stamped_cmd.header.seq = count 
    stamped_cmd.header.frame_id = "odom"

    
    return stamped_cmd


def generate_command_message(action):
    
    cmd = Twist()
    cmd.linear.x = cmd.linear.y = cmd.linear.z = 0.0
    cmd.angular.x = cmd.angular.y = cmd.angular.z = 0.0
    cmd.linear.x = action[0] if type(action[0]) != np.int64 else 0
    cmd.angular.z = float(action[1])

    print(cmd.linear.x , cmd.angular.z)
    print(type(cmd.linear.x) , type(cmd.linear.z))
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

    if getattr(args, "obstacles_config_file").lower() == "none":
        env_cfg["obstacles_config_path"] = None
    else:
        env_cfg["obstacles_config_path"] = f"{DISPROD_PATH}/env/assets/{args.obstacles_config_file}.json"
        env_cfg["map_name"] = args.map_name
    return env_cfg

def create_track_marker(h , k , alpha , scale_x , scale_y , pub , c):
    r , g , b , a = c
    
    q_new = tf.transformations.quaternion_from_euler(0, 0, alpha)  
    
    goal = Marker()
    goal.id = 0
    goal.type = Marker.CYLINDER
    goal.header.frame_id = "odom"
    goal.action = Marker.ADD
    goal.pose.orientation.x = q_new[0]
    goal.pose.orientation.y= q_new[1]
    goal.pose.orientation.z = q_new[2]
    goal.pose.orientation.w = q_new[3]
    goal.scale.x, goal.scale.y, goal.scale.z = scale_x * 2, scale_y * 2, 0.0
    goal.color.r = r 
    goal.color.g = g
    goal.color.b = b
    goal.color.a = a
    goal.pose.position.x = h 
    goal.pose.position.y = k
    goal.pose.position.z = 0.1
    pub.publish(goal)


def update_with_goal_and_obstacle_config(env_cfg, args):
    if getattr(args, "obstacles_config_file").lower() == "none":
        env_cfg["config"] = DEFAULT_CONFIG["config"]
        env_cfg["boundary"] = DEFAULT_CONFIG["boundary"]
    else:
        obstacles_config_path = f"{DISPROD_PATH}/env/assets/{args.obstacles_config_file}.json"
        config, boundary = get_configuration(obstacles_config_path, args.map_name)
        env_cfg["config"] = config
        env_cfg["boundary"] = boundary
    return env_cfg


def prepare_config(planner, cfg_path=None, model = "clip_dubins_car"):

    if planner == "naive":
        return dict()
    env_name = "dubins_car" if planner == "sogbofa" else "continuous_dubins_car_w_velocity"
    planner_default_cfg = OmegaConf.load(f"{cfg_path}/planning/default.yaml")
    sogbofa_default_cfg = OmegaConf.load(f"{cfg_path}/sogbofa_default.yaml")
    planner_env_cfg = OmegaConf.load(f"{cfg_path}/planning/{env_name}.yaml")
    return OmegaConf.merge(planner_default_cfg, sogbofa_default_cfg, planner_env_cfg, OmegaConf.load(f"{cfg_path}/{env_name}.yaml"))


def main(args):
    device = "cpu"

    print("From main , the clip model is ", args.model)

    env_cfg = prepare_config(args.planner, DISPROD_CONF_PATH, args.model)
    env_cfg = update_with_args(env_cfg, args)
    env_cfg["device"] = device
    env_cfg["naive"] = True if args.planner.lower() == "naive" else False
    depth , restart = env_cfg["depth"], env_cfg["n_restarts"]

    if args.poseVisualization:
        pose_array_viz = PoseArrayRviz(depth, restart)
    else:
        pose_array_viz = None

    tw = TurtleBotWrapper()
    tw.divide_ellipse(N = 15)
    
    # Goal is set in env at this point.
    # Any changes to the goal is reflected in env when the planner_callback calls the plan function.
    if args.model == "clip_dubins_car":
        from env.dubins_env_properties_w_velocity import EnvironmentProperties
        tw.env = EnvironmentProperties(env_cfg)
    else:
        from env.dubins_env_properties import EnvironmentProperties
        tw.env = EnvironmentProperties(env_cfg)

    tw.render_model(x = 0 , y = 0 , yaw = -2.35)

    tw.create_track_markers(tw.env)

    tw.use_sogbofa = True
    env_cfg["learning"] = False
    tw.agent = setupAgent(tw.env, env_cfg, pose_array_viz = pose_array_viz)

    # Sync wrapper goal to config
    

    tw.is_goal_set = True

    tw.env.goal_x = tw.goal_xs[1]
    tw.env.goal_y = tw.goal_ys[1]

    tw.publish_goal_marker()

    
    while not rospy.is_shutdown():
        tw.planner_callback()
        tw.create_track_markers(tw.env)
        tw.publish_goal_marker()
        tw.rate.sleep()


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--log_file', type=str, default=None)
        parser.add_argument('--seed', type=int, help='Seed for PRNG', default=42)
        parser.add_argument('--obstacles_config_file', type=str, help="Config filename without the JSON extension",
                            default="none")
        parser.add_argument('--planner', type=str, default="sogbofa")
        parser.add_argument('--goal', nargs=2, type=int, default=None)
        parser.add_argument('--model', type=str, default="clip_dubins_car")

        
        parser.add_argument('--poseVisualization', type=bool, default=True)
        args = parser.parse_args()
        main(args)
    except rospy.ROSInterruptException:
        pass
