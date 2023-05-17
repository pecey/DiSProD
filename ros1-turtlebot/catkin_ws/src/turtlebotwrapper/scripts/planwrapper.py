#!/usr/bin/env python3

import argparse
import rospy
import numpy as np
import sys
import os
import xml.etree.ElementTree as ET
from omegaconf import OmegaConf
import jax

DISPROD_PATH = os.getenv("DISPROD_PATH")
sys.path.append(DISPROD_PATH)
sys.path.append(os.path.join(DISPROD_PATH, "ros1-turtlebot"))
DISPROD_CONF_PATH = os.path.join(DISPROD_PATH, "config")
DISPROD_MOD_PATH = os.path.join(DISPROD_PATH, "ros1-turtlebot/catkin_ws/sdf_models")

from visualization_helpers.marker_array_rviz import PoseArrayRviz

from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from math import atan2, asin
from tracking_pid.msg import states, state

from visualization_msgs.msg import Marker, MarkerArray

from planners.ros_interface import setup_planner 
from utils.common_utils import print_, set_global_seeds, prepare_config, update_config_with_args, setup_output_dirs, load_method

DEGREE_TO_RADIAN_MULTIPLIER = np.pi / 180

class LowLevelController():
    def __init__(self) -> None:
        super().__init__()

    def register(self , msg_type , topic_name , controller):
        self.publisher = rospy.Publisher('/' + topic_name, msg_type, queue_size=10)
        self.controller = controller
        
    def publish(self, msg):
        self.publisher.publish(msg)


class TurtleBotWrapper:
    def __init__(self, cfg, control_type, odom_msg, skip_waypoints, frame = "/odom"):
        rospy.init_node('turtlebotwrapper', anonymous=True, disable_signals = True)
        
        # Subscribers
        rospy.Subscriber(odom_msg, Odometry, self.pose_listener_callback)
        
        self.track_pub_1 = rospy.Publisher('/track1', Marker, queue_size=1)
        self.skip_waypoints = skip_waypoints

        self.odom_msg = frame

        self.publisher = LowLevelController()
        
        # Self controller controls the bot itself.
        # PID controller issues waypoints to PID which in-turn controls the bot.
        if control_type == "self":
            print("Registering self controller")
            self.publisher.register(Twist , "cmd_vel" , "self")
        else:
            print("Registering PID controller")
            self.publisher.register(states , "states_to_be_followed" , "pid")

        # Publishers
        self.goal_pub = rospy.Publisher('/goal_marker', Marker, queue_size=2)
        self.goal_markers = MarkerArray()
        self.obstacle_pub = rospy.Publisher('/obstacles', MarkerArray, queue_size=10)
        #self.waypoint_pub = rospy.Publisher('/states_to_be_followed', states, queue_size=10)
        self.obstacle_markers = MarkerArray()
        self.start_pub = rospy.Publisher('/start_marker' , MarkerArray , queue_size=2)
        self.start_markers = MarkerArray()
        self.pub_rate = 10
        self.fixed_time_mode = cfg['fixed_time_pub_mode']
        self.vehicle_model = cfg['vehicle_model']
        if self.fixed_time_mode == True:
            print("Starting the fixed pub mode")
            self.action_generated = False
            self.timer = rospy.Timer(rospy.Duration(1/self.pub_rate), self.publisher_callback)
        self.pose_received = False
        self.rate = rospy.Rate(self.pub_rate)  # hz ie 0.2 sec
        
        self.pose = None
        self.last_linear_vel, self.last_angular_vel = 0,0

        self.planner = None
        self.env = None

    # Publish the goal marker.
    def publish_goal_marker(self):
        goal = self.create_marker(self.env.goal_x, self.env.goal_y, c=[1, 0, 0])
        self.goal_markers.markers.append(goal)
        self.goal_pub.publish(goal)

    # Publish the start marker.
    def publish_start_marker(self):
        goal = self.create_marker(self.env.x , self.env.y)
        self.start_markers.markers.append(goal)
        self.start_pub.publish(self.start_markers)
    
    # Publish the obstacles
    def publish_obstacle_markers(self, render_gazebo=False):
        obstacles = self.create_obstacle_markers(render_gazebo)
        for obstacle in obstacles:
            self.obstacle_markers.markers.append(obstacle)
        self.obstacle_pub.publish(self.obstacle_markers)

    # Create markers
    def create_marker(self, x, y, c=[0, 0, 1]):
        goal = Marker()
        goal.id = 0
        goal.type = Marker.CYLINDER
        goal.header.frame_id = "odom"
        goal.action = Marker.ADD
        goal.ns = "Goal"
        goal.pose.orientation.w = 1.0
        goal.scale.x, goal.scale.y, goal.scale.z = 1, 1, 0.1
        goal.color.r = c[0]
        goal.color.b = c[1]
        goal.color.g = c[2]
        goal.color.a = 0.5
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0
        return goal
        

    def render_model(self, vehicle_type, x, y, yaw=0.0):
        os.system("rosservice call gazebo/delete_model '{model_name: turtlebot3_burger}'")
        os.system(
            'rosrun gazebo_ros spawn_model -urdf -model turtlebot3_burger -x {} -y {} -Y {} -param robot_description'.format(
                x, y, yaw))

    def shutdownHook(self):       
        os.system("rosservice call gazebo/delete_model '{model_name: turtlebot3_burger}'")
        
        for idx in range(self.obstacle_length):
            name = f"model_name : box_target_red_{idx}"
            command = "rosservice call gazebo/delete_model " + "\'{" + name + "}\'"
            os.system(command)

    def render_in_gazebo(self, idx, pose_x=0, pose_y=0, pose_z=0, size_x=0.5, size_y=0.5, size_z=0.5):
        rospy.loginfo("Rendering object in gazebo")

        model = 'box'
        model_path = os.path.join(DISPROD_MOD_PATH, model + '.sdf')

        tree = ET.parse('{}'.format(model_path))
        root = tree.getroot()

        for pose in root.iter('pose'):
            pose.text = '{} {} {} 0 0 0'.format(pose_x, pose_y, pose_z)

        for size in root.iter('size'):
            size.text = '{} {} {}'.format(size_x, size_y, size_z)

        file_name = '{}_pose_{}_{}_size_{}_{}_{}.sdf'.format(model, pose_x, pose_y, size_x, size_y, size_z)

        output_file_path = os.path.join(DISPROD_MOD_PATH, file_name)
        tree.write(output_file_path)

        os.system("rosrun gazebo_ros spawn_model -file {} -sdf -model box_target_red_{}".format(output_file_path, idx))

        os.remove(output_file_path)

    def create_obstacle_markers(self, render_gazebo=False):
        if not self.env.obstacles:
            return list()

        # Create obstacle cubes using the config
        obstacles = list()
        # Create cubes
        for idx, obstacle in enumerate(self.env.obstacles):
            obst = Marker()
            obst.type = Marker.CUBE
            obst.header.frame_id = "odom"
            obst.action = Marker.ADD
            obst.ns = "Obstacles"
            obst.color.r, obst.color.a = 1, 1
            obst.id = idx + 1

            obst.scale.x = obstacle["width"]
            obst.scale.y = obstacle["height"]
            obst.scale.z = 1
            obst.pose.orientation.x = obst.pose.orientation.y = obst.pose.orientation.z = 0
            obst.pose.orientation.w = 1
            obst.pose.position.x = obstacle["x"] + obstacle["width"]/2
            obst.pose.position.y = obstacle["y"] + obstacle["height"]/2

            if render_gazebo:
                self.render_in_gazebo(idx, pose_x=obst.pose.position.x, pose_y=obst.pose.position.y, pose_z=0,
                                      size_x=obstacle["width"], size_y=obstacle["height"], size_z=0.5)

            obst.pose.position.z = 0
            obstacles.append(obst)

        self.obstacle_length = len(obstacles)
        self.action = []
        return obstacles

    # Listen to odometry and update the robot's position for the planner.
    def pose_listener_callback(self, msg):
        state = {'x_pos': msg.pose.pose.position.x, 'y_pos': msg.pose.pose.position.y,
                 'z_pos': msg.pose.pose.position.z}

        self.pose_received = True
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

        state['ang_vel'] = msg.twist.twist.angular.z

        self.pose = state

    # Call the planner with the current state and publish command/waypoint accordingly.
    def planner_callback(self, step_num, ac_seq, key):
        # Either goal is not set or the current pose of bot is not set
        if not self.pose_received:
            rospy.loginfo('Either goal not set or current pose is missing, nothing to do')
            return 0

        self.publish_goal_marker()
        state = np.array([self.pose['x_pos'], self.pose['y_pos'], self.pose['yaw'], self.last_linear_vel , self.last_angular_vel])[:self.env.nS]
        goal = np.array([self.env.goal_x, self.env.goal_y])

        time1 = rospy.Time().now().to_sec()

        _, ac, ac_seq, state_seq, key = self.plan_one_step(self.planner, self.env, state, ac_seq , key)
            
        dist = ((state[0] - goal[0])**2 + (state[1] - goal[1])**2)**0.5
        
        print(f"Distance to goal {dist}")
        if dist < 1:
            cmd = generate_command_message([0 , 0])
            #self.publisher.publish(cmd)
            rospy.sleep(0.2)
            rospy.loginfo("Reached goal, nothing more to do")
            return step_num, ac_seq, key

        self.action_generated = True
        self.action_cache = ac

        if self.publisher.controller == "self":
            cmd = generate_command_message(ac)
            self.publisher.publish(cmd)
        else:
            waypoints = self.gen_waypoints_pid(state_seq[:])
            self.publisher.publish(waypoints)

        self.last_linear_vel , self.last_angular_vel = ac[0] , ac[1]
        
        time2 = rospy.Time().now().to_sec()
        print(f"Action generated after time {time2 - time1}")
        
        return step_num + 1, ac_seq, key

    # Generate waypoints for PID. Use every nth state where n=self.skip_waypoints
    def gen_waypoints_pid(self, state_seq):      
        imagined_state_arr = []
        for imagined_state in state_seq[::self.skip_waypoints]:
            msg = state()
            msg.x = imagined_state[0]
            msg.y = imagined_state[1]
            msg.yaw = imagined_state[2]
            msg.ux = imagined_state[3]
            msg.utheta = imagined_state[4]

            imagined_state_arr.append(msg)

        full_msg = states()
        full_msg.states = imagined_state_arr.copy()

        return full_msg

    def publisher_callback(self, timer):  
        if not self.action_generated:
            return
        rospy.loginfo("Sending cmd_vel at a fixed rate")
        linear_velocity, angular_velocity = self.action_cache
        linear_velocity =  np.clip(linear_velocity, self.env.min_velocity, self.env.max_velocity)
        angular_velocity = np.clip(angular_velocity , self.env.min_angular_velocity , self.env.max_angular_velocity)

        cmd = generate_command_message([linear_velocity , angular_velocity])
        self.publisher.publish(cmd)

    def state_reader(self):
        return [self.pose['x_pos'], self.pose['y_pos'], self.pose['yaw'] , self.pose['x_vel'] , self.pose['ang_vel']]

    # This can be used to plot the trajectory that the planner thinks it will take
    def imag_traj_pub(self, states):
        self.pose_array_viz.publish(states)
        
    def planner_reset(self):
        self.planner.reset()

def generate_stamped_command_message(cmd,count):
    # for visualization
    stamped_cmd = TwistStamped()
    stamped_cmd.twist = cmd
    stamped_cmd.header.seq = count 
    stamped_cmd.header.frame_id = "world"
    return stamped_cmd

def generate_command_message(action):
    cmd = Twist()
    cmd.linear.x = cmd.linear.y = cmd.linear.z = 0.0
    cmd.angular.x = cmd.angular.y = cmd.angular.z = 0.0
    cmd.linear.x = max(0 , action[0])
    cmd.angular.z = action[1]
    return cmd

def prepare_config(env_name, cfg_path=None):
    planner_default_cfg = OmegaConf.load(f"{cfg_path}/default.yaml")
    planner_env_cfg = OmegaConf.load(f"{cfg_path}/{env_name}.yaml")
    return OmegaConf.merge(planner_default_cfg, planner_env_cfg)

def main(args):
    if args.env_name != "continuous_dubins_car":
        raise Exception(f"Planwrapper is only intended for continuous_dubins_car. Got {args.env_name}")
    cfg = prepare_config(args.env_name, DISPROD_CONF_PATH)
    cfg['mode'] = 'tbot_evaluation'
    cfg = update_config_with_args(cfg, args , base_path=DISPROD_PATH)
    
    set_global_seeds(cfg['seed'])
    setup_output_dirs(cfg, cfg["run_name"] , DISPROD_PATH)

    odom_msg = "/odom" if args.vehicle_type == "turtlebot" else "/odometry/filtered"
    frame = "odom" if args.vehicle_type == "turtlebot" else "odom"

    tw = TurtleBotWrapper(cfg=cfg, 
                          control_type=args.control, 
                          frame=frame, 
                          odom_msg=odom_msg, 
                          skip_waypoints=args.skip_waypoints)
    
    tw.env = load_method(cfg['env_file'])(cfg) 

    if args.pose_viz:
        tw.pose_array_viz = PoseArrayRviz(cfg["depth"], cfg["disprod"]["n_restarts"])
    else:
        tw.pose_array_viz = None 

    tw.plan_one_step = load_method(cfg['ros_interface'])    
    
    # Set up the planner using the environment and env_cfg
    tw.planner = setup_planner(tw.env, cfg)
    key = jax.random.PRNGKey(args.seed)
    ac_seq, key = tw.planner.reset(key)

    # Render the model with the specified vehicle type and coordinates
    tw.render_model(args.vehicle_type, tw.env.x, tw.env.y)

    # Treat the boundaries of the environment as obstacles
    tw.env.boundaries_as_obstacle()

    # Publish markers for the goal position, start position, and obstacles
    tw.publish_goal_marker()
    tw.publish_start_marker()
    tw.publish_obstacle_markers(render_gazebo=True)

    step_num = 0

    rospy.on_shutdown(tw.shutdownHook)
    
    # tw.publish_goal_marker()
    # tw.publish_start_marker()
    # tw.publish_obstacle_markers(render_gazebo=1)
        
    while not rospy.is_shutdown():
        curr_step_num, ac_seq, key = tw.planner_callback(step_num, ac_seq, key)
    
        # Check if the current step number has changed
        if step_num == curr_step_num:
            print(f"{cfg['map_name']}, {step_num}", cfg['log_file'])
            rospy.signal_shutdown("Environment is solved")

        # Check if the step number has exceeded the timeout limit
        if step_num > 400:
            rospy.signal_shutdown("Environment Timeout")
            print_(f"{cfg['map_name']}, 400", cfg['log_file'])
            raise TimeoutError
        
        step_num = curr_step_num
    
        # Sleep for a fraction of a second (to control the loop frequency)
        rospy.sleep(1/100)

        

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, help='Seed for PRNG', default=42)
        # Passing env_name as args to as to reuse the same helper functions
        parser.add_argument('--env_name', type=str, default= "continuous_dubins_car" , help='Note: we are using the same configurations for the boat experiments')
        parser.add_argument('--alg', type=str, default="disprod" , choices = ['mppi' , 'cem' , 'disprod'])
        parser.add_argument('--pose_viz', type=bool, default=False)
        parser.add_argument('--map_name', type=str, help="Specify the map name to be used")
        parser.add_argument('--run_name', type=str)
        parser.add_argument('--vehicle_type', type=str , choices=['turtlebot' , 'uuv'] , default= "turtlebot")
        parser.add_argument('--control', type=str , choices=['self' , 'pid'] , default="self" , help="self publishes message to /cmd_vel while pid publishes to the PID controller")
        parser.add_argument('--skip_waypoints', type=int , default=1 , help = 'Number of states to skip to generate waypoints for PID')
        args = parser.parse_args()
        main(args)
    except rospy.ROSInterruptException:
        pass