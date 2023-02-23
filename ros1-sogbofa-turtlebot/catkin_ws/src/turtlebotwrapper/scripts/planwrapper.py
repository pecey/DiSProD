import argparse
from ast import Raise
import imp
from logging import shutdown

import rospy
import numpy as np
import sys
import os
import xml.etree.ElementTree as ET
import json
import tf
from omegaconf import OmegaConf
import torch as T
import time
import jax.numpy as jnp
from matplotlib import pyplot as plt
from datetime import date
import jax




DISPROD_PATH = os.getenv("DISPROD_PATH")
sys.path.append(DISPROD_PATH)
sys.path.append(os.path.join(DISPROD_PATH, "ros1-sogbofa-turtlebot"))
DISPROD_CONF_PATH = os.path.join(DISPROD_PATH, "config")
DISPROD_MOD_PATH = os.path.join(DISPROD_PATH, "ros1-sogbofa-turtlebot/catkin_ws/sdf_models")
from visualization_helpers.marker_array_rviz import PoseArrayRviz


from geometry_msgs.msg import Twist, Point, TwistStamped
from nav_msgs.msg import Odometry
from math import atan2, asin

from visualization_msgs.msg import Marker, MarkerArray

# note: this fails for local import so I moved it to global Python path
# from .planalg import planalg
from utils.common_utils import print_, set_global_seeds, prepare_config , load_method , load_config_if_exists
from planners.ros_interface import setup_jax_model, setup_planner , setup_mbrl_agent , setup_torch_model
from pathlib import Path


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


def setup_output_dirs(cfg, run_name):

    base_dir = f"{DISPROD_PATH}/results/{cfg['env_name']}/evaluation/{run_name}"

    print(base_dir)
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    cfg["results_dir"] = base_dir

    model_dir = f"{base_dir}/model"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    cfg["model_dir"] = model_dir

    data_dir = f"{base_dir}/data"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cfg["data_dir"] = data_dir

    log_dir = f"{base_dir}/logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    cfg["log_dir"] = log_dir
    cfg["log_file"] = f"{log_dir}/output.log"

    graph_dir = f"{base_dir}/graphs"
    Path(graph_dir).mkdir(parents=True, exist_ok=True)
    cfg["graph_dir"] = graph_dir
    print(f"Output is available in: {base_dir}")
    return base_dir


def update_config_with_args(cfg, args):
    keys_to_update = ["seed", "log_file", "render", "depth", "run_name"]
    for key in keys_to_update:
        if args.__contains__(key) and getattr(args,key) is not None:
            if key in ["render", "compute_baseline"]:
                cfg[key] = getattr(args, key).lower() == "true"
            else:
                cfg[key] = getattr(args, key)

    # Overwrite dynamics model path if value is set.
    if args.__contains__("dynamics_model_path") and getattr(args, "dynamics_model_path"):
        cfg["dynamics_model_path"] = args.dynamics_model_path

    # If run_name is set, the update in config. Else set default value to {running_mode}_{current_time}
    if args.__contains__("run_name") and getattr(args, "run_name"):
        cfg["run_name"] = args.run_name
    else:
        from datetime import date

        today = date.today()
        cfg["run_name"] = f"{today.strftime('%y-%m-%d')}_{cfg['mode']}_{int(time.time())}"

    # Update config for dubins car
    if "dubins" in cfg["env_name"]:
        if getattr(args, "obstacles_config_file").lower() == "none":
            cfg["obstacles_config_path"] = None
        else:
            cfg["obstacles_config_path"] = f"{DISPROD_PATH}/env/assets/{args.obstacles_config_file}.json"
            cfg["map_name"] = args.map_name

    cfg['nn_model'] = False
    return cfg


class TurtleBotWrapper:
    def __init__(self, env_config, debug=False, ser=None):
        rospy.init_node('turtlebotwrapper', anonymous=True , disable_signals = True)
        # Subscribers
        rospy.Subscriber('/odom', Odometry, self.pose_listener_callback)
        rospy.Subscriber('/turtlewrapper/GoalLocation', Point, self.goal_listener_callback)
        self.track_pub_1 = rospy.Publisher('/track1', Marker, queue_size=1)

        # Publishers
        self.goal_pub = rospy.Publisher('/goal_marker', MarkerArray, queue_size=2)
        self.goal_markers = MarkerArray()
        self.obstacle_pub = rospy.Publisher('/obstacles', MarkerArray, queue_size=10)
        self.obstacle_markers = MarkerArray()
        self.start_pub = rospy.Publisher('/start_marker' , MarkerArray , queue_size=2)
        self.start_markers = MarkerArray()
        self.cmd_pub = rospy.Publisher('/cmd_vel',Twist, queue_size=10)
        self.myrate = 10
        self.fixed_time_mode = env_config['fixed_time_pub_mode']
        self.vehicle_model = env_config['vehicle_model']
        if self.fixed_time_mode == True:
            self.action_generated = False
            self.timer = rospy.Timer(rospy.Duration(1/self.myrate), self.publisher_callback)
        self.flag = False
        self.rate = rospy.Rate(self.myrate)  # hz ie 0.2 sec
        # self.timer = self.create_timer(timer_period, self.planner_callback)
        self.goal_x = None
        self.goal_y = None
        
        self.pose = None
        self.last_linear_vel , self.last_angular_vel = 0,0

        self.agent = None
        self.env = None
        self.first = True

        self.target_deltas = []
        self.predicted_delta_mus = []
        self.predicted_delta_vars = []
        self.env_config = env_config
        self.true_next_states = []
        self.predicted_next_states = []
        self.predicted_next_state_model = []
        

    # Listen to the goal setter. Update goal for agent, and publish goal markers
    def goal_listener_callback(self, msg):
        rospy.loginfo('Goal Received: ("%f","%f")' % (msg.x, msg.y))
        self.env.x , self.env.y = self.pose['x_pos'], self.pose['y_pos']
        self.env.goal_x = msg.x
        self.env.goal_y = msg.y
        self.goal_x = msg.x 
        self.goal_y = msg.y
        self.is_goal_set = True
        self.publish_goal_marker()

    # Publish goal marker
    def publish_goal_marker(self):
        goal = self.create_marker(self.env.goal_x , self.env.goal_y , c = [1 , 0 , 0])
        self.goal_markers.markers.append(goal)
        self.goal_pub.publish(self.goal_markers)


    # Publish goal marker
    def publish_start_marker(self):
        goal = self.create_marker(self.start_x , self.start_y)
        self.start_markers.markers.append(goal)
        self.start_pub.publish(self.start_markers)
    # Publish the obstacles
    def publish_obstacle_markers(self, mode=0):
        obstacles = self.create_obstacle_markers(mode)
        for obstacle in obstacles:
            self.obstacle_markers.markers.append(obstacle)
        self.obstacle_pub.publish(self.obstacle_markers)

    # Create the goal marker
    def create_marker(self , x , y , c = [ 0 , 0 , 1]):
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
        

    def render_model(self, x, y, yaw=0.0):

        os.system("rosservice call gazebo/delete_model '{model_name: turtlebot3_burger}'")

        if self.vehicle_model == "turtlebot":
            os.system(
                'rosrun gazebo_ros spawn_model -urdf -model turtlebot3_burger -x {} -y {} -Y {} -param robot_description'.format(
                    x, y, yaw))

        elif self.vehicle_model == "jackal":
            os.system(
                'rosrun gazebo_ros spawn_model -urdf -model jackal -x {} -y {} -Y {} -param robot_description2'.format(
                    x, y, yaw))

        else:
            print(f"Vehicle model {self.vehicle_model} not implemented")
            raise NotImplementedError

    def plot_actual_vs_predicted_states(self, actual_states, predicted_states, predicted_variance, filename):
        nS = actual_states.shape[1]
        n_timesteps = len(actual_states)
        height = 5 * nS if nS > 2 else 12
        width = n_timesteps // 25 if n_timesteps > 200 else 8
        fig, axs = plt.subplots(nS, figsize=(width, height))
        timesteps = list(range(n_timesteps))
        for i in range(nS):
            axs[i].plot(timesteps, actual_states[:, i], 'o-', label="Actual")
            axs[i].plot(timesteps, predicted_states[:, i], 'b-', label="Predicted")
            axs[i].fill_between(timesteps, predicted_states[:, i] - predicted_variance[:, i],
                                predicted_states[:, i] + predicted_variance[:, i], alpha=0.2, color="blue")
            axs[i].set_title(f"State {i}")
            axs[i].legend()
            axs[i].grid()
        plt.tight_layout()
        path = f"{self.env_config['graph_dir']}/{filename}_model_accuracy.svg"
        plt.savefig(path, format='svg')
        plt.close()


    def plot_actual_vs_predicted_states_full(self, actual_states, predicted_states, filename):
        nS = actual_states.shape[1]
        n_timesteps = len(actual_states)
        height = 5 * nS if nS > 2 else 12
        width = n_timesteps // 25 if n_timesteps > 200 else 8
        fig, axs = plt.subplots(nS, figsize=(width, height))
        timesteps = list(range(n_timesteps))
        for i in range(nS):
            axs[i].plot(timesteps, actual_states[:, i], 'o-', label="Actual")
            axs[i].plot(timesteps, predicted_states[:, i], 'b-', label="Predicted")
            axs[i].set_title(f"State {i}")
            axs[i].legend()
            axs[i].grid()
        plt.tight_layout()
        path = f"{self.env_config['graph_dir']}/{filename}_model_accuracy.svg"
        plt.savefig(path, format='svg')
        plt.close()

    def shutdownHook(self):

        if self.compare_with_simulator:

        #self.plot_actual_vs_predicted_states(np.array(self.target_deltas),
        #                                     np.array(self.predicted_delta_mus),
        #                                     np.array(self.predicted_delta_vars),
        #                                     filename=f"policy_traj")

            self.plot_actual_vs_predicted_states_full(
                np.array(self.true_next_states)[:40],
                np.array(self.predicted_next_state_model)[:40] , 
                filename = f"policy_traj_full"
            )
        
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

    def create_obstacle_markers(self, mode=0):
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

            if mode == 1:
                self.render_in_gazebo(idx, pose_x=obst.pose.position.x, pose_y=obst.pose.position.y, pose_z=0,
                                      size_x=obstacle["width"], size_y=obstacle["height"], size_z=0.5)

            obst.pose.position.z = 0
            obstacles.append(obst)

        self.obstacle_length = len(obstacles)
        return obstacles

    def pose_listener_callback(self, msg):
        state = {'x_pos': msg.pose.pose.position.x, 'y_pos': msg.pose.pose.position.y,
                 'z_pos': msg.pose.pose.position.z}

        self.flag = True

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

    def planner_callback(self, step_num):
        # Either goal is not set or the current pose of bot is not set
        if not self.flag:
            rospy.loginfo('Either goal not set or current pose is missing, nothing to do')
            return 0

        

        state = np.array([self.pose['x_pos'], self.pose['y_pos'], self.pose['yaw'], self.last_linear_vel, self.last_angular_vel])[:self.env.nS]

        if self.first:
            self.state_for_model = state 
            self.first = False
        ### at beginning lets say its 0 , 0 , pi/2 , 0 , 0
        goal = np.array([self.goal_x, self.goal_y])

        dist = np.linalg.norm(goal - state[:2])

        if dist <= self.env.goal_boundary:
            rospy.loginfo('Arrived at Goal')
            rospy.loginfo(
                f"Goal: {self.goal_x, self.goal_y}, Location: {self.pose['x_pos'], self.pose['y_pos']}")
            self.is_goal_set = False
            cmd = generate_command_message([0, 0])
            self.cmd_pub.publish(cmd)
            rospy.loginfo(f"Steps taken to solve: {step_num}")
            rospy.sleep(0.2)
            return step_num

        
        time1 = rospy.Time().now().to_sec()
        _ , action , imagined_trajectory = self.plan_one_step(self.planner, self.env, state, goal)

        if self.pose_array_viz and imagined_trajectory!= None:
            rospy.loginfo("Sending pose visualization")
            self.pose_array_viz.publish(imagined_trajectory)
        self.last_linear_vel , self.last_angular_vel = action[0] , action[1]
        cmd = generate_command_message(action)
        time2 = rospy.Time().now().to_sec()

        print(f"Action generated after time {time2 - time1}")
        
        
        if self.compare_with_simulator:
            true_next_state = np.array([self.pose['x_pos'], self.pose['y_pos'], self.pose['yaw'], self.last_linear_vel, self.last_angular_vel][:self.env.nS]).reshape(-1 , 1)
            true_action = np.array([true_action[0] , true_action[1]]).reshape(-1 , 1)
            state = state.reshape(-1,1)
            x = jnp.concatenate(((self.state_for_model.reshape(-1 , 1)), true_action))
            predicted_delta_mu, predicted_delta_var = self.jax_model.forward(x)

            self.state_for_model += predicted_delta_mu

            self.predicted_delta_mus.append(predicted_delta_mu)
            self.predicted_delta_vars.append(predicted_delta_var)
            self.target_deltas.append(true_next_state - state)


            self.true_next_states.append(true_next_state)
            self.predicted_next_state_model.append(self.state_for_model)
        
            

        
        if self.fixed_time_mode == True:
            self.command_generated = cmd
            self.action_generated = True
            return step_num + 1

        else:
            self.cmd_pub.publish(cmd)
            return step_num + 1

        
    def publisher_callback(self, timer):
        if not self.action_generated:
            return
        self.cmd_pub.publish(self.command_generated)

    def action_sender(self , action):
        cmd = generate_command_message(action)
        self.cmd_pub.publish(cmd)

    def state_reader(self):
        return [self.pose['x_pos'], self.pose['y_pos'], self.pose['yaw'] , self.pose['x_vel'] , self.pose['ang_vel']]

    def start_and_goal_pub(self , x  , y , goal_x , goal_y):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.start_x , self.start_y = x , y
        self.publish_goal_marker()
        self.publish_start_marker()

    def imag_trajec_pub(self , imag_traj):
        self.pose_array_viz.publish(imag_traj)
    def planner_reset(self):
        self.planner.reset()
    
    
    

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
    cmd.linear.x = max(0 , action[0])
    cmd.angular.z = action[1]
    return cmd


# Read configuration from file
def get_configuration(config_path, config_num):
    if config_path is not None:
        with open(config_path, 'r') as f:
            config_data = f.read()
        config_json = json.loads(config_data)
        configs = config_json['config']
        selected_config = np.random.choice(configs) if config_num == -1 else configs[config_num]
        boundary = np.random.choice(config_json['boundary'])
        return selected_config, boundary
    return None


def update_config_with_args(cfg, args):
    keys_to_update = ["seed", "log_file", "render", "depth", "run_name", "alg" , "nn_model"]
    for key in keys_to_update:
        if args.__contains__(key) and getattr(args,key) is not None:
            if key in ["render", "compute_baseline"]:
                cfg[key] = getattr(args, key).lower() == "true"
            else:
                cfg[key] = getattr(args, key)

    # Overwrite dynamics model path if value is set.
    if args.__contains__("dynamics_model_path") and getattr(args, "dynamics_model_path"):
        cfg["dynamics_model_path"] = args.dynamics_model_path

    # If run_name is set, the update in config. Else set default value to {running_mode}_{current_time}
    if args.__contains__("run_name") and getattr(args, "run_name"):
        cfg["run_name"] = args.run_name
    else:
        today = date.today()
        cfg["run_name"] = f"{today.strftime('%y-%m-%d')}_{cfg['mode']}_{int(time.time())}"

    # Update config for dubins car
    if "dubins" in cfg["env_name"]:
        if getattr(args, "obstacles_config_file").lower() == "none":
            cfg["obstacles_config_path"] = None
        else:
            cfg["obstacles_config_path"] = f"{DISPROD_PATH}/env/assets/{args.obstacles_config_file}.json"
            print("Map name is " , args.map_name)
            cfg["map_name"] = args.map_name
    return cfg




def prepare_config(planner, env_name, cfg_path=None):
    if planner == "naive":
        return dict()
        
    planner_default_cfg = OmegaConf.load(f"{cfg_path}/planning/default.yaml")
    default_cfg = OmegaConf.load(f"{cfg_path}/default.yaml")
    sogbofa_default_cfg = OmegaConf.load(f"{cfg_path}/sogbofa_default.yaml")
    planner_env_cfg = OmegaConf.load(f"{cfg_path}/planning/{env_name}.yaml")
    learning_env_cfg = OmegaConf.load(f"{cfg_path}/learning/{env_name}.yaml")
    learning_default_cfg = OmegaConf.load(f"{cfg_path}/learning/default.yaml")
    return OmegaConf.merge(default_cfg , planner_default_cfg,  sogbofa_default_cfg, planner_env_cfg, learning_default_cfg , learning_env_cfg,OmegaConf.load(f"{cfg_path}/{env_name}.yaml"))


def main(args):
    device = "cuda" if T.cuda.is_available() else "cpu"
    env_cfg = prepare_config(args.alg, args.env, DISPROD_CONF_PATH)
    env_cfg['mode'] = 'tbot_evaluation'
    env_cfg = update_config_with_args(env_cfg, args)
    env_cfg["device"] = device
    env_cfg = update_config_with_args(env_cfg , args)
    
    
    set_global_seeds(env_cfg['seed'])

    tw = TurtleBotWrapper(env_config = env_cfg)
    # Goal is set in env at this point.
    # Any changes to the goal is reflected in env when the planner_callback calls the plan function.

    tw.env = load_method(env_cfg['env_file'])(env_cfg , tw.state_reader , tw.action_sender , tw.render_model , tw.start_and_goal_pub , tw.imag_trajec_pub , tw.planner_reset) 
    run_name = env_cfg["run_name"]

    depth , restart = env_cfg["depth"], env_cfg["n_restarts"]

    setup_output_dirs(env_cfg, run_name)

    tw.compare_with_simulator = args.compare_with_simulator

    if args.poseVisualization:
        tw.pose_array_viz = PoseArrayRviz(depth, restart)
    else:
        tw.pose_array_viz = None 

       
    
    if args.alg == 'sogbofa':
        tw.plan_one_step = load_method(env_cfg['ros_interface'])
    else:
        tw.plan_one_step = load_method(env_cfg['baseline_ros_interface'])
    # model is either clip_dubins_car , learning or dubins_car
    #### model can be either clip_dubins_car or dubins_car or learning

    env_cfg['nn_model'] = False
    key = jax.random.PRNGKey(args.seed)
    tw.planner = setup_planner(tw.env , env_cfg , key)
     
    if env_cfg["nn_model"]:
        print("Learning module started")
        if env_cfg['nn_model'] and env_cfg["model"] == "learning":
            if not os.path.exists(env_cfg['evaluation']['dynamics_model_path']):
                print(f"File {env_cfg['evaluation']['dynamics_model_path']} does not exist ")
                import sys 
                sys.exit()
        jax_model = setup_jax_model(tw.env, env_cfg)
        torch_model = setup_torch_model(tw.env, env_cfg)
        agent = setup_mbrl_agent(tw.env, env_cfg, jax_model, torch_model, tw.planner)
        agent.update_model_in_planner()
        dynamics_path = env_cfg["evaluation"]["dynamics_model_path"]
        tw.jax_model = agent.jax_model
        agent.sync_jax_model_with_torch(T.load(dynamics_path, map_location=device))


    
    
    tw.render_model(env_cfg["config"]["x"], env_cfg["config"]["y"])
    tw.env.boundaries_as_obstacle()
    tw.goal_x = env_cfg["config"]["goal_x"]
    tw.goal_y = env_cfg["config"]["goal_y"]
    tw.start_x , tw.start_y = env_cfg['config']["x"] , env_cfg['config']['y'] 
    tw.publish_goal_marker()
    tw.publish_start_marker()
    tw.publish_obstacle_markers(mode=1)
    
    step_num = 0

    rospy.on_shutdown(tw.shutdownHook)
    tic = time.perf_counter()
    tw.planner.reset()


    while not rospy.is_shutdown():
        curr_step_num = tw.planner_callback(step_num)
        if step_num == curr_step_num:
            print_(f"{env_cfg['map_name']}  , {step_num}" , env_cfg['log_file'])
            rospy.signal_shutdown("Environment is solved") 

        if step_num > 400:
            rospy.signal_shutdown("Environment Timeout")
            print_(f"{env_cfg['map_name']}  , 400" , env_cfg['log_file'])
            raise TimeoutError
        step_num = curr_step_num
        rospy.sleep(1/100)
    
        

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--log_file', type=str, default=None)
        parser.add_argument('--seed', type=int, help='Seed for PRNG', default=42)
        parser.add_argument('--env', type=str, default= "continuous_dubins_car_w_velocity")
        parser.add_argument('--noise', type=str, default="False")
        parser.add_argument('--obstacles_config_file', type=str, help="Config filename without the JSON extension",
                            default="dubins")
        parser.add_argument('--compare_with_simulator' , type = bool , help = 'flag to compare the true motion with the learnt dynamics' , default = False)
        parser.add_argument('--alg', type=str, default="sogbofa")
        parser.add_argument('--mode', type=str, default="evaluation")
        parser.add_argument('--poseVisualization', type=bool, default=True)
        parser.add_argument('--map_name', type=str, help="Specify the map name to be used. Only called if dubins or continuous dubins env", default="random")
        parser.add_argument('--nn_model',  help="If true nn based model will be used",type = bool, default=False)
        parser.add_argument('--run_name', type = str)
        
        args = parser.parse_args()
        main(args)
    except rospy.ROSInterruptException:
        pass