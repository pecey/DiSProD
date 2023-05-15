import numpy as np
import json
from time import sleep
from gym import spaces
DEGREE_TO_RADIAN_MULTIPLIER = np.pi/180

def generate_boundary(min_x_position, max_x_position, min_y_position, max_y_position, boundary_width=1):
    return [ {"x": min_x_position, "y": min_y_position, "width": max_x_position, "height": boundary_width}, # bottom
             {"x": min_x_position, "y": max_y_position - boundary_width, "width": max_x_position, "height": boundary_width}, # top
             {"x": min_x_position, "y": min_y_position, "width": boundary_width, "height": max_y_position}, # left
             {"x": max_x_position - boundary_width, "y": min_y_position, "width": boundary_width, "height": max_y_position} # right
             ]

class EnvironmentProperties:
    """
    If ROS interface is used, then we don't need to construct a Gym environment. It is enough to pass an object of this class.
    """
    def __init__(self, env_cfg , state_reader , action_sender , model_renderer , start_and_goal_pub , imaginary_trajec_sender , planner_reset):
        self.state_reader = state_reader
        self.action_sender = action_sender
        self.model_render = model_renderer
        self.start_and_goal_pub = start_and_goal_pub
        self.imaginary_trajec_sender = imaginary_trajec_sender
        self.planner_reset = planner_reset

        
        self.goal_boundary = 0.5
        self.discrete = env_cfg["discrete"]

        self.min_x_position, self.max_x_position = -10, 20
        self.min_y_position, self.max_y_position = -10, 20
        
        self.min_theta = -3.14
        self.max_theta = 3.14

        self.min_velocity, self.max_velocity = 0, 0.5
        self.min_angular_velocity, self.max_angular_velocity = -60 * DEGREE_TO_RADIAN_MULTIPLIER , 60 * DEGREE_TO_RADIAN_MULTIPLIER
        self.min_delta_velocity, self.max_delta_velocity = -0.05 , 0.05
        self.min_delta_angular_velocity, self.max_delta_angular_velocity = -1 , 1
        self.delta_angular_velocity_multiplier = 6
        

        self.reset_observation_space()

        with open(env_cfg['obstacles_config_path'], 'r') as f:
            config_data = f.read()
            config_json = json.loads(config_data)

        if config_json['use_map'] == 1:
            maps = config_json['maps']
            config = np.random.choice(maps) if env_cfg['map_name'] == "random" else maps[env_cfg['map_name']]

            env_cfg["config"] = config
            self.nS = 5


        
            
            self.x, self.y = config['x'], config['y']
            self.goal_x, self.goal_y = config['goal_x'], config['goal_y']
            self.obstacles = config['obstacles'] if env_cfg['add_obstacles'] else list()


            # Update boundary
            boundary = config["boundary"] if config.get("boundary", None) else config_json['boundary']

            if len(boundary) == 1:
                boundary = boundary[0]
            self.min_x_position, self.max_x_position, self.min_y_position, self.max_y_position = boundary['x_min'], boundary['x_max'], boundary['y_min'], boundary['y_max']

            self.boundary_width = boundary['boundary_width']
            self.boundary = generate_boundary(self.min_x_position, self.max_x_position, self.min_y_position,
                                            self.max_y_position, self.boundary_width)

            self.obstacle_matrix = np.stack([np.array([el['x'], el['x']+el['width'], el['y'], el['y']+el['height']]) for el in (self.obstacles + self.boundary)])



            print(f"Obstacles  : {self.obstacles}")
            print(f"Goal : {self.goal_x, self.goal_y}")
            # Define the observation space.
        else:
            self.obstacles = list()
            
            self.boundary_width = 0.5
            self.boundary = generate_boundary(self.min_x_position, self.max_x_position, self.min_y_position, self.max_y_position, self.boundary_width)
            self.obstacle_matrix = np.stack([np.array([el['x'], el['x']+el['width'], el['y'], el['y']+el['height']]) for el in (self.obstacles + self.boundary)])


            self.x , self.y , self.theta = self.get_random_state()[:3]
            self.goal_x , self.goal_y = self.get_random_state()[:2] 

        self.low_state = np.array([self.min_x_position, self.min_y_position, self.min_theta, -1 , -1], dtype=np.float32)
        self.high_state = np.array([self.max_x_position, self.max_y_position, self.max_theta , 1 , 1], dtype=np.float32)

        

        print("Observation shape is ",self.observation_space.shape)

            
        # Define the action space.
        self.low_action = np.array([self.min_delta_velocity, self.min_delta_angular_velocity], dtype=np.float32)
        self.high_action = np.array([self.max_delta_velocity, self.max_delta_angular_velocity], dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.low_action,
            high=self.high_action,
            dtype=np.float32
        )
        self.time_interval = 0.2


    def boundaries_as_obstacle(self):
        self.obstacles += self.boundary

    def reset(self):
        self.count = 0
        self.planner_reset()

        

        self.reset_observation_space()


        self.obstacles = list()
        self.obstacle_matrix = np.stack([np.array([el['x'], el['x']+el['width'], el['y'], el['y']+el['height']]) for el in (self.obstacles + self.boundary)])

            
        self.x , self.y , self.theta = self.get_random_state()[:3]
        self.goal_x , self.goal_y = self.get_random_state()[:2] 

        self.start_and_goal_pub(self.x , self.y , self.goal_x , self.goal_y)

        self.model_render(self.x , self.y , self.theta)
        sleep(2)
        
        self.linear_velocity, self.angular_velocity = 0, 0
        

        self.done = False
        return np.array((self.x, self.y, self.theta, self.linear_velocity, self.angular_velocity))


    def reset_observation_space(self):
        self.low_state = np.array([self.min_x_position + 10 , self.min_y_position + 10, self.min_theta, self.min_velocity , self.min_angular_velocity], dtype=np.float32)
        self.high_state = np.array([self.max_x_position - 10, self.max_y_position - 10, self.max_theta , self.max_velocity , self.max_angular_velocity], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

    def step(self , action , imaginary_trajectory):
        x , y , theta , ux , uy = self.state_reader()
        del_ux , del_uy = action

        



        final_action = [np.clip(ux + del_ux , self.min_velocity , self.max_velocity) ,\
            uy + del_uy * DEGREE_TO_RADIAN_MULTIPLIER * self.delta_angular_velocity_multiplier]
        

        self.action_sender(final_action)
        self.imaginary_trajec_sender(imaginary_trajectory)

        dist = ((x - self.goal_x)**2 + (y - self.goal_y)**2)**0.5

        if  dist < 0.5:
            self.done = True
        

        sleep(0.2)
        next_observation = self.state_reader()
        reward = 0

        self.count += 1
        if self.count == 200:
            self.done = True

        return np.array([x , y, theta , final_action[0] , final_action[1]]),np.array(next_observation), reward, self.done, {}

    
    def get_random_state(self):
        '''
        Gets theta in radian!
        '''
        state =  self.observation_space.sample()
        return state

