import numpy as np
import json
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
    def __init__(self, env_cfg):
        self.alpha = 0
        self.goal_boundary = 0.5

        self.min_x_position, self.max_x_position = -10, 20
        self.min_y_position, self.max_y_position = -10, 20
        
        self.min_theta = -3.14
        self.max_theta = 3.14

        self.min_velocity, self.max_velocity = 0, 0.5
        self.min_angular_velocity, self.max_angular_velocity = -60 * DEGREE_TO_RADIAN_MULTIPLIER , 60 * DEGREE_TO_RADIAN_MULTIPLIER
        self.min_delta_velocity, self.max_delta_velocity = -0.05 , 0.05
        self.min_delta_angular_velocity, self.max_delta_angular_velocity = -1 , 1
        self.delta_angular_velocity_multiplier = 6
        
        self.nS = 5

        # Read the map data and load the relevant config
        with open(env_cfg['obstacles_config_path'], 'r') as f:
            config_data = f.read()
            map_data = json.loads(config_data)

        maps = map_data['maps']
        map = np.random.choice(maps) if env_cfg['map_name'] == "random" else maps[env_cfg['map_name']]
        self.load_map_config(env_cfg, map_data, map)
        
        # Define the observation space.
        self.reset_observation_space()
        print(f"Shape of observation shape: {self.observation_space.shape}")
            
        # Define the action space.
        self.low_action = np.array([self.min_delta_velocity, self.min_delta_angular_velocity], dtype=np.float32)
        self.high_action = np.array([self.max_delta_velocity, self.max_delta_angular_velocity], dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.low_action,
            high=self.high_action,
            dtype=np.float32
        )
        self.time_interval = 0.2

    def load_map_config(self, env_cfg, global_map_config, map_config):
        self.x, self.y = map_config['x'], map_config['y']
        self.goal_x, self.goal_y = map_config['goal_x'], map_config['goal_y']
        self.obstacles = map_config['obstacles'] if env_cfg['add_obstacles'] else list()

        # Update boundary
        boundary = map_config["boundary"] if map_config.get("boundary", None) else global_map_config['boundary']

        if len(boundary) == 1:
            boundary = boundary[0]
        
        self.min_x_position, self.max_x_position, self.min_y_position, self.max_y_position = boundary['x_min'], boundary['x_max'], boundary['y_min'], boundary['y_max']
        self.boundary_width = boundary['boundary_width']
        self.boundary = generate_boundary(self.min_x_position, self.max_x_position, self.min_y_position,
                                        self.max_y_position, self.boundary_width)

        self.obstacle_matrix = np.stack([np.array([el['x'], el['x']+el['width'], el['y'], el['y']+el['height']]) for el in (self.obstacles + self.boundary)])

        print(f"Obstacles  : {self.obstacles}")
        print(f"Goal : {self.goal_x, self.goal_y}")

    def boundaries_as_obstacle(self):
        self.obstacles += self.boundary

    def reset_observation_space(self):
        self.low_state = np.array([self.min_x_position + 10 , self.min_y_position + 10, self.min_theta, self.min_velocity , self.min_angular_velocity], dtype=np.float32)
        self.high_state = np.array([self.max_x_position - 10, self.max_y_position - 10, self.max_theta , self.max_velocity , self.max_angular_velocity], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

    
    def get_random_state(self):
        '''
        Gets theta in radian!
        '''
        state =  self.observation_space.sample()
        return state

