import numpy as np
from gym import spaces
import json

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
        self.goal_boundary = 0.5
        self.discrete = env_cfg["discrete"]

        self.min_x_position, self.max_x_position = 0, 11
        self.min_y_position, self.max_y_position = 0, 11
        self.min_theta, self.max_theta = -180, 180

        with open(env_cfg['obstacles_config_path'], 'r') as f:
            config_data = f.read()
            config_json = json.loads(config_data)

        maps = config_json['maps']
        config = np.random.choice(maps) if env_cfg['map_name'] == "random" else maps[env_cfg['map_name']]

        env_cfg["config"] = config


        
            
        self.x, self.y = config['x'], config['y']
        self.goal_x, self.goal_y = config['goal_x'], config['goal_y']
        self.obstacles = config['obstacles'] if env_cfg['add_obstacles'] else list()


        # Update boundary
        boundary = config["boundary"] if config.get("boundary", None) else config_json['boundary']

        boundary = boundary[0]
        self.min_x_position, self.max_x_position, self.min_y_position, self.max_y_position = boundary['x_min'], boundary['x_max'], boundary['y_min'], boundary['y_max']

        self.boundary_width = boundary['boundary_width']
        self.boundary = generate_boundary(self.min_x_position, self.max_x_position, self.min_y_position,
                                          self.max_y_position, self.boundary_width)

        self.obstacle_matrix = np.stack([np.array([el['x'], el['x']+el['width'], el['y'], el['y']+el['height']]) for el in (self.obstacles + self.boundary)])

        self.delta_velocity_model = True
        self.angular_velocity_multiplier = 60
        self.alpha = 0.0
        self.nS = 3


        print(f"Obstacles  : {self.obstacles}")
        print(f"Goal : {self.goal_x, self.goal_y}")
        # Define the observation space.
        self.low_state = np.array([self.min_x_position, self.min_y_position, self.min_theta, -1 , -1], dtype=np.float32)
        self.high_state = np.array([self.max_x_position, self.max_y_position, self.max_theta , 1 , 1], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )


        

        if self.discrete:
            self.default_velocity = 0.5
            self.turning_velocity = 0.05
            self.angular_velocity = 30
            self.action_space = spaces.Discrete(4)
            self.time_interval = 0.5
        else:
            self.min_velocity, self.max_velocity = 0, 1
            self.min_angular_velocity, self.max_angular_velocity = -60,60

            # Define the action space.
            self.low_action = np.array([self.min_velocity, self.min_angular_velocity], dtype=np.float32)
            self.high_action = np.array([self.max_velocity, self.max_angular_velocity], dtype=np.float32)
            self.action_space = spaces.Box(
                low=self.low_action,
                high=self.high_action,
                dtype=np.float32
            )
            self.time_interval = 0.2

    def boundaries_as_obstacle(self):
        self.obstacles += self.boundary




