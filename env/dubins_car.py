from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import json
import gym
from gym import spaces

DEGREE_TO_RADIAN_MULTIPLIER = np.pi/180

# Actions: Fixed velocity for actions other than 3.
# 0: Left : -steering
# 1: Forward : no steering
# 2: Right : steering
# 3: Brake : velocity = 0

def generate_boundary(min_x, max_x, min_y, max_y, boundary_width=1):
    return [{"x": min_x, "y": min_y, "width": max_x, "height": boundary_width},  # bottom
            {"x": min_x, "y": max_y - boundary_width, "width": max_x, "height": boundary_width},  # top
            {"x": min_x, "y": min_y, "width": boundary_width, "height": max_y},  # left
            {"x": max_x - boundary_width, "y": min_y, "width": boundary_width, "height": max_y}  # right
            ]


class DubinsCar(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, alpha=0.0, add_obstacle=False, config_file=None, map_name="random"):
        # Min and Max positions for goal and starting point
        self.min_y_position = 0
        self.min_x_position = 0
        self.max_y_position = 10
        self.max_x_position = 10

        self.min_theta = -60
        self.max_theta = 60

        # Random position for goal
        self.goal_x = np.random.randint(self.min_x_position, self.max_x_position)
        self.goal_y = np.random.randint(self.min_y_position, self.max_y_position)

        # Random starting point for agent
        self.x = np.random.randint(self.min_x_position, self.max_x_position)
        self.y = np.random.randint(self.min_y_position, self.max_y_position)

        self.config_file = config_file
        self.map_name = map_name

        self.add_obstacle = add_obstacle
        self.plot_boundary = True
        self.boundary_width = 0.5
        self.boundary =  generate_boundary(self.min_x_position, self.max_x_position, self.min_y_position, self.max_y_position, self.boundary_width)
        if self.add_obstacle:
            self.obstacles = []
            self.max_obstacle_width = 3
            self.max_obstacle_height = 1

        # Define the observation space.
        self.reset_observation_space()

        # Define the action space
        self.action_space = spaces.Discrete(4)


        self.theta = 0.0
        self.time_interval = 0.5

        self.default_velocity = 1
        self.turning_velocity = 0.2
        self.angular_velocity = 60

        self.goal_boundary = 0.5

        self.x_traj = [self.x]
        self.y_traj = [self.y]

        self.imagine_x_traj = [self.x]
        self.imagine_y_traj = [self.y]

        self.done = False

        self.alpha = alpha

        self.reset_canvas()

    def reset_observation_space(self):
        self.low_state = np.array([self.min_x_position, self.min_y_position, self.min_theta], dtype=np.float32)
        self.high_state = np.array([self.max_x_position, self.max_y_position, self.max_theta], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )


    def step(self, action):
        reward = 0
        linear_velocity, angular_velocity = self.get_velocities_from_action(action)

        noise = np.random.normal(loc=0, scale=1)

        x_old = self.x
        y_old = self.y

        self.x, self.y, self.theta = self.update_state(self.x, self.y, self.theta, angular_velocity, linear_velocity, self.time_interval, noise)

        # Check if the car is within the boundary around the goal
        if np.isclose(self.x, self.goal_x, atol=self.goal_boundary) and np.isclose(self.y, self.goal_y, atol=self.goal_boundary):
            self.done = True

        if self.add_obstacle:
            if self.check_collision(self.x, self.y):
                reward += -10
                self.x = x_old
                self.y = y_old

        self.x_traj.append(self.x)
        self.y_traj.append(self.y)

        if self.done:
            reward += 1

        observation = self.x, self.y, self.theta

        return np.array(observation), reward, self.done, {"goal": (self.goal_x, self.goal_y)}

    def get_velocities_from_action(self, action):
        # Steer left
        if action == 0:
            linear_velocity = self.turning_velocity
            angular_velocity = -self.angular_velocity
        # Go straight
        elif action == 1:
            angular_velocity = 0
            linear_velocity = self.default_velocity
        # Steer right
        elif action == 2:
            linear_velocity = self.turning_velocity
            angular_velocity = self.angular_velocity
        # Apply brakes
        elif action == 3:
            linear_velocity = 0
            angular_velocity = 0
        return linear_velocity, angular_velocity

    # TODO: Can handle boundaries separately. Currently handling them as obstacles.
    def check_collision(self, x, y):
        obstacles = self.obstacles + self.boundary
        for obstacle in obstacles:
            collision_x = True if obstacle["x"] <= x <= obstacle["x"] + obstacle["width"] else False
            collision_y = True if obstacle["y"] <= y <= obstacle["y"] + obstacle["height"] else False
            if collision_x and collision_y:
                return True
        return False

    def reset(self):
        # Load config from file
        if self.config_file is not None:
            with open(self.config_file, 'r') as f:
                config_data = f.read()
            config_json = json.loads(config_data)
            # Update obstacles
            maps = config_json['maps']
            selected_config = np.random.choice(maps) if self.map_name == "random" else maps[self.map_name]
            self.x, self.y = selected_config['x'], selected_config['y']
            self.goal_x, self.goal_y = selected_config['goal_x'], selected_config['goal_y']
            self.obstacles = selected_config['obstacles'] if self.add_obstacle else list()
            
            # Update boundary
            boundary = selected_config["boundary"] if selected_config.get("boundary", None) else np.random.choice(config_json['boundary'])
            self.min_x_position, self.max_x_position, self.min_y_position, self.max_y_position = boundary['x_min'],  boundary['x_max'],  boundary['y_min'],  boundary['y_max']
            self.reset_observation_space()

            self.boundary_width = boundary['boundary_width']
            self.boundary = generate_boundary(self.min_x_position, self.max_x_position, self.min_y_position,
                                              self.max_y_position, self.boundary_width)
            self.obstacle_matrix = np.stack([np.array([el['x'], el['x']+el['width'], el['y'], el['y']+el['height']]) for el in (self.obstacles + self.boundary)])

        else:
            self.obstacles = list()
            goal_behind_car = np.random.randint(0, 2)
            if goal_behind_car:
                self.goal_x = np.random.randint(self.min_x_position, self.max_x_position/2)
                self.goal_y = np.random.randint(self.min_y_position, self.max_y_position/2)

                self.x = np.random.randint(self.goal_x, self.max_x_position)
                self.y = np.random.randint(self.goal_y, self.max_y_position)
            else:
                # Random position for goal
                self.goal_x = np.random.randint(self.min_x_position, self.max_x_position)
                self.goal_y = np.random.randint(self.min_y_position, self.max_y_position)
                # Random starting point for agent
                self.x = np.random.randint(self.min_x_position, self.max_x_position)
                self.y = np.random.randint(self.min_y_position, self.max_y_position)

        self.theta = 0.0

        self.x_traj = [self.x]
        self.y_traj = [self.y]

        self.imagine_x_traj = [self.x]
        self.imagine_y_traj = [self.y]

        plt.close(self.fig)
        self.reset_canvas()

        self.done = False
        return np.array((self.x, self.y, self.theta))

    def reset_canvas(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.ax.set_xlim(self.min_x_position, self.max_x_position)
        self.ax.set_ylim(self.min_y_position, self.max_y_position)

        self.graph, = self.ax.plot(self.x_traj, self.y_traj, '--', alpha=0.8)
        self.imagine_graph, = self.ax.plot(self.imagine_x_traj, self.imagine_y_traj, '--', c='r')
        self.car, = self.ax.plot(self.x, self.y, 'o')

        # Goal plotted by a green "x"
        self.ax.plot(self.goal_x, self.goal_y, "gx")

        # Plot the goal boundary
        boundary = plt.Circle((self.goal_x, self.goal_y), radius=self.goal_boundary, color='orange', alpha=0.8)
        self.ax.add_artist(boundary)

        # Plot the obstacle
        if self.add_obstacle:
            for obstacle in self.obstacles:
                self.ax.add_patch(Rectangle((obstacle["x"], obstacle["y"]), obstacle["width"], obstacle["height"]), color="brown")

        if self.plot_boundary:
            for boundary in self.boundary:
                self.ax.add_patch(Rectangle((boundary["x"], boundary["y"]), boundary["width"], boundary["height"], color='black'))

    def render(self, mode='human', close=False):
        """
        Update the trajectory
        """
        self.graph.set_data(self.x_traj, self.y_traj)
        self.car.set_data(self.x, self.y)

        plt.pause(0.01)

        if mode == "rgb_array":
            self.fig.canvas.draw()
            data = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data

    def set_imagined_trajectory_data(self, path):
        self.imagine_x_traj = path[:, 0]
        self.imagine_y_traj = path[:, 1]
        self.imagine_graph.set_data(self.imagine_x_traj, self.imagine_y_traj)

    def save_trajectory(self, path, filename):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlim(self.min_x_position, self.max_x_position)
        ax.set_ylim(self.min_y_position, self.max_y_position)

        ax.plot(self.x_traj, self.y_traj, '--', alpha=0.8)
        ax.plot(self.x, self.y, 'o')

        ax.plot(self.goal_x, self.goal_y, "gx")
        boundary = plt.Circle((self.goal_x, self.goal_y), radius=self.goal_boundary, color='orange', alpha=0.8)
        ax.add_artist(boundary)

        graph_data = {  "x_lim": [self.min_x_position, self.max_x_position], 
                        "y_lim": [self.min_y_position, self.max_y_position],
                        "tau": [self.x_traj, self.y_traj],
                        "position": [self.x, self.y],
                        "goal": [self.goal_x, self.goal_y],
                        "goal_boundary": [self.goal_boundary]
                    }

        if self.add_obstacle:
            graph_data["obstacles"] = self.obstacles
            for obstacle in self.obstacles:
                ax.add_patch(Rectangle((obstacle["x"], obstacle["y"]), obstacle["width"], obstacle["height"], color="brown"))


        if self.plot_boundary:
            graph_data["boundary"] = self.boundaries
            for boundary in self.boundary:
                self.ax.add_patch(
                    Rectangle((boundary["x"], boundary["y"]), boundary["width"], boundary["height"], color='black'))

        with open(f"{path}/{filename}.json") as f:
            f.write(json.dumps(graph_data))
                
        plt.grid()
        plt.savefig(f"{path}/{filename}.pdf")
        plt.close()

    # remove all variables and use self.
    # keep direction variable
    def update_state(self, x, y, theta, angular_velocity, velocity, time_interval, noise):
        """
        Update the state [x, y, theta] of the robot according to Dubins dynamics.
        """
        new_x = np.clip(x + velocity * np.cos(theta) * time_interval, self.min_x_position, self.max_x_position)
        new_y = np.clip(y + velocity * np.sin(theta) * time_interval, self.min_y_position, self.max_y_position)
        new_theta = theta + (angular_velocity + self.alpha*noise) * time_interval * DEGREE_TO_RADIAN_MULTIPLIER
        if self.add_obstacle and self.check_collision(new_x, new_y):
            return x, y, new_theta
        return new_x, new_y, new_theta


if __name__ == '__main__':
    from gym.envs.registration import register

    register(
        id='dubins-v0',
        entry_point='env.dubins_car:DubinsCar',
        max_episode_steps=250
    )

    env = gym.make('dubins-v0')
    env.reset()
    done = False
    step = 0
    while not done:
        if step < 25:
            move = 2
        else:
            move = int(input())
        obs, rew, done, info = env.step(move)
        step += 1
        print(step)
        if step >= 25:
            env.render().show()
