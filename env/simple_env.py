from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import json
import gym
from gym import spaces

DEGREE_TO_RADIAN_MULTIPLIER = np.pi/180


def euclidean_distance(pos, target):
    return np.sqrt(np.square(pos[0]-target[0]) + np.square(pos[1]-target[1]))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class SimpleEnv(gym.Env):

    def __init__(self, alpha=0.0, sparsity=1.0):
        self.x, self.y = None, None
        self.theta = None
        self.x_traj, self.y_traj = None, None
        self.goal_x, self.goal_y = 20, 20
        self.goal_boundary = 0.5
        
        self.min_x = 0
        self.min_y = 0
        
        self.max_x = 40
        self.max_y = 40

        self.dt = 1
        
        self.sparsity = sparsity
        
        
        self.low_state = np.array([self.min_x, self.min_y], dtype=np.float32)
        self.high_state = np.array([self.max_x, self.max_y], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )
        
        self.max_action =  np.array((1, 1), dtype=np.float32)
        self.action_space = spaces.Box(-self.max_action, self.max_action, dtype=np.float32)
        
        self.alpha = alpha


    def step(self, action):
        delta_x, delta_y = action
        
        noise = np.random.normal(loc=0, scale=1)
        # print(f"Noise applied: {(noise + 0.1*np.power(noise, 2)) * self.alpha}")
        
        # self.x = np.clip(self.x + np.power(delta_x, 3) - (noise + 0.1*np.power(noise, 2)) * self.alpha, self.min_x, self.max_x)
        # self.y = np.clip(self.y + np.power(delta_y, 3), self.min_y, self.max_y)
        
        action_x = np.power(delta_x,3) 
        action_y = np.power(delta_y,3)
        
        self.x = np.clip(self.x + action_x + (0.1 * noise + np.power(noise, 2)) * self.alpha, self.min_x, self.max_x)
        self.y = np.clip(self.y + action_y, self.min_y, self.max_y)
        
        self.x_traj.append(self.x)
        self.y_traj.append(self.y)
        
        reward = self.reward_fn(self.x, self.y)
        return np.array((self.x, self.y)), reward, False, {}    


    def reward_fn(self, x, y):
        return 1-2 * sigmoid(self.sparsity * (euclidean_distance((x, y), (self.goal_x, self.goal_y))-self.goal_boundary))
    
    def reset(self):
        self.x = np.random.uniform(0, 2)
        self.y = np.random.uniform(0, 9)
        
        self.x_traj = [self.x]
        self.y_traj = [self.y]
        
        self.imagine_x_traj = np.array([self.x])
        self.imagine_y_traj = np.array([self.y])
        
        self.reset_canvas()
        return np.array((self.x, self.y))
    
    def set_imagined_trajectory_data(self, best_path, all_paths = None):
        self.imagine_x_traj = np.array(best_path[0, :, 0])
        self.imagine_y_traj = np.array(best_path[0, :, 1])
        self.imagine_x_sd = np.sqrt(np.array(best_path[1, :, 0]))
        self.imagine_y_sd = np.sqrt(np.array(best_path[1, :, 1]))
        
        for el in self.ax.lines[4:]:
            el.remove()
        
        if all_paths is not None:
            for i in range(all_paths.shape[0]):
                x_traj = np.array(all_paths[i, 0, :, 0])
                y_traj = np.array(all_paths[i, 0, :, 1])
                self.ax.plot(x_traj, y_traj, alpha=0.2)
        
        self.imagine_graph.set_data(self.imagine_x_traj, self.imagine_y_traj)
        self.ax.collections.clear()
        
        x,y,z = self.contour_data
        self.ax.contour(x,y,z)
        self.ax.fill_between(self.imagine_x_traj, self.imagine_y_traj - self.imagine_y_sd, self.imagine_y_traj + self.imagine_y_sd, alpha=0.2, interpolate=True, color='r')
        self.ax.fill_betweenx(self.imagine_y_traj, self.imagine_x_traj - self.imagine_x_sd, self.imagine_x_traj + self.imagine_x_sd, alpha=0.2, interpolate=True, color='r')
        
    def render(self, mode='human', close=False):
        self.graph.set_data(self.x_traj, self.y_traj)
        self.marker.set_data(self.x, self.y)

        plt.pause(0.01)

        if mode == "rgb_array":
            self.fig.canvas.draw()
            data = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data


    def reset_canvas(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.tmp_lines=list()

        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)

        self.graph, = self.ax.plot(self.x_traj, self.y_traj, '--', alpha=0.8)
        self.imagine_graph, = self.ax.plot(self.imagine_x_traj, self.imagine_y_traj, '--', c='r')
        self.marker, = self.ax.plot(self.x, self.y, 'o')

        # Goal plotted by a green "x"
        self.ax.plot(self.goal_x, self.goal_y, "gx")
        x_coord = np.arange(self.min_x, self.max_x)
        y_coord = np.arange(self.min_y, self.max_y)
        [x, y] = np.meshgrid(x_coord, y_coord)
        z = np.array([[self.reward_fn(x[i][j], y[i][j]) for j in range(len(y_coord))] for i in range(len(x_coord))])
        self.contour_data = (x,y,z)
    
