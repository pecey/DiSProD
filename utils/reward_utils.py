import jax
import jax.numpy as jnp

def is_elliptical_limits_outside(x_centre , y_centre , a , b , x , y , alpha):
    '''
    This function needs to return 0 if position x , y is inside the ellipse of dimentions x_centre, y_centre, a , b
    1 otherwise.
    1e7 can be justified here because the value of tanh should always be either 0 or +1. Intuitively the car cannot be 
    inside and outside the track, therefore the result of this should be either 0 or 1 always.
    '''
    return  jnp.tanh(1e7 * jax.nn.relu(jnp.power( (x - x_centre)*jnp.cos(alpha) + (y - y_centre) * jnp.sin(alpha) , 2.0)/(a ** 2.0) + jnp.power((x - x_centre) * jnp.sin(alpha) - (y - y_centre)*jnp.cos(alpha) , 2)/(b ** 2.0) - 1)) 

def is_elliptical_between(x_centre , y_centre , a , b , x , y , alpha , offset):

    inner_than = is_elliptical_limits_outside(x_centre , y_centre , a + offset , b + offset , x , y , alpha)
    outer_than = (1 - is_elliptical_limits_outside(x_centre , y_centre ,  a -  offset , b - offset , x , y , alpha))
    return inner_than + outer_than


# # Utility functions

# Returns [0,1] -> 0.5 if x = target, <0.5 if x < target, >0.5 if x > target
# Higher the sparsity multiplier, more it looks like a step function.
def greater_than_or_eq_to(x, target, sparsity_multiplier=1, epsilon=1e-5):
    return jax.nn.sigmoid(10 * sparsity_multiplier * (x - target + epsilon))


def is_between_limits(x, lower_limit, upper_limit):
    return 1 - jnp.tanh(1e7 * (jax.nn.relu(lower_limit - x) + jax.nn.relu(x - upper_limit)))


def euclidean_distance(position, target):
    x, y = position
    goal_x, goal_y = target
    return jnp.sqrt(jnp.square(x - goal_x) + jnp.square(y - goal_y))


# Used in Hybrid cartpole. Returns 0 if outside limits and a value close to 1 if within limits.
# To make it sharp, increase sharpness
def is_between_smooth(x, lower_limit, upper_limit, sharpness=10):
    return 2 * jax.nn.sigmoid(sharpness * (jax.nn.relu(x - lower_limit) * jax.nn.relu(upper_limit - x))) - 1


# If collision with wall, then return 10, else 0
# https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
# F.relu(self.env.obstacle_x - x) : 0 if x > obstacle_left_boundary
# F.relu(x - (self.env.obstacle_x + self.env.obstacle_width)) : 0 if x < obstacle_right_boundary
# When collision_x and collision_y both are zero, (x,y) is within obstacle boundary. Return 1
# (x,y) is the lower left corner.
# TODO: Collision is happening with wall corners. Need to check collision with borders.
# TODO: Move walls to env file
def get_number_of_collisions(x, y, env):
    #obstacles = env.obstacles + env.boundary
    # collision_score = 1 + T.clamp(outside_x_min + outside_x_max + outside_y_min + outside_y_max, 0, 1) * 9
    #collision_score = sum([detect_collision_with_rectangular_obstacle(x, y, obstacle) for obstacle in obstacles])
    collision_scores = detect_collision_with_rectangular_obstacle(x, y, jnp.array(env.obstacle_matrix))
    return jnp.sum(collision_scores)

def detect_collision_with_rectangular_obstacle(x, y, obstacle):
    obstacle_buffer = 0.1
    x_between_limits = is_between_limits(x, obstacle[:, 0] - obstacle_buffer, obstacle[:, 1] + obstacle_buffer)
    y_between_limits = is_between_limits(y, obstacle[:, 2] - obstacle_buffer, obstacle[:, 3] + obstacle_buffer)
    # collision happens when both of them are true together
    return jax.nn.relu(x_between_limits + y_between_limits - 1)
#
#


def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi