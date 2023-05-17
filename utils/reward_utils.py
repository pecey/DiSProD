import jax
import jax.numpy as jnp

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


# To make it sharp, increase sharpness
def is_between_smooth(x, lower_limit, upper_limit, sharpness=10):
    return 2 * jax.nn.sigmoid(sharpness * (jax.nn.relu(x - lower_limit) * jax.nn.relu(upper_limit - x))) - 1


# If collision with wall, then return 10, else 0
# https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
# F.relu(self.env.obstacle_x - x) : 0 if x > obstacle_left_boundary
# F.relu(x - (self.env.obstacle_x + self.env.obstacle_width)) : 0 if x < obstacle_right_boundary
# When collision_x and collision_y both are zero, (x,y) is within obstacle boundary. Return 1
# (x,y) is the lower left corner.
def get_number_of_collisions(x, y, env):
    collision_scores = detect_collision_with_rectangular_obstacle(x, y, jnp.array(env.obstacle_matrix))
    return jnp.sum(collision_scores)

def detect_collision_with_rectangular_obstacle(x, y, obstacle):
    obstacle_buffer = 0.1
    x_between_limits = is_between_limits(x, obstacle[:, 0] - obstacle_buffer, obstacle[:, 1] + obstacle_buffer)
    y_between_limits = is_between_limits(y, obstacle[:, 2] - obstacle_buffer, obstacle[:, 3] + obstacle_buffer)
    # collision happens when both of them are true together
    return jax.nn.relu(x_between_limits + y_between_limits - 1)


def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi