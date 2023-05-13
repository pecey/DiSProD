import jax.numpy as jnp
import jax
from jax import custom_jvp

DEGREE_TO_RADIAN_MULTIPLIER = jnp.pi / 180


def simple_env(state, actions, env):
    delta_x, delta_y = actions
    x, y, noise = state
    
    action_x = jnp.power(delta_x, 3)
    action_y = jnp.power(delta_y, 3)
    
    x_new = jnp.clip(x + action_x + (0.1 * noise + jnp.power(noise,2)) * env.alpha, env.min_x, env.max_x)
    y_new = jnp.clip(y + action_y, env.min_y, env.max_y)

    # x_new = jnp.clip(x + jnp.power(delta_x,3) - (noise + 0.1 * jnp.power(noise,2)) * alpha, env.min_x, env.max_x)
    # y_new = jnp.clip(y + jnp.power(delta_y,3), env.min_y, env.max_y)    
    
    return jnp.array([x_new, y_new])
    

# Transition function for Dubins Car - Discrete and Continuous
def dubins_car(state, actions, env):
    left_action, nothing, right_action, brakes = actions
    velocity = (env.turning_velocity * left_action
                + env.turning_velocity * right_action
                + env.default_velocity * nothing)

    angular_velocity = (-env.angular_velocity * left_action
                        + env.angular_velocity * right_action
                        + 0 * nothing)

    return _dubins_car(env, state, velocity, angular_velocity)


def continuous_dubins_car_w_obstacles(state, actions, env):
    velocity, angular_velocity = actions
    return _dubins_car_w_obstacles(env, state, velocity, angular_velocity)

def continuous_dubins_car(state, actions, env):
    velocity, angular_velocity = actions
    return _dubins_car(env, state, velocity, angular_velocity)


def continuous_dubins_car_w_velocity_state(state, actions, env):
    delta_velocity, delta_angular_velocity = actions
    if len(state) == 6:
        x , y, theta, old_velocity, old_angular_velocity, noise = state
    else:
        x , y, theta, old_velocity, old_angular_velocity = state[0] , state[1] , state[2] , state[3] , state[4] 
        noise = 0

    velocity = jnp.clip(old_velocity + delta_velocity, env.min_velocity, env.max_velocity)
    delta_angular_velocity_ = (env.alpha * noise + delta_angular_velocity) * env.delta_angular_velocity_multiplier * DEGREE_TO_RADIAN_MULTIPLIER
    angular_velocity = jnp.clip(old_angular_velocity + delta_angular_velocity_ ,env.min_angular_velocity , env.max_angular_velocity)

    dx_dt = velocity * jnp.cos(theta) * env.time_interval
    dy_dt = velocity * jnp.sin(theta) * env.time_interval
    dtheta_dt = angular_velocity * env.time_interval 

    x = jnp.clip(x + dx_dt, env.min_x_position, env.max_x_position)
    y = jnp.clip(y + dy_dt, env.min_y_position, env.max_y_position)

    theta = theta + dtheta_dt
    return jnp.array([x, y, theta, velocity, angular_velocity])


def continuous_dubins_car_ablation(state, actions, env):
    velocity, angular_velocity = actions
    if len(state) == 4:
        x , y, theta, noise = state
    else:
        x , y, theta = state
        noise = 0

    angular_velocity = angular_velocity + noise * env.alpha
    velocity = jnp.clip(velocity**2, env.min_velocity, env.max_velocity)

    dx_dt = velocity * jnp.cos(theta) * env.time_interval
    dy_dt = velocity * jnp.sin(theta) * env.time_interval
    dtheta_dt =  angular_velocity * env.time_interval * DEGREE_TO_RADIAN_MULTIPLIER

    new_x = jnp.clip(x + dx_dt, env.min_x_position, env.max_x_position)
    new_y = jnp.clip(y + dy_dt, env.min_y_position, env.max_y_position)
    new_theta = theta + dtheta_dt

    return jnp.array([new_x, new_y, new_theta])


def _dubins_car(env, state, velocity,angular_velocity):
    if len(state) == 6:
        x, y, theta , _ , _ , _ = state
        noise = 0
    else:
        x, y, theta, noise = state
    dx_dt = velocity * jnp.cos(theta) * env.time_interval
    dy_dt = velocity * jnp.sin(theta) * env.time_interval
    dtheta_dt = (env.angular_velocity_multiplier * angular_velocity + env.alpha * noise) * env.time_interval * DEGREE_TO_RADIAN_MULTIPLIER
    
    x_new = jnp.clip(x + dx_dt, env.min_x_position, env.max_x_position)
    y_new= jnp.clip(y + dy_dt, env.min_y_position, env.max_y_position)
    theta_new = theta + dtheta_dt

    new_position = jnp.array([x_new, y_new, theta_new])

    return new_position


def _dubins_car_w_obstacles(env, state, velocity,angular_velocity):
    if len(state) == 3:
        x, y, theta = state
        noise = 0
    else:
        x, y, theta, noise = state
    dx_dt = velocity * jnp.cos(theta) * env.time_interval
    dy_dt = velocity * jnp.sin(theta) * env.time_interval
    dtheta_dt = (angular_velocity + env.alpha * noise) * env.time_interval * DEGREE_TO_RADIAN_MULTIPLIER
    
    x_new = jnp.clip(x + dx_dt, env.min_x_position, env.max_x_position)
    y_new= jnp.clip(y + dy_dt, env.min_y_position, env.max_y_position)
    theta_new = theta + dtheta_dt

    collision = jnp.tanh(get_number_of_collisions(x_new, y_new, env))
    new_position = jnp.array([x_new, y_new, theta_new])
    old_position = jnp.array([x, y, theta])

    return (1-collision) * new_position + collision * old_position


# Transition function for Cartpole - Discrete and Continuous
def cartpole(state, actions, env):
    left_action, right_action = actions
    force = -env.force_mag * left_action + env.force_mag * right_action
    return _cartpole(env, state, force)


def continuous_cartpole_hybrid(state, action, env):
    if len(state) == 6:
        x, x_dot, theta, theta_dot, left_of_marker, normal_noise = state
        uniform_noise = 0
    else:
        x, x_dot, theta, theta_dot, left_of_marker, normal_noise, uniform_noise = state
    
    shaky_action = action[0] #if env.ignore_shaky_in_planner else action[0] + 5 * token * noise 
    force = env.force_mag * shaky_action

    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    force += env.alpha * normal_noise
    
    temp = (force + env.polemass_length * theta_dot ** 2 * sin_theta) / env.total_mass
    theta_acc = (env.gravity * sin_theta - cos_theta * temp) / (
            env.length * (4.0 / 3.0 - env.masspole * cos_theta ** 2 / env.total_mass))
    x_acc = temp - env.polemass_length * theta_acc * cos_theta / env.total_mass

    x = jnp.add(x, env.tau * x_dot)
    x_dot = x_dot + env.tau * x_acc
    theta = theta + env.tau * theta_dot
    theta_dot = theta_dot + env.tau * theta_acc
    
    p_left_of_marker = jax.nn.sigmoid(5.0*(env.reward_marker - x)) #is_between_smooth(x, env.x_shaky_threshold_left, env.x_shaky_threshold_right)
    # left_of_marker = left_of_marker #if env.ignore_bin_var_in_planner else jax.nn.sigmoid(5 * (p_left_of_marker - 0.5 - uniform_noise)) ## Sticky
    left_of_marker = jax.nn.sigmoid(5 * (p_left_of_marker - 0.5 - uniform_noise))

    # This is the JAX equivalent of the transition function in the simulator.
    # is_left_of_margin = jnp.where(x < env.unstable_left, jnp.ones_like(x), jnp.zeros_like(x)) 
    # p_left_of_margin = is_left_of_margin
    # if env.ignore_shaky_in_planner:
    #     left_of_margin = left_of_margin
    # else:
    #     left_of_margin = jnp.where(uniform_noise <= (p_left_of_margin - 0.5), jnp.ones_like(p_left_of_margin), jnp.zeros_like(p_left_of_margin))
    
    return jnp.array([x, x_dot, theta, theta_dot, left_of_marker])



def continuous_cartpole(state, action, env):
    force = env.force_mag * action
    return _cartpole(env, state, force[0])


def _cartpole(env, state, force):
    if len(state) == 4:
        x, x_dot, theta, theta_dot = state
        noise = 0
    else:
        x, x_dot, theta, theta_dot, noise = state

    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    force += env.alpha * noise
    temp = (force + env.polemass_length * theta_dot ** 2 * sin_theta) / env.total_mass
    theta_acc = (env.gravity * sin_theta - cos_theta * temp) / (
            env.length * (4.0 / 3.0 - env.masspole * cos_theta ** 2 / env.total_mass))
    x_acc = temp - env.polemass_length * theta_acc * cos_theta / env.total_mass

    x = jnp.add(x, env.tau * x_dot)
    x_dot = x_dot + env.tau * x_acc
    theta = theta + env.tau * theta_dot
    theta_dot = theta_dot + env.tau * theta_acc

    return jnp.array([x, x_dot, theta, theta_dot])


# Transition Function for Mountain Car - Discrete and Continuous
def mountain_car(state, actions, env):
    left, nothing, right = actions
    force = (right * env.force) + (left * - env.force) + nothing * 0
    return _mountain_car(env, state, force)


def continuous_mountain_car(state, actions, env):
    force = actions[0]    
    # Compute the next state
    force = jnp.clip(force, env.min_action, env.max_action) * env.power
    return _mountain_car(env, state, force)

def continuous_mountain_car_high_dim(state, actions, env):
    force = actions[0]    
    # Compute the next state
    force = jnp.clip(force, env.min_action, env.max_action) * env.power
    return _mountain_car(env, state, force)


def _mountain_car(env, state, force):
    if len(state) == 2:
        position, velocity = state
        noise = 0
    else:
        position, velocity, noise = state

    # Add noise
    force += env.alpha * noise
    cos_position = jnp.cos(3 * position)
    velocity = velocity + force - 0.0025 * cos_position

    velocity = jnp.clip(velocity, -env.max_speed, env.max_speed)
    position = position + velocity
    position = jnp.clip(position, env.min_position, env.max_position)

    return jnp.array([position, velocity])


def pendulum(state, action, env):
    if len(state) == 4:
        theta, cos_theta, sin_theta, thdot = state 
        noise = 0
    else:
        theta, cos_theta, sin_theta, thdot, noise = state 
    g = env.g
    m = env.m
    l = env.l
    dt = env.dt

    u = jnp.clip(action, -env.max_torque, env.max_torque)[0]

    newthdot = thdot + (3 * g / (2 * l) * jnp.sin(theta) + 3.0 / (m * l ** 2) * u) * dt
    newthdot = jnp.clip(newthdot, -env.max_speed, env.max_speed)
    newth = theta + (newthdot + env.alpha * jnp.exp(noise)) * dt

    return jnp.array([newth, jnp.cos(newth), jnp.sin(newth), newthdot])


@custom_jvp
def smooth_min(x, threshold):
    return jnp.minimum(x, threshold)

@smooth_min.defjvp
def smooth_min_jvp(primals, tangents):
    x, threshold = primals
    x_dot, threshold_dot = tangents
    ans = smooth_min(x, threshold)
    ans_dot = - jax.nn.sigmoid(threshold - x) * (x_dot)
    return ans, ans_dot


@custom_jvp
def smooth_max(x, threshold):
    return jnp.maximum(x, threshold)

@smooth_max.defjvp
def smooth_max_jvp(primals, tangents):
    x, threshold = primals
    x_dot, threshold_dot = tangents
    ans = smooth_max(x, threshold)
    ans_dot = jax.nn.sigmoid(x - threshold) * x_dot
    return ans, ans_dot

def is_between_limits(x, lower_limit, upper_limit):
    return 1 - jnp.tanh(1e7 * (jax.nn.relu(lower_limit - x) + jax.nn.relu(x - upper_limit)))

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

if __name__ == "__main__":
    print(jax.grad(smooth_max)(4.0, 5.0))
