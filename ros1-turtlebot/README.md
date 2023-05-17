## Reference : [Turtlebot3 Tutorial](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/)

## Environment configuration

The environment configuration is defined in `env/assets/dubins.json`. 

## Pre-requisite
```shell
# Install the package for Turtlebot3 Simulations.
sudo apt install ros-noetic-turtlebot3-simulations
```
Add the following exports to `~/.bashrc` file:
```shell
export DISPROD_PATH=<path-to-DiSProD>
```
Don't forget to source the updated file.

In order to use PID as the low level controller, clone [this repository](https://github.com/itsmeashutosh43/pid-heron) inside `$DISPROD_PATH/ros1-turtlebot/catkin_ws/src`.

```shell
cd $DISPROD_PATH/ros1-turtlebot/catkin_ws/src
git clone https://github.com/itsmeashutosh43/pid-heron
```

## Build the files
```shell
cd $DISPROD_PATH/ros1-turtlebot/catkin_ws
rm -rf build devel
catkin_make
cd devel && source setup.sh
```


## Lowering the simulation time
> Note: Running DiSProD is a computationally extensive task, therefore we should lower the simulation time to see it working. 

```shell
roscd turtlebot3_gazebo/worlds/
vim empty.world
```

And update the world file with the following settings inside the physics tag. More on the dynamics about these three parameters can be found [here](http://gazebosim.org/tutorials?tut=physics_params&cat=physics)

```
<real_time_update_rate>200.0</real_time_update_rate>
<max_step_size>0.001</max_step_size>
<real_time_factor>0.2</real_time_factor>
```

If time step size is 1ms, throttling at 200 steps/second effectively
throttles simulation down to .2X real-time.

## Set x_pose, y_pose from config file

To set the x,y coordinate of the turtlebot from the config file, 

```shell
roscd turtlebot3_gazebo/launch
vim turtlebot3_empty_world.launch
```

and remove the following line (It should be towards the bottom of the document in the last few lines.) 

```shell
<node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model" args="-urdf -model turtlebot3 -x $(arg x_pos) -y $(arg y_pos) -z $(arg z_pos) -param robot_description" />

```

## Running the simulation
```shell
# Terminal 1: Launch Gazebo
# For headless mode, see example below
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch

# Terminal 2: Launch RViz
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch
```

## Running the planner

### Running in self mode
```
rosrun turtlebotwrapper planwrapper.py --alg disprod --map_name no-ob-1
```

### Running in pid mode

In case of using the PID controller, run this in one terminal

```shell
cd $DISPROD_PATH/ros1-turtlebot/catkin_ws/src/pid-heron/scripts
python3 tracking_pid_node.py
```

and 

```
rosrun turtlebotwrapper planwrapper.py --alg disprod --map_name no-ob-1 --pose_viz False --control pid
```


### Configuration Options

- `--log_file`: The path to the log file. If not specified, no log file will be created.
- `--seed`: Seed for the Pseudo Random Number Generator (PRNG). Default is `42`.
- `--env`: The environment to be used. Default is `continuous_dubins_car_w_velocity`. Note: this configuration can also be used for boat experiments.
- `--noise`: A flag that indicates whether to add noise to the system. Default is `False`.
- `--alg`: The algorithm to be used. Default is `disprod`. Options are `mppi`, `cem`, and `disprod`.
- `--poseVisualization`: A flag that indicates whether to visualize the pose of the system. Default is `True`.
- `--map_name`: The name of the map to be used. The default map configs are located in `env/assets/dubins.json`
- `--run_name`: The name of the run. Can be used to identify different runs. No default value.
- `--vehicle_type`: The type of vehicle to be used. Default is `turtlebot`. Options are `turtlebot` and `uuv`.
- `--control`: The type of control to be used. Default is `self`. Options are `self` (publishes message to `/cmd_vel`) and `pid` (publishes to the PID controller).
- `--skip_waypoints`: The number of waypoints to be skipped before sending to the PID controller. Default is `1`. Relevant only if using PID controller. If set to `1`, the PID controller will be slow as it will try to match waypoints that are clustered together.

### No-var mode

To run the experiments in no variance mode, set `taylor_expansion_mode = no_var` in `config/continuous_dubins_car_w_velocity.yaml`
