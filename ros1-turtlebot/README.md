 ## Setup
 
- We assume that ROS Noetic Desktop-Full (`ros-noetic-desktop-full`) is already installed on the system. 
- We use the package `tf` in [`visualization_helpers`](../visualization_helpers/) which is installed as a part of `ros-noetic-desktop-full`.

Reference : [Turtlebot3 Tutorial](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/)

#### Pre-requisite
```shell
# Install the package for Turtlebot3 Simulations.
sudo apt install ros-noetic-turtlebot3-simulations
```
If `DISPROD_PATH` is not set in `~/.bashrc`, then add the following:
```shell
export DISPROD_PATH=<path-to-DiSProD>
```
and then `source ~/.bashrc`.

In order to use PID as the low level controller, clone [this repository](https://github.com/itsmeashutosh43/pid-heron) inside `$DISPROD_PATH/ros1-turtlebot/catkin_ws/src`.

```shell
cd $DISPROD_PATH/ros1-turtlebot/catkin_ws/src
git clone https://github.com/itsmeashutosh43/pid-heron
```

#### Build the files
```shell
cd $DISPROD_PATH/ros1-turtlebot/catkin_ws
rm -rf build devel
catkin_make
cd devel && source setup.sh
```

#### Lowering the simulation time
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

#### Set x_pose, y_pose from config file

To set the x,y coordinate of the turtlebot from the config file, 

```shell
roscd turtlebot3_gazebo/launch
vim turtlebot3_empty_world.launch
```

and remove the following line (It should be towards the bottom of the document in the last few lines.) 

```shell
<node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model" args="-urdf -model turtlebot3 -x $(arg x_pos) -y $(arg y_pos) -z $(arg z_pos) -param robot_description" />

```

## Running the simulation and the planner

1. Run Gazebo in terminal (T1).
```shell
# Terminal 1: Launch Gazebo
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch
```

2. Run RViz in a separate terminal (T2).
```shell
# Terminal 2: Launch RViz
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch
```

3. Run the planner (T3)
```shell
# Terminal 3: Run the planner to control the bot directly
rosrun turtlebotwrapper planwrapper.py --alg=disprod --map_name=no-ob-1
```

## Using PID

By default, the agent controls the TurtleBot directly by publishing the action commands to `/cmd_vel`. However, it is possible to use a low level controller such as PID to issue action commands. In this case, the agent sends a sequence of waypoints to the PID controller. 

4. Start the PID controller first in a separate terminal (T4)

```shell
# Terminal 4: Start the PID controller node
cd $DISPROD_PATH/ros1-turtlebot/catkin_ws/src/pid-heron/scripts
python3 tracking_pid_node.py
```

5. And run the planner (T3) using `--control=pid` argument.

```shell
# Terminal 3: Run the planner with control set to pid
rosrun turtlebotwrapper planwrapper.py --alg=disprod --map_name=no-ob-1 --control=pid
```


## Configuration Options

- `--seed`: Seed for the Pseudo Random Number Generator (PRNG).
- `--alg`: The planner to be used. Valid values are `disprod`,`cem` and `mppi`.
- `--pose_viz`: A flag that indicates whether to plot the trajectory being imagined by the planner. 
- `--map_name`: The name of the map to be used. Valid values are listed below.
- `--run_name`: The name of the run to identify different runs. 
- `--vehicle_type`: The type of vehicle to be used. Valid values are `turtlebot` and `uuv`.
- `--control`: The type of control to be used. Valid values are `self` and `pid`.
- `--skip_waypoints`: The number of waypoints to be skipped before sending to the PID controller.

> Note: If `--skip_waypoints==1`, then the PID controller will be slow as it will try to match waypoints that are clustered together.

#### Map configrations
All the map configurations are defined in [env/assets/dubins.json](../env/assets/dubins.json). At present, the available maps are `no-ob-[1-5]`, `ob-[1-11]`, `u` and `cave-mini`.

## Using DiSProD-NV mode

To run the experiments using DiSProD-NV, set `taylor_expansion_mode = no_var` in `config/continuous_dubins_car.yaml`

