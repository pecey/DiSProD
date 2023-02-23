## Reference : [Turtlebot3 Tutorial](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/)

## Environment configuration

The environment configuration is defined in `env/assets/dubins.json`. 

## Pre-requisite
```shell
# Download Turtlebot3 Simulation in the src folder.
cd ros1-sogbofa-turtlebot/catkin_ws/src
git clone -b kinetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
# OR,
sudo apt install ros-kinetic-turtlebot3-simulations
```
Add the following exports to `~/.bashrc` file:
```shell
export DISPROD_PATH=<path-to-awesome-sogbofa>
```
Don't forget to source the updated file.

## Build the files
```shell
rm -rf build devel
catkin_make
cd devel && source setup.sh
```

## Lowering the simulation time
> Note: Running SOGBOFA is a computationally extensive task, therefore we should lower the simulation time to see it working. 
```
roscd turtlebot3_gazebo/worlds/
vim empty.world
```

And update the world file with the following settings inside the physics tag. More on the dynamics about these three parameters can be found [here](http://gazebosim.org/tutorials?tut=physics_params&cat=physics)

```
<real_time_update_rate>200.0</real_time_update_rate>
<max_step_size>0.001</max_step_size>
<real_time_factor>0.2</real_time_factor>
```

if time step size is 1ms, throttling at 200 steps/second effectively
throttles simulation down to .2X real-time.

## Set x_pose, y_pose from config file

If you need to be able to set the x,y coordinate of the turtlebot from the config file, 

```shell
roscd turtlebot3_gazebo/launch
vim turtlebot3_empty_world.launch
```

and remove the line(you may find it in the last few lines) 

```shell
<node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model" args="-urdf -model turtlebot3 -x $(arg x_pos) -y $(arg y_pos) -z $(arg z_pos) -param robot_description" />

```

## Running the simulation
> Note: `roscore` is not required for Turtlebot. Gazebo starts a ROS Master.
```shell
# Terminal 1: Launch Gazebo
# For headless mode, see example below
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch

# Terminal 2: Launch RViz
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch
```

To launch Gazebo in headless mode:
- Modify the corresponding launch file. (eg: `ros1-sogbofa-turtlebot/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch/turtlebot3_empty_world.launch`)
```xml
    <arg name="gui" value="false"/>
    <arg name="headless" value="true"/>
```
Other worlds: `turtlebot3_world.launch`, `turtlebot3_house.launch`

The goal and obstacle are published in separate topics. To visualize them, they would need to be added separately in RViz. The process to do so is as follows:

- Click on `Add` (near bottom left corner of screen). Then `By topic`, and select the corresponding topics (`goal_marker` and `obstacles`). 
> It may take some time before these topics appear in the list. But it should appear no later than after the first action is taken.

## Running the planner

There are a set of configurations for the model of the planner

### Dubins Model

This is the naive standard dubins model which was used to solve this problem in gym, but this model doesn't give good 
performance on asynchronous environment like gazebo. 

To run in this mode, 

set config/planning/continuous_dubins_car = dubins_car

and run,
```shell
# Terminal 3
rosrun turtlebotwrapper planwrapper.py --planner cont-sogbofa --env continuous_dubins_car
```

### Clip Dubins Model

This is an improvement to the naive standard dubins model where we add velocity to the states.
To run in this mode, 

set config/planning/continuous_dubins_car_w_velocity = clip_dubins_car

and run,
```shell
# Terminal 3
rosrun turtlebotwrapper planwrapper.py --planner cont-sogbofa --env continuous_dubins_car_w_velocity
```



### Learnt Model

This is a ML based learned model 

set config/planning/continuous_dubins_car_w_velocity = learning

and run,
```shell
# Terminal 3
rosrun turtlebotwrapper planwrapper.py --planner cont-sogbofa --env continuous_dubins_car_w_velocity
```



## Resetting the bot position
To reset the bot position while using Gazebo:
```shell
rosservice call /gazebo/reset_simulation "{}"
```

## Visualize the bot trajectory

### Installing dependencies (One time)
```shell
sudo apt install ros-noetic-hector-trajectory-server # replace noetic by melodic if you have a melodic version
sudo apt install ros-noetic-hector-geotiff-launch
```

### Editing Files (One Time)
```shell
roscd hector_geotiff_launch/launch
sudo vim geotiff_mapper.launch

# Change the param value "target_frame_name" inside "hector_trajectory_server" from /map to /odom, lower the "trajectory_update_rate" for faster updates
```

### Launcher file
Open a new terminal to visualize nav_msgs/Path in rviz
```shell
roslaunch hector_geotiff_launch geotiff_mapper.launch
```

# Common errors:
```shell
RLException: [turtlebot3_gazebo_rviz.launch] is neither a launch file in package [turtlebot3_gazebo] nor is [turtlebot3_gazebo] a launch file name
```
Solution : 
- Ensure you are building the correct project.
- Ensure you have sourced the setup file after building.

