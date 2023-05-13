

This demo includes a python wrapper interface for turtlesim in ROS 1 Melodic/Noetic 

Works with trivial planner and sogbofa.

------

## Installs and setup

1: You may already have these but just in case

`sudo apt-get install ros-melodic-ros-tutorials ros-melodic-rqt-common-plugins`

2: Then for sogbofa we need python3 support and following libraries

```shell
sudo apt install python3-pip
pip3 install numpy gym Pillow setuptools sympy matplotlib torch rospkg omegaconf
```

3: Then: 
- ROS Melodic
It seems  to work through python 2.7 but we need python3.
Following this link 
    https://dhanoopbhaskar.com/blog/2020-05-07-working-with-python-3-in-ros-kinetic-or-melodic/

We can do the following (works in some cases, and worked for turtlesim demo). 

```shell
sudo apt install python3-pip python3-all-dev
sudo apt install ros-melodic-desktop-full --fix-missing
```

- ROS Noetic: http://wiki.ros.org/Installation/Ubuntu

4: Add the following exports to `~/.bashrc` file:
```shell
export DISPROD_PATH=<path-to-awesome-sogbofa>
```

------

## Using the planners

```shell
cd catkin_ws/
rm -rf build devel 
catkin_make 
# This has to be run in any of the terminals where we want to invoke turtlewrapper
cd devel && source setup.bash
```


Use 3 to 4 terminals as follows:

- T1: Start ROS Master, ROS parameter server and rosout logging node : `roscore`    

- T2: Start the turtlebot simulator : `rosrun turtlesim turtlesim_node`    

- T3: 
  - Trivial planner: will wait for more goals assignments on corresponding topic : `rosrun turtlewrapper planwrapper.py naive`   
  - Trivial planner: will start interface and go to 7 9 then will wait for more goals assignments on corresponding topic : `rosrun turtlewrapper planwrapper.py --planner naive --goal 7 9`
  - Sogbofa planner: will start interface and go to 7 9 then will wait for more goals assignments on corresponding topic : `rosrun turtlewrapper planwrapper.py --planner sogbofa --goal 7 9`
  - Continuous Sogbofa planner: will start interface and go to 7 9 then will wait for more goals assignments on corresponding topic : `rosrun turtlewrapper planwrapper.py --planner cont-sogbofa --goal 7 9` 

- T4: Goal setter: `rosrun turtlewrapper goalsetter.py 1 3`

