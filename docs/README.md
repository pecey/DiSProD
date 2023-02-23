# Tutorial

This tutorial provides a step-by-step guide to run AutoRally and SOGBOFA on a virtual machine, which might help keeping everyone on the same page and hopefully facilitating the process of integration. My host machine is a Windows 10 but hopefully there is little difference when running on Mac or Linux.

## Installation of Ubuntu on VirtualBox

Considering that SOGBOFA requires Python3, we will use Ubuntu 20.04. Its corresponding ROS version - noetic - supports Python3 by default (in comparison, melodic supports Python 2). First, download Ubuntu 20.04.2.0 LTS from https://ubuntu.com/download/desktop. Then, install VirtualBox from https://www.virtualbox.org/ and setup a virtual machine. There are many tutorials online for this step. The following one is an example.

* https://brb.nci.nih.gov/seqtools/installUbuntu.html.

If you find the window size of the virtual machine too small, install the following packages first.

```bash
sudo apt-get update
sudo apt-get install build-essential gcc make perl dkms terminator
# Actually, `terminator` is not related here. It's just a more convenient terminal application compared to the built-in one.
```

Then, click `Devices` and select `Insert Guest Additions CD image`. The window size will be adjustable after restarting the system. If you encounter issues during this process, the following link might be helpful:

* https://askubuntu.com/a/960324.



## Installation of Miniconda

Download Miniconda-Python3.9 from https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh.

Follow the installation instructions: https://conda.io/projects/conda/en/latest/user-guide/install/linux.html

```bash
cd ~/Downloads
bash Miniconda3-py39_4.9.2-Linux-x86_64.sh
# Keep pressing `Enter` until you have to answer `yes`

# Do you wish the installer to initialize Miniconda3
# by running conda init? [yes|no]
# [no] >>> yes

# We will use the `base` environment throughout this tutorial.
# We do no need to run any command since this is the default setting:
# conda config --set auto_activate_base true
```


## Installation of ROS Noetic

```bash
# Setup your computer to accept software from packages.ros.org
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# Setup your apt-key
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# Update software information
sudo apt update
# Install ROS Noetic. This step is time consuming. Go for a walk :)
sudo apt install ros-noetic-desktop-full
# Automatically source the system-level setup.bash file everytime when we launch a terminal
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
# Install some packages that will be helpful for building and managing ROS workspace
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y
# `rosdep` helps us install system dependencies before installing some packages
# We have to do the one-time initialization for `rosdep`
sudo rosdep init
rosdep update
```


## Installation of AutoRally

1. Create `catkin_ws`

   ```bash
   cd ~
   mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws/src
   ```

2. Clone a modified version of autorally `git clone https://github.com/sogbofa-autorally/junior-autorally.git`

   1. If you would like to start from scratch and repeat what I've modified, here are the detailed steps
   2. Clone autorally repository `git clone https://github.com/AutoRally/autorally.git`
   3. Delete everything related to `autorally_control`

   ```bash
   cd ~/catkin_ws/src/autorally
   rm -rf ./autorally_control
   
   # Remove the following lines from `autorally_util/setupEnvVariables.sh`
   if [[ $MASTER_HOSTNAME == "localhost" ]]
       then
           export AR_MPPI_PARAMS_PATH=`rospack find autorally_control`/src/path_integral/params/
       else
           export AR_MPPI_PARAMS_PATH=`ssh $user@$MASTER_HOSTNAME rospack find autorally_control`/src/path_integral/params/
   fi
   
   # Delete line 34 from `autorally/package.xml`
   <run_depend>autorally_control</run_depend>
   
   # Delete line 88 from `catkin_ws/src/autorally/autorally_gazebo/launch/autoRallyTrackGazeboSim.launch`
   <include file="$(find autorally_control)/launch/joystickController.launch" />
   ```

3. Install system dependencies

   ```bash
   cd ~/catkin_ws
   rosdep install --from-path src --ignore-src -y
   ```

4. Install `geographiclib`
   Download from https://sourceforge.net/projects/geographiclib/files/distrib/GeographicLib-1.51.tar.gz/download

   ```bash
   cd ~/Downloads
   # Extract tar ball
   tar xfpz GeographicLib-1.51.tar.gz
   cd GeographicLib-1.51/
   # Build
   mkdir build
   cd build
   cmake ..
   # Compilation. Feel free to use -j#threads to speedup this process if you have multiple CPU cores
   make
   # Installation
   sudo make install
   ```

5. Compilation

   ```bash
   pip install empy catkin_pkg pyyaml rospkg numpy defusedxml
   cd ~/catkin_ws
   catkin_make
   ```

6. Setup environment variables

   ```bash
   # Add the following two lines to your ~/.bashrc and restart the terminal
   source ~/catkin_ws/devel/setup.sh
   source ~/catkin_ws/src/autorally/autorally_util/setupEnvLocal.sh
   # You will receive warning about `No joystick detected`. That's fine.
   ```

7. Install controller

   ```bash
   cd ~/catkin_ws/src
   git clone https://github.com/sogbofa-autorally/naive_controller.git
   cd ..
   catkin_make
   . devel/setup.bash
   ```

8. Split the terminal via `ctrl+shift+e` or `ctrl+shift+o` and run the following commands  on two terminals

   ```bash
   roslaunch autorally_gazebo autoRallyTrackGazeboSim.launch
   ```

   ```bash
   rosrun naive_controller controller.py
   ```

   The car will start moving forward. Yay! Running Gazebo on a virtual machine is very slow though :(

## Get A Sense of SOGBOFA

1. Clone `awesome-sogobofa`

   ```bash
   cd ~
   git clone https://github.com/Weizhe-Chen/awesome-sogbofa.git
   ```

2. Install some Python packages used by SOGBOFA

   ```bash
   pip install gym matplotlib Pillow sympy torch
   ```

3. Setup Python path

   ```bash
   # Add this line to your ~/.bashrc. This is not a recommended practice but we will bear with it for simplicity.
   export PYTHONPATH=$PYTHONPATH:~/awesome-sogbofa/ros1-sogbofa-turtle/catkin_ws/src
   # Don't forget to source bashrc after any modification
   source ~/.bashrc
   # and our old friend setup.bash
   . devel/setup.bash
   ```

4. Compilation

   ```bash
   cd ~/awesome-sogbofa/ros1-sogbofa-turtle/catkin_ws
   catkin_make
   ```

5. Run `roscore`

   ```bash
   . devel/setup.bash
   roscore
   ```

6. Run TurtleSim

   ```bash
   # Open another terminal via Ctrl+shift+o
   . devel/setup.bash
   rosrun turtlesim turtlesim_node
   # open another terminal via ctrl+shift+e
   . devel/setup.bash
   rosrun turtlewrapper planwrapper.py cont-sogbofa 7 9
   ```

   The adorable turtle will start moving but it seems to have a hard time reaching the goal :-)

