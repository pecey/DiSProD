#!/usr/bin/env python3
import rospy
import numpy as np
import sys
import os
from collections import namedtuple
import tf
from datetime import datetime
import pickle


from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist 

DISPROD_PATH = os.getenv("DISPROD_PATH")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))



def render_model(x, y):
    os.system(
        'rosrun gazebo_ros spawn_model -urdf -model turtlebot3_burger -x {} -y {} -param robot_description'.format(
            x, y))



class Record:
    def __init__(self):
        rospy.init_node('recorder', anonymous=True)
        
        self.rate = rospy.Rate(5)
        self.recordOdom = False
        self.recordCmd = False
        self.chassisCmd = False
        self.path = os.path.join(DISPROD_PATH , "collected_data/turtlebot_updated")
        self.sub = rospy.Subscriber("cmd_vel" , Twist , self.commandCallback)
        self.sub1 = rospy.Subscriber("/odom" , Odometry , self.odomCallback)
        self.count = 0
        self.new_command = False
        self.memory = []
        self.command_linear = 0
        self.command_angular = 0


    def odomCallback(self , data):

        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        
        self.linear_vel = data.twist.twist.linear.x
        self.angular_vel = data.twist.twist.angular.z

        q0 = data.pose.pose.orientation.x
        q1 = data.pose.pose.orientation.y
        q2 = data.pose.pose.orientation.z
        q3 = data.pose.pose.orientation.w


        _ , _, self.yaw = tf.transformations.euler_from_quaternion([q0 , q1 , q2 , q3])

        self.odomTime = data.header.stamp

        self.recordOdom = True

    def commandCallback(self , data):
        if (self.command_linear - data.linear.x) ** 2 + (self.command_angular - data.angular.z) ** 2 > 0.3:
            self.new_command = True

        self.command_linear = data.linear.x 
        self.command_angular = data.angular.z
        

        self.recordCmd = True

    def observe(self):

        if not (self.recordCmd and self.recordOdom) or self.new_command:
            self.new_command = False
            print(f"New command is  , {self.command_linear} , {self.command_angular}")
            
            return False

        if self.command_angular == 0 and self.command_angular == 0:
            if np.random.random() > 0.1:
                rospy.logwarn("Since joystick is not giving any commands, not recording ...")
                return False
        
        self.state = np.array([self.x , self.y , self.yaw , self.linear_vel , self.angular_vel])

        self.action = np.array([self.command_linear , self.command_angular])
        return True 

    def step(self):
        self.count += 1
        print("Recording ", self.count)

        next_state = np.array([self.x , self.y , self.yaw , self.linear_vel , self.angular_vel ])

        self.memory.append(Transition(self.state , self.action , next_state , 0))

    def save_to_disk(self):

        y = input("Save?")
        if y != "y":
            return
        
        print("Saving to disk ...")
        txt = str(datetime.now()) + '.pkl'
        path = os.path.join(self.path , txt)
        with open(path, 'wb') as f:
            # store the data as binary data stream
            pickle.dump(self.memory, f)


if __name__ =='__main__':
    tw = Record()
    print("Starting recorder ...")
    x = (0.5 - np.random.random()) * np.random.randint(10)
    y = (0.5 - np.random.random()) * np.random.randint(10)
    render_model(x , y)
    rospy.sleep(2.0)
    rospy.on_shutdown(tw.save_to_disk)
    while not rospy.is_shutdown():
        status = tw.observe()
        tw.rate.sleep()
        if status:
            tw.step()
        

        
