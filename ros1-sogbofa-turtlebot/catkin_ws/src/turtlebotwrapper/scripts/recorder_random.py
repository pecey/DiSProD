#!/usr/bin/env python3
import rospy
import numpy as np
import sys
import os
from collections import namedtuple
import tf
from datetime import datetime
import pickle
from tqdm import tqdm


from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist 

DISPROD_PATH = os.getenv("DISPROD_PATH")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

DEGREE_TO_RADIAN_MULTIPLIER = np.pi / 180



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
        self.path = os.path.join(DISPROD_PATH , "collected_data/turtlebot_delta_model")
        self.cmd_pub = rospy.Publisher('/cmd_vel',Twist, queue_size=10)
        self.sub1 = rospy.Subscriber("/odom" , Odometry , self.odomCallback)
        self.count = 0
        self.new_command = False
        self.memory = []
        self.command_linear = 0
        self.command_angular = 0


    def odomCallback(self , data):

        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        
        self.actual_linear_vel = data.twist.twist.linear.x
        self.actual_angular_vel = data.twist.twist.angular.z

        q0 = data.pose.pose.orientation.x
        q1 = data.pose.pose.orientation.y
        q2 = data.pose.pose.orientation.z
        q3 = data.pose.pose.orientation.w


        _ , _, self.yaw = tf.transformations.euler_from_quaternion([q0 , q1 , q2 , q3])

        self.odomTime = data.header.stamp

        self.recordOdom = True
    
    def setBaseVelocities(self , data):
        self.baseLinear = data[0]
        self.baseAngular = data[1] * DEGREE_TO_RADIAN_MULTIPLIER

    def issueDeltaAction(self):

        
        self.action0 = np.random.uniform(-0.1 , 0.1)
        self.action1 = np.random.uniform(-6 , 6) * DEGREE_TO_RADIAN_MULTIPLIER
        

        self.recordCmd = True

    def observe(self):

        if not (self.recordOdom):            
            return False

        linear_vel = self.baseLinear + self.action0
        angular_vel = self.baseAngular + self.action1
        
        self.state = np.array([self.x , self.y , self.yaw , self.actual_linear_vel , self.actual_angular_vel])

        self.action = np.array([self.action0 , self.action1])

        

        cmd = generate_command_message([linear_vel , angular_vel])
        self.cmd_pub.publish(cmd)
        return True 

    def step(self):
        self.count += 1
        print("Recording ", self.count)

        next_state = np.array([self.x , self.y , self.yaw , self.actual_linear_vel , self.actual_angular_vel ])
        
        self.memory.append(Transition(self.state , self.action , next_state , 0))

    def save_to_disk(self):

        
        print("Saving to disk ...")
        txt = str(datetime.now()) + '.pkl'
        path = os.path.join(self.path , txt)
        with open(path, 'wb') as f:
            # store the data as binary data stream
            pickle.dump(self.memory, f)


def generate_command_message(action):
    cmd = Twist()
    cmd.linear.x = cmd.linear.y = cmd.linear.z = 0.0
    cmd.angular.x = cmd.angular.y = cmd.angular.z = 0.0
    cmd.linear.x = max(0 , action[0])
    cmd.angular.z = action[1]
    return cmd


if __name__ =='__main__':
    tw = Record()
    print("Starting recorder ...")
    x = (np.random.random()) * np.random.randint(10)
    y = (np.random.random()) * np.random.randint(10)
    render_model(x , y)
    rospy.sleep(2.0)
    lin_vel_space = np.linspace(0 , 0.5 , 6)[::-1]
    ang_vel_space = np.linspace(-60 , 60 , 21)

    while not rospy.is_shutdown():
        for base_lin_vel in tqdm(lin_vel_space):
            for base_ang_vel in ang_vel_space:
                tw.setBaseVelocities([base_lin_vel , base_ang_vel])
                count = 0
                while count != 1500:
                    count += 1
                    tw.issueDeltaAction()
                    status = tw.observe()
                    tw.rate.sleep()
                    if status:
                        tw.step()

        tw.save_to_disk()
                    

        
