#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
import rospy
import numpy as np
import sys
import torch as T

from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point

# note: this fails for local import so I moved it to global Python path
#from .planalg import planalg
from oldplanners import planalg

from oldplanners.dubins_interface import setupAgent
from oldplanners.dubins_interface import sogbofaPlanOneStep

DEGREE_TO_RADIAN_MULTIPLIER = np.pi / 180

class EnvironmentProperties:
    def __init__(self):
        self.default_velocity = 0.5
        self.turning_velocity = 0.1
        self.steering_angle = 15
        #self.goal_boundary = 0.2
        self.goal_boundary = 0.3
        # Delta t
        #self.time_interval = 0.2
        self.time_interval = 1.5
        self.min_x_position, self.max_x_position = 0, 11
        self.min_y_position, self.max_y_position = 0, 11
        self.goal_x, self.goal_y = 1, 5

class turtlewrapper():
    def __init__(self, debug=False ,ser= None):
        rospy.init_node('turtlewrapper', anonymous=True)
        rospy.Subscriber('/turtle1/pose',Pose, self.pose_listener_callback)
        rospy.Subscriber('/turtlewrapper/GoalLocation',Point, self.goal_listener_callback)
        self.pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        #self.rate = rospy.Rate(5) # hz ie 0.2 sec
        self.rate = rospy.Rate(5) # hz ie 0.2 sec
        #self.timer = self.create_timer(timer_period, self.planner_callback)
        self.useSogbofa=False
        self.GoalSet=False
        self.goalx=1.0
        self.goaly=5.0
        self.currentpose=np.array([0.0,0.0,0.0,0.0,0.0])

    def goal_listener_callback(self,msg):
        rospy.loginfo('Goal Received: ("%f","%f")' %(msg.x,msg.y))
        self.goalx=msg.x
        self.goaly=msg.y
        self.GoalSet=True

    def pose_listener_callback(self,msg):
        #rospy.loginfo('Pose Received: ("%f","%f")' %(msg.x,msg.y))
        self.currentpose=np.array([msg.x,msg.y,msg.theta,msg.linear_velocity,msg.angular_velocity])

    def planner_callback(self):
        if (self.GoalSet == False):
            #rospy.loginfo('Goal not set, nothing to do')
            return
        currentlocation=self.currentpose[:2]
        currenttheta=self.currentpose[2]
        currentgoal=np.array([self.goalx,self.goaly])

        logstring = ""
        logstring +='Goal: ("%f,%f")\n' % (self.goalx,self.goaly)
        logstring +='Location: ("%f,%f")\n' % (self.currentpose[0],self.currentpose[1])
        rospy.loginfo(logstring)

        delta=currentgoal-currentlocation
        dist=np.linalg.norm(delta)
        #logstring = ""
        #logstring +='Distance to goal: "%f"\n' % dist
        #rospy.loginfo(logstring)
        
        if (dist< 0.2):
            rospy.loginfo('Arrived at Goal')
            logstring = ""
            logstring +='Goal: ("%f,%f")\n' % (self.goalx,self.goaly)
            logstring +='Location: ("%f,%f")\n' % (self.currentpose[0],self.currentpose[1])
            rospy.loginfo(logstring)
            self.GoalSet = False
            return
        
        cmd=Twist()
        cmd.linear.x=cmd.linear.y=cmd.linear.z=0.0
        cmd.angular.x=cmd.angular.y=cmd.angular.z=0.0

        state = np.zeros(3)
        state[:2]=currentlocation
        state[2]=currenttheta

        #logstring,myaction=planalg.planalg(state,currentgoal)

        if self.useSogbofa:             
            logstring, myaction = sogbofaPlanOneStep(self.agent, self.env_to_be_passed, state, currentgoal)
            myaction[1] *= DEGREE_TO_RADIAN_MULTIPLIER
        else:
            logstring,myaction=planalg.planalg(state,currentgoal)

        rospy.loginfo(logstring)
        cmd.linear.x=myaction[0]
        cmd.angular.z=myaction[1]
        self.pub.publish(cmd)


def main():
    tw = turtlewrapper()
    
    if (len(sys.argv)> 1 and sys.argv[1]=="sogbofa"):
        config = {
            'env_name': 'dubins_car',
            'env_entry_point': 'env.dubins_env:DubinsEnv',
            'add_noise': "true",
            'max_episode_steps': 500,
            'depth': 50,
            'lr': 1e-3,
            'n_restarts': 10,
            'log_file': None,
            'n_episodes': 1,
            'device': "cuda" if T.cuda.is_available() else "cpu"
        }
        tw.env_to_be_passed = EnvironmentProperties()
        tw.useSogbofa = True
        tw.agent = setupAgent(tw.env_to_be_passed, config)
        if (len(sys.argv)  > 3 ): 
            tw.goalx = float(sys.argv[2])
            tw.goaly = float(sys.argv[3])
            tw.GoalSet = True            
    elif (len(sys.argv)  > 2 ): 
        tw.goalx = float(sys.argv[1])
        tw.goaly = float(sys.argv[2])
        tw.GoalSet = True

    while not rospy.is_shutdown():
        tw.planner_callback()
        tw.rate.sleep()
        
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
