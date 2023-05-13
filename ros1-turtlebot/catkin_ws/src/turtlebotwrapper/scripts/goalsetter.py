#!/usr/bin/env python
# Software License Agreement (BSD License)

import rospy
import sys
from geometry_msgs.msg import Point

def main():
    pub = rospy.Publisher('/turtlewrapper/GoalLocation',Point, queue_size=10)
    rospy.init_node('goalsetter', anonymous=True)
    
    if (len(sys.argv)  == 3): 
        msg=Point()
        msg.x = float(sys.argv[1])
        msg.y = float(sys.argv[2])
        msg.z = 0.0
        rospy.loginfo('Planner Goal is set to ("%f","%f") ' % (msg.x,msg.y))
        pub.publish(msg)

        
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
