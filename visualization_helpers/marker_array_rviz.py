# helper function to visualize the pose array
import tf
from geometry_msgs.msg import Pose,PoseArray
import rospy

from geometry_msgs.msg import Pose, Point,Vector3
import rospy
from .pose_array import PoseArrayViz
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA



class PoseArrayRviz(PoseArrayViz):
    def __init__(self, depth , restart, frame_id = "odom" , env = None):
        
        self.count = 0
        self.cmd_pub = rospy.Publisher('/odom_marker_array' , MarkerArray , queue_size = 1)
        self.frame_id = frame_id
        self._intialize_array()

    def pose_to_marker(self, pose , start, end):
        marker = Marker()
        marker.header.stamp = rospy.Time().now()
        #marker.pose = pose
        marker.type = Marker.ARROW
        marker.points = [start.position, end.position]
        marker.header.frame_id = self.frame_id
        marker.id = self.count
        self.count+=1

        v = Vector3(x = 0.1 , y = 0.2, z = 0.3)
        c = ColorRGBA(1,0,0,1)
        marker.scale = v
        marker.color = c

        return marker



    def publish(self , msg_list):
        
        msg_list = [self._get_pose(i) for i in msg_list]

        markers = [
            self.pose_to_marker(msg_list[i], start = msg_list[i] , end = msg_list[i + 1]) for i in range(len(msg_list) -1)
        ]

        self.msg.markers = markers


        self.cmd_pub.publish(self.msg)
        self.count = 0
        self._intialize_array()

    
    def get_array(self):
        msg = MarkerArray()
        return msg

    def _intialize_array(self):
        self.msg = self.get_array()
        self.msg_list = []

    def _get_pose(self, seq_pose):

        x, y, theta = seq_pose[:3]
        roll = 0
        pitch = 0
        pose = Pose()
        pose.position.x = x 
        pose.position.y = y 
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, theta)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        return pose 
