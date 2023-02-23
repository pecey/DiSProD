# helper function to visualize the pose array
import abc
from geometry_msgs.msg import Pose
import tf
import numpy as np

class PoseArrayViz(abc.ABC):
    def __init__(self, depth , restart, env = None):
        self.depth = depth
        self.restart = restart
        self._intialize_array()
        self.env = env

    def _intialize_array(self):
        # self.all_restart_depth = {'x':torch.zeros(self.depth, self.restart).double() , 'y' : torch.zeros(self.depth, self.restart).double() \
        #     , 'theta':torch.zeros(self.depth, self.restart).double()}
        
        self.msg = self.get_array()
        self.msg_list = []

    @abc.abstractmethod
    def get_array(self):
        pass

    def record_restart_depth(self, array, k):

        x , y , theta = array

        
        # array of length restart 1 for depth k

        self.all_restart_depth['x'][k,:] = x
        self.all_restart_depth['y'][k,:] = y
        self.all_restart_depth['theta'][k,:] = theta

    def get_record_restart_depth(self,idx):
        print(self.all_restart_depth['x'] , self.all_restart_depth['y'] , self.all_restart_depth['theta'])
        return self.all_restart_depth['x'][:,idx] , self.all_restart_depth['y'][:,idx] , self.all_restart_depth['theta'][:,idx]
        

    def add_pose_list(self, seq_pose_list, indexes_to_select):
        pose_list = [self._get_pose(np.take(el, indexes_to_select)) for el in seq_pose_list]
        self.msg_list.extend(pose_list)

    def add_pose(self,seq_pose):
        pose = self._get_pose(seq_pose)
        self.msg_list.append(pose)

    
    def _get_pose(self, seq_pose):
        
        x, y, theta = seq_pose
        roll = 0
        pitch = 0

        if self.env is not None:
            '''
            then its gym, we only plot x,y
            '''
            return (x,y)

        
        pose = Pose()

        
        pose.position.x = x 
        pose.position.y = y 
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, theta)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        return pose 

    @abc.abstractmethod    
    def publish(self):
        pass


    
