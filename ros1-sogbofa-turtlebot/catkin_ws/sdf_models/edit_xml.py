import xml.etree.ElementTree as ET

pose_x, pose_y, pose_z = 2 , 5 , 0
size_x, size_y, size_z = 0.5 , 0.5 , 1
model = 'box'

tree = ET.parse('{}.sdf'.format(model))
root = tree.getroot()


for pose in root.iter('pose'):
    
    print(pose.text)
    pose.text = '{} {} {} 0 0 0'.format(pose_x , pose_y , pose_z)
    print(pose.text)


for size in root.iter('size'):
    print(size.text)
    size.text = '{} {} {}'.format(size_x , size_y , size_z)
    print(size.text)


file_name = '{}_pose_{}_{}_size_{}_{}_{}.sdf'.format(model, pose_x, pose_y , size_x , size_y, size_z) 
tree.write(file_name)

import os 

os.system("rosrun gazebo_ros spawn_model -file ./{} -sdf -model box_target_red".format(file_name))

os.remove("./{}".format(file_name))