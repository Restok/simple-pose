import numpy as np
import json
from preprocess_data import *
import os
import multiprocessing as mp
from euler_to_rot import rotation_matrix_to_euler_zyz, euler_to_rotation_matrix_zyz
def rot_mat_to_euler(R):
    a = np.arcsin(R[0,2])
    b = np.arctan2(R[0,1]/-np.cos(a), R[0,0]/np.cos(a))
    c = np.arctan2(R[1,2]/-np.cos(a), R[2,2]/np.cos(a))
    if abs(R[0,0])<10^-16 or abs(R[2,2])<10^-16:
        print("Matrix has infinite Solutions")
    return c,a,b

subjects = ["s03", "s04", "s05", "s07", "s08", "s09", "s11"]
# cameras = ["50591643/", "58860488/", "60457274/", "65906101/"]

root = "C:/Users/DAQ/Desktop/simple-pose/simplepose/"
base = root+"data/fit3d_train/train/"

if not os.path.exists("processed"):
    os.makedirs("processed")

convert_to_angles = True

for subject in subjects:
    print(subject)
    if not os.path.exists(root+"processed/" + subject):
        os.makedirs(root+"processed/" + subject)
    if not os.path.exists(root+"processed/" + subject + "/smplx"):
        os.makedirs(root+"processed/" + subject + "/smplx")
    if not os.path.exists(root+"processed/" + subject + "/smplx_mat"):
        os.makedirs(root+"processed/" + subject + "/smplx_mat")
    actions = os.listdir(base + subject + "/smplx")
    for action_data in actions:
      f = open(base + subject + "/smplx/" + action_data)
      smplx = json.load(f)
      global_orient = np.array(smplx['global_orient'])
      body_pose = np.array(smplx['body_pose'])
      transl = np.array(smplx['transl'])
      seq_len = global_orient.shape[0]
      if convert_to_angles:
        global_orient = rotation_matrix_to_euler_zyz(global_orient)
        body_pose = rotation_matrix_to_euler_zyz(body_pose)
        # global_orient_converted_back = euler_to_rotation_matrix_zyz(global_orient_euler)
      global_orient = global_orient.reshape(seq_len, -1)
      
      body_pose = body_pose.reshape(seq_len, -1)
      transl = transl.reshape(seq_len, -1)
      smplx = np.concatenate((global_orient, body_pose, transl), axis=1)
      # print(root + "processed/" + subject + "/smplx/"+action_data)
      if(convert_to_angles):
        np.save(root + "processed/" + subject + "/smplx/"+action_data[:-5], smplx)
      else:      
        np.save(root + "processed/" + subject + "/smplx_mat/"+action_data[:-5], smplx)

exit()
