import numpy as np
import json
from preprocess_data import *
import os
import multiprocessing as mp

def rot_mat_to_euler(rot_mats):
    # a = np.arcsin(R[0,2])
    # b = np.arctan2(R[0,1]/-np.cos(a), R[0,0]/np.cos(a))
    # c = np.arctan2(R[1,2]/-np.cos(a), R[2,2]/np.cos(a))
    # if abs(R[0,0])<10^-16 or abs(R[2,2])<10^-16:
    #     print("Matrix has infinite Solutions")
    # return c,a,b
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]
  rot_mats = torch.tensor(rot_mats)
  sy = torch.sqrt(rot_mats[0, 0] * rot_mats[0, 0] +
                  rot_mats[1, 0] * rot_mats[1, 0])
  res= torch.atan2(-rot_mats[2, 0], sy)
  print(res)   
  return torch.atan2(-rot_mats[2, 0], sy)


subjects = ["s03", "s04", "s05", "s07", "s08", "s09", "s11"]
# cameras = ["50591643/", "58860488/", "60457274/", "65906101/"]

root = "C:/Users/DAQ/Desktop/simple-pose/simplepose/"
base = root+"data/fit3d_train/train/"

if not os.path.exists("processed"):
    os.makedirs("processed")


for subject in subjects:
    if not os.path.exists(root+"processed/" + subject):
        os.makedirs(root+"processed/" + subject)
    if not os.path.exists(root+"processed/" + subject + "/smplx"):
        os.makedirs(root+"processed/" + subject + "/smplx")
    actions = os.listdir(base + subject + "/smplx")
    for action_data in actions:
      f = open(base + subject + "/smplx/" + action_data)
      smplx = json.load(f)
      global_orient = np.array(smplx['global_orient'])
      body_pose = np.array(smplx['body_pose'])
      transl = np.array(smplx['transl'])

      global_euler = []
      for fr in range(global_orient.shape[0]):
        global_euler.append(rot_mat_to_euler(global_orient[fr,0]))
      bodies = []
      for fr in range(body_pose.shape[0]):
        body_euler = []
        for j in range(21):
          body_euler+= rot_mat_to_euler(body_pose[fr, 1])
        
        bodies.append(body_euler)
      smplx = np.concatenate((global_euler, bodies, transl), axis=1)
      # print(root + "processed/" + subject + "/smplx/"+action_data)
      np.save(root + "processed/" + subject + "/smplx/"+action_data[:-5], smplx)
            
            


exit()
