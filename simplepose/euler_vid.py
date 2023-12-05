import numpy as np
import json
from preprocess_data import *
import os
import multiprocessing as mp

def rot_mat_to_euler(R):
    a = np.arcsin(R(1,3))
    b = np.arctan2(R(1,2)/-np.cos(a), R(1,1)/np.cos(a))
    c = np.arrctan2(R(2,3)/-np.cos(a), R(3,3)/np.cos(a))
    if abs(R(1,1))<10^-16 or abs(R(3,3))<10^-16:
        print("Matrix has infinite Solutions")
    return c,a,b

def rpw(root, base, subject, camera, action_vid):
    if not os.path.exists(root + "processed/" + subject + "/videos/" + camera + "/" + action_vid[:-4]):
        os.makedirs(root + "processed/" + subject + "/videos/" + camera + "/" + action_vid[:-4])
    v = read_video(base + subject + "/videos/" + camera + "/" + action_vid)
            # print(v.shape)
            # print(v[0].shape)
    for frame in range(v.shape[0]):
        cv2.imwrite(root + "processed/" + subject + "/videos/" + camera + "/" + action_vid[:-4] + f"/{frame}.jpg", v[frame])


subjects = ["s03", "s04", "s05", "s07", "s08", "s09", "s11"]
# cameras = ["50591643/", "58860488/", "60457274/", "65906101/"]

root = "C:/Users/DAQ/Desktop/simple-pose/simplepose/"
base = root+"data/fit3d_train/train/"

if not os.path.exists("processed"):
    os.makedirs("processed")


for subject in subjects:
    if not os.path.exists(root+"processed/" + subject):
        os.makedirs(root+"processed/" + subject)
    cameras = os.listdir(base + subject + "/videos")
    for camera in cameras:
        if not os.path.exists(root+"processed/" + subject + "/videos/" + camera):
            os.makedirs(root+"processed/" + subject + "/videos/" + camera)
        actions = os.listdir(base + subject + "/videos/" + camera)
        # action_parse_processes = []
        for action_vid in actions:
            rpw(root, base, subject, camera, action_vid)
        #     proc = mp.Process(target=rpw, args=(root, base, subject, camera, action_vid))
        #     action_parse_processes.append(proc)
        #     proc.start()
        # for proc in action_parse_processes:
        #     proc.join()
            
            


exit()

action_name = action_name.split('.')[0]
vid_path = '%s/%s/%s/%s/videos/%s/%s.mp4' % (data_root, dataset_name, subset, subj_name, camera_name, action_name)
j3d_path = '%s/%s/%s/%s/joints3d_25/%s.json' % (data_root, dataset_name, subset, subj_name, action_name)
smplx_path = '%s/%s/%s/%s/smplx/%s.json' % (data_root, dataset_name, subset, subj_name, action_name)
with open(j3d_path) as f:
    j3ds = np.array(json.load(f)['joints3d_25'])
global_orient = None
body_pose = None
transl = None
with open(smplx_path) as f:
    smplx = json.load(f)
    global_orient = np.array(smplx['global_orient'])
    # print(global_orient.shape) #(seq_len, 1, 3, 3)
    body_pose = np.array(smplx['body_pose'])
    # print(body_pose.shape) #(seq_len, 21, 3, 3)
    transl = np.array(smplx['transl'])
    # print(transl.shape) #(seq_len, 3)
seq_len = j3ds.shape[-3]
global_orient_flat = global_orient.reshape(global_orient.shape[0], -1) # shape: (seq_len, 9)
body_pose_flat = body_pose.reshape(body_pose.shape[0], -1)             # shape: (seq_len, 189)
smplx = np.concatenate((global_orient_flat, body_pose_flat, transl), axis=1) # shape: (seq_len, 201)
frames = read_video(vid_path)[:seq_len]
print(frames.shape)
print(smplx.shape)
print(global_orient.shape)
