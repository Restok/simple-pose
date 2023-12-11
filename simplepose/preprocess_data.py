import json
import numpy as np
import cv2
import os
from data.Fit3DDataset import Fit3DDataset
from sklearn.model_selection import train_test_split
import torch
from data.Fit3DVideo import Fit3DVideo
import random

from euler_to_rot import euler_to_rotation_matrix_zyz_tensor

def read_video(vid_path):
    frames = []
    cap = cv2.VideoCapture(vid_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        #resize to 256x256
        frame = cv2.resize(frame, (256, 256))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) )
    cap.release()
    frames = np.array(frames)
    return frames

# def read_data(data_root, dataset_name, subset, subj_name, action_name, camera_name, read_video=False):
#     action_name = action_name.split('.')[0]
#     vid_path = '%s/%s/%s/%s/videos/%s/%s.mp4' % (data_root, dataset_name, subset, subj_name, camera_name, action_name)
#     j3d_path = '%s/%s/%s/%s/joints3d_25/%s.json' % (data_root, dataset_name, subset, subj_name, action_name)
#     smplx_path = '%s/%s/%s/%s/smplx/%s.json' % (data_root, dataset_name, subset, subj_name, action_name)
#     with open(j3d_path) as f:
#         j3ds = np.array(json.load(f)['joints3d_25'])
#     global_orient = None
#     body_pose = None
#     transl = None
#     with open(smplx_path) as f:
#         smplx = json.load(f)
#         global_orient = np.array(smplx['global_orient'])
#         # print(global_orient.shape) #(seq_len, 1, 3, 3)
#         body_pose = np.array(smplx['body_pose'])
#         # print(body_pose.shape) #(seq_len, 21, 3, 3)
#         transl = np.array(smplx['transl'])
#         # print(transl.shape) #(seq_len, 3)
#     seq_len = j3ds.shape[-3]
#     global_orient_flat = global_orient.reshape(global_orient.shape[0], -1) # shape: (seq_len, 9)
#     body_pose_flat = body_pose.reshape(body_pose.shape[0], -1)             # shape: (seq_len, 189)
#     smplx = np.concatenate((global_orient_flat, body_pose_flat, transl), axis=1) # shape: (seq_len, 201)
#     frames=None
#     if read_video:
#         frames = read_video(vid_path)[:seq_len]
#     return frames, j3ds, smplx

def getSMPLXParams(prediction):
    global_orient = prediction[:, :3].reshape(-1, 1, 3)
    body_pose = prediction[:, 3:66].reshape(-1, 21, 3)
    transl = prediction[:, 66:69].reshape(-1, 3)
    return global_orient, body_pose, transl

def getSMPLXParamsMat(prediction):
    global_orient = prediction[:, :9].reshape(-1, 3, 3)
    body_pose = prediction[:, 9:198].reshape(-1, 21, 3, 3)
    transl = prediction[:, 198:201].reshape(-1, 3)
    return global_orient, body_pose, transl

def getCameraParams(metadatas):
    data_root = 'data/fit3d_train'
    subset = 'train'
    all_params = []
    batches = len(metadatas['subj_name'])
    for i in range(batches):
        subj_name = metadatas['subj_name'][i]
        action_name = metadatas['action_name'][i]
        camera_name = metadatas['camera_name'][i]
        param_path = '%s/%s/%s/camera_parameters/%s/%s.json' % (data_root, subset, subj_name, camera_name, action_name)
        with open(param_path) as f:
            params = json.load(f)  
            all_params.append(params)

    return all_params

# def load_data(process_frames=False):
#     data_root = 'data'
#     dataset_name = 'fit3d_train'
#     subset = 'train'
#     camera_name = '60457274'
#     frames_save = 'data/processed/frames'
#     subj_names = os.listdir('%s/%s/%s' % (data_root, dataset_name, subset))
#     action_names = os.listdir('%s/%s/%s/%s/videos/%s' % (data_root, dataset_name, subset, subj_names[0], camera_name))
#     for subj_name in subj_names:
#         for action_name in action_names:
#             frames, j3ds,smplx = read_data(data_root, dataset_name, subset, subj_name, action_name, camera_name, read_video=process_frames)
#             if process_frames:
#                 for i in range(len(j3ds)):
#                     #save frame to frames_save with name subj_name_camera_action_name_i.jpg
#                     frame_path = '%s/%s_%s_%s_%d.jpg' % (frames_save, subj_name, camera_name, action_name, i)
#                     if process_frames:
#                         cv2.imwrite(frame_path, frames[i])
#             j3ds = j3ds.reshape(j3ds.shape[0], -1) # shape: (seq_len, 75)
#             np.save('data/processed/labels_smplx/%s_%s' % (subj_name, action_name), smplx)
#             np.save('data/processed/labels_j3ds/%s_%s' % (subj_name, action_name), j3ds)
#         print('finished processing subject %s' % subj_name)

def get_video_slices():
    pass

def get_datasets(shuffle=True, random_state=42):
    root = 'processed'
    all_subjects = os.listdir(root)
    all_subjects = sorted(all_subjects)
    all_cameras = os.listdir('%s/%s/videos' % (root, all_subjects[0]))
    all_cameras = sorted(all_cameras)
    all_actions = os.listdir('%s/%s/videos/%s' % (root, all_subjects[0], all_cameras[0]))
    all_actions = sorted(all_actions)
    frame_paths = []
    for subj_name in all_subjects:
        for camera_name in all_cameras:
            for action_name in all_actions:
                action_name = action_name.split('.')[0]
                frames = os.listdir('%s/%s/videos/%s/%s' % (root, subj_name, camera_name, action_name))
                sorted_frames = sorted(frames, key=lambda x: int(x.split('.')[0]))
                if len(sorted_frames) < 31:
                    continue
                label = np.load('%s/%s/smplx/%s.npy' % (root, subj_name, action_name))
                if label.shape[0] < 31:
                    continue
                frame_paths += ['%s/%s/videos/%s/%s/%s' % (root, subj_name, camera_name, action_name, frame) for frame in sorted_frames[31:min(len(sorted_frames),label.shape[0])]]
    train_frame_paths, test_frame_paths = train_test_split(frame_paths, test_size=0.1, shuffle=shuffle, random_state=random_state)
    train_dataset = Fit3DDataset(train_frame_paths)
    test_dataset = Fit3DDataset(test_frame_paths)
    return train_dataset, test_dataset

def get_video_datasets(shuffle = True, slice_len=5, random_state=42):
    root = 'processed'
    all_subjects = os.listdir(root)
    all_subjects = sorted(all_subjects)
    all_cameras = os.listdir('%s/%s/videos' % (root, all_subjects[0]))
    all_cameras = sorted(all_cameras)
    all_actions = os.listdir('%s/%s/videos/%s' % (root, all_subjects[0], all_cameras[0]))
    all_actions = sorted(all_actions)
    frame_paths = []
    for subj_name in all_subjects:
        for camera_name in all_cameras:
            for action_name in all_actions:
                action_name = action_name.split('.')[0]
                frames = os.listdir('%s/%s/videos/%s/%s' % (root, subj_name, camera_name, action_name))
                sorted_frames = sorted(frames, key=lambda x: int(x.split('.')[0]))
                if len(sorted_frames) < slice_len:
                    continue
                label = np.load('%s/%s/smplx/%s.npy' % (root, subj_name, action_name))
                if label.shape[0] < slice_len:
                    continue
                frame_paths += ['%s/%s/videos/%s/%s/%s' % (root, subj_name, camera_name, action_name, frame) for frame in sorted_frames[(slice_len-1):min(len(sorted_frames),label.shape[0])]]
    
    train_frame_paths, test_frame_paths = train_test_split(frame_paths, test_size=0.1, shuffle=shuffle, random_state=random_state)
    train_dataset = Fit3DVideo(train_frame_paths, slice_len=slice_len)
    test_dataset = Fit3DVideo(test_frame_paths,slice_len=slice_len)
    return train_dataset, test_dataset

def get_random_video(euler_gt=False):
    root = 'data/fit3d_train/train'
    all_subjects = os.listdir(root)
    all_subjects = sorted(all_subjects)
    all_cameras = os.listdir('%s/%s/videos' % (root, all_subjects[0]))
    all_cameras = sorted(all_cameras)
    all_actions = os.listdir('%s/%s/videos/%s' % (root, all_subjects[0], all_cameras[0]))
    random_subject = random.choice(all_subjects)
    random_camera = random.choice(all_cameras)
    random_action = random.choice(all_actions)
    random_video_path = '%s/%s/videos/%s/%s' % (root, random_subject, random_camera, random_action)
    print(random_video_path)
    param_path = '%s/%s/camera_parameters/%s/%s.json' % (root, random_subject, random_camera, random_action.split('.')[0])
    params= None
    with open(param_path) as f:
        params = json.load(f)
    root = 'processed'
    if euler_gt:
        labels = np.load('%s/%s/smplx/%s.npy' % (root, random_subject, random_action.split('.')[0]))
    else:
        labels = np.load('%s/%s/smplx_mat/%s.npy' % (root, random_subject, random_action.split('.')[0]))
    
    return read_video(random_video_path)[:labels.shape[0]], labels, params


def euler_output_to_rot_tensor(output):
    batch_size = output.shape[0]
    global_orient, body_pose, transl = getSMPLXParams(output)
    global_orient = euler_to_rotation_matrix_zyz_tensor(global_orient)
    body_pose = euler_to_rotation_matrix_zyz_tensor(body_pose)
    global_orient = global_orient.reshape(batch_size, -1)
    body_pose = body_pose.reshape(batch_size, -1)
    outputs_rot = torch.cat((global_orient, body_pose, transl), dim=1)
    return outputs_rot