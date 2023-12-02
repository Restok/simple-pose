import json
import numpy as np
import cv2
import os
from data.Fit3DDataset import Fit3DDataset
from sklearn.model_selection import train_test_split

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

def read_data(data_root, dataset_name, subset, subj_name, action_name, camera_name, read_video=False):
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
    frames=None
    if read_video:
        frames = read_video(vid_path)[:seq_len]
    return frames, j3ds, smplx

def getSMPLXParams(prediction):
    global_orient = prediction[:, :9].reshape(-1, 3, 3)
    body_pose = prediction[:, 9:198].reshape(-1, 21, 3, 3)
    transl = prediction[:, 198:]
    return global_orient, body_pose, transl

def getCameraParams(metadata):
    subj_name = metadata['subj_name']
    action_name = metadata['action_name']
    camera_name = metadata['camera_name']
    data_root = 'data/fit3d_train'
    subset = 'train'
    param_path = '%s/%s/%s/camera_parameters/%s/%s.json' % (data_root, subset, subj_name, camera_name, action_name)
    with open(param_path) as f:
        params = json.load(f)
    return params

def load_data(process_frames=False):
    data_root = 'data'
    dataset_name = 'fit3d_train'
    subset = 'train'
    camera_name = '60457274'
    frames_save = 'data/processed/frames'
    subj_names = os.listdir('%s/%s/%s' % (data_root, dataset_name, subset))
    action_names = os.listdir('%s/%s/%s/%s/videos/%s' % (data_root, dataset_name, subset, subj_names[0], camera_name))
    for subj_name in subj_names:
        for action_name in action_names:
            frames, j3ds,smplx = read_data(data_root, dataset_name, subset, subj_name, action_name, camera_name, read_video=process_frames)
            if process_frames:
                for i in range(len(j3ds)):
                    #save frame to frames_save with name subj_name_camera_action_name_i.jpg
                    frame_path = '%s/%s_%s_%s_%d.jpg' % (frames_save, subj_name, camera_name, action_name, i)
                    if process_frames:
                        cv2.imwrite(frame_path, frames[i])
            j3ds = j3ds.reshape(j3ds.shape[0], -1) # shape: (seq_len, 75)
            np.save('data/processed/labels_smplx/%s_%s' % (subj_name, action_name), smplx)
            np.save('data/processed/labels_j3ds/%s_%s' % (subj_name, action_name), j3ds)
        print('finished processing subject %s' % subj_name)

def get_datasets(reload_data=False, shuffle=True):
    if reload_data:
        load_data(False)
    root = 'data/processed/frames'
    frame_paths = os.listdir('data/processed/frames')
    train_frame_paths, test_frame_paths = train_test_split(frame_paths, test_size=0.2, shuffle=shuffle)
    train_dataset = Fit3DDataset(train_frame_paths, root_dir=root)
    test_dataset = Fit3DDataset(test_frame_paths, root_dir=root)
    return train_dataset, test_dataset