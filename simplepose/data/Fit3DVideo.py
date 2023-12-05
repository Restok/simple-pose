import numpy as np
import av
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import time
# def read_video_pyav(container, indices, height=224, width=224):
#     '''
#     Decode the video with PyAV decoder.
#     Args:
#         container (`av.container.input.InputContainer`): PyAV container.
#         indices (`List[int]`): List of frame indices to decode.
#     Returns:
#         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
#     '''
#     frames = []
#     container.seek(0)
#     start_index = indices[0]
#     end_index = indices[-1]
#     for i, frame in enumerate(container.decode(video=0)):
#         if i > end_index:
#             break
#         if i >= start_index and i in indices:
#             frame = frame.reformat(width, height, "rgb24").to_ndarray()
#             frames.append(frame)
#     stacked = np.stack([x for x in frames])
#     return stacked

# def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
#     '''
#     Sample a given number of frame indices from the video.
#     Args:
#         clip_len (`int`): Total number of frames to sample.
#         frame_sample_rate (`int`): Sample every n-th frame.
#         seg_len (`int`): Maximum allowed index of sample's last frame.
#     Returns:
#         indices (`List[int]`): List of sampled frame indices
#     '''
#     converted_len = int(clip_len * frame_sample_rate)
#     end_idx = np.random.randint(converted_len, seg_len)
#     start_idx = end_idx - converted_len
#     indices = np.linspace(start_idx, end_idx, num=clip_len)
#     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
#     return indices

# class Fit3DVideo:
#     def __init__(self, data_paths):
#         self.data_paths = data_paths

#     def __getitem__(self, idx):
#         data_path = self.data_paths[idx] #data/fit3d_train/train/subj/videos/camera/action.mp4
#         split_path = data_path.split('/')
#         subj_name = split_path[-4]
#         camera_name = split_path[-2]
#         action_name = split_path[-1]
#         metadata = {
#             'subj_name': subj_name,
#             'action_name': action_name.split('.')[0],
#             'camera_name': camera_name
#         }
#         labels = np.load('data/processed/labels_smplx/%s_%s.npy' % (subj_name, action_name))
#         container = av.open(data_path)
#         seg_len = container.streams.video[0].frames
#         # indices = sample_frame_indices(clip_len, frame_sample_rate, seg_len)
#         #Read all frames
#         frames = read_video_pyav(container, range(seg_len))
#         return frames, torch.tensor(labels, dtype=torch.float32), metadata
    
#     def __len__(self):
#         return len(self.data_paths)

class Fit3DVideo(Dataset):
    def __init__(self, frames, transform=None, slice_len=5):
        self.frame_paths = frames
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),    # Resize the image to 224x224
                transforms.ToTensor(),            # Convert PIL images to tensors
            ])
        self.slice_len = slice_len

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        slice_len = self.slice_len
        start = time.time()
        frame_path = self.frame_paths[idx]
        frames = []
        frame_path_split = frame_path.split('/')
        subj_name = frame_path_split[1]
        camera_name = frame_path_split[3]
        action_name = frame_path_split[4]
        i = int(frame_path_split[5].split('.')[0])
        metadata = {
            'subj_name': subj_name,
            'action_name': action_name.split('.')[0],
            'i': i,
            'camera_name': camera_name
        }
        #get the last 31 frames + current frame

        #frame_path = 'rootdir/%s/videos/%s/%s/%d.jpg' % (subj_name, camera_name, action_name, i)
        
        for j in range(slice_len):
            frame_path_split = frame_path.split('/')
            frame_path_split[-1] = str((i - (slice_len-1) + j)) + ".jpg"
            frame_path = '/'.join(frame_path_split)
            frame = Image.open(frame_path)
            frames.append(frame)
        frames = np.stack([self.transform(x) for x in frames])
        
        

        labels = None
        labels = np.load(f'processed/{subj_name}/smplx_mat/{action_name}.npy')
        label = labels[i]
        # print('Loading data took: %d seconds' % (time.time() - start))
        return frames, torch.tensor(label, dtype=torch.float32), metadata