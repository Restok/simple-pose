import numpy as np
import av
import torch
def read_video_pyav(container, indices, height=224, width=224):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frame = frame.reformat(width, height, "rgb24").to_ndarray()
            frames.append(frame)
    stacked = np.stack([x for x in frames])
    return stacked

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

class Fit3DVideo:
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __getitem__(self, idx):
        data_path = self.data_paths[idx] #data/fit3d_train/train/subj/videos/camera/action.mp4
        split_path = data_path.split('/')
        subj_name = split_path[-4]
        camera_name = split_path[-2]
        action_name = split_path[-1]
        metadata = {
            'subj_name': subj_name,
            'action_name': action_name.split('.')[0],
            'camera_name': camera_name
        }
        labels = np.load('data/processed/labels_smplx/%s_%s.npy' % (subj_name, action_name))
        container = av.open(data_path)
        seg_len = container.streams.video[0].frames
        # indices = sample_frame_indices(clip_len, frame_sample_rate, seg_len)
        #Read all frames
        frames = read_video_pyav(container, range(seg_len))
        return frames, torch.tensor(labels, dtype=torch.float32), metadata
    
    def __len__(self):
        return len(self.data_paths)