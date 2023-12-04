from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
class Fit3DDataset(Dataset):
    def __init__(self, frames, use_smplx = True, use_j3ds = False, transform=None, root_dir=None):
        self.frame_paths = frames
        self.use_smplx = use_smplx
        self.use_j3ds = use_j3ds
        self.root_dir = root_dir
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),    # Resize the image to 224x224
                transforms.ToTensor(),            # Convert PIL images to tensors
            ])

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        frame = Image.open(self.root_dir + '/' + frame_path)
        #frame_path = '%s/%s_%s_%s_%d.jpg' % (frames_save, subj_name, camera_name, action_name, i)
        split_frame_path = frame_path.split('_')
        subj_name = split_frame_path[0]
        i = int(split_frame_path[-1].split('.')[0])
        action_name = '_'.join(split_frame_path[2:-1])
        camera_name = split_frame_path[1]
        metadata = {
            'subj_name': subj_name,
            'action_name': action_name.split('.')[0],
            'i': i,
            'camera_name': camera_name
        }
        labels = None
        if self.use_smplx:
            labels = np.load('data/processed/labels_smplx/%s_%s.npy' % (subj_name, action_name))
        elif self.use_j3ds:
            labels = np.load('data/processed/labels_j3ds/%s_%s.npy' % (subj_name, action_name))
        label = labels[i]
        return self.transform(frame), torch.tensor(label, dtype=torch.float32), metadata