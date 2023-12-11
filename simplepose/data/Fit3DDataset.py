from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
class Fit3DDataset(Dataset):
    def __init__(self, frames, transform=None):
        self.frame_paths = frames
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
        try:
            frame_path = self.frame_paths[idx]
            frame = Image.open(frame_path)
            #frame_path = 'rootdir/%s/videos/%s/%s/%d.jpg' % (subj_name, camera_name, action_name, i)
            
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

            labels = None
            
            labels = np.load(f'processed/{subj_name}/smplx_mat/{action_name}.npy')
            label = labels[i]
            return self.transform(frame), torch.tensor(label, dtype=torch.float32), metadata
        except Exception as e:
            print(e)
            print(frame_path)
            exit()