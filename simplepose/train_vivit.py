from transformers import VivitConfig

from model.vivit import VivitPose
from preprocess_data import get_video_datasets
from torch.utils.data import DataLoader
import torch
configuration = VivitConfig()

model = VivitPose(configuration)
train_data, test_data = get_video_datasets()
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

for i, v in enumerate(train_loader):
    #sliding window of 32 frames or less, if we are at the beginning
    for j in range(len(v)):
        start_range = max(0, j-32)
        end_range = j
        frames = v[start_range:end_range]
        #convert frames to tensor
        frames = torch.tensor(frames)
        print(frames.size)
    break