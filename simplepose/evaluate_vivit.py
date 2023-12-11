#load model mobilenet_v2_224x224.pth
import torch
from torchvision.models import mobilenet_v2

from preprocess_data import get_datasets, get_video_datasets, getCameraParams, getSMPLXParams, getSMPLXParamsMat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from metrics.losses import mpjpe
from euler_to_rot import euler_to_rotation_matrix_zyz_tensor
from model.vivit import VivitPose
from smplx_util import SMPLXHelper
import imageio
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from transformers import VivitConfig

device = 'cuda'
configuration = VivitConfig()
vivit_num_frames = 3
configuration.num_labels = 69
configuration.num_frames=vivit_num_frames
configuration.num_attention_heads = 6
configuration.num_hidden_layers = 6
train_data, test_data = get_video_datasets(slice_len=vivit_num_frames)
model = VivitPose(configuration).to(device)
model.load_state_dict(torch.load('checkpoints/vivit_mobilenet_MSE.pth'))

# load test data
train_data, test_data = get_video_datasets(slice_len=3)
test_loader = DataLoader(test_data, batch_size=20, shuffle=False)

SMPLX_Models_Path = '../smplx/models'
smplx_helper = SMPLXHelper(SMPLX_Models_Path)
# inference
with torch.no_grad():
    # get a sample from test data
    total_err = 0
    for i, v in enumerate(test_loader):
        frames, labels, metadatas = v
        frames = frames.to('cuda')
        labels = labels.to('cuda')
        output = model(frames)
        global_orient, body_pose, transl = getSMPLXParams(output)
        global_orient = euler_to_rotation_matrix_zyz_tensor(global_orient)
        body_pose = euler_to_rotation_matrix_zyz_tensor(body_pose)
        smplx_params = {'global_orient': global_orient, 'body_pose': body_pose, 'transl': transl}
        global_orient_gt, body_pose_gt, transl_gt = getSMPLXParamsMat(labels)
        smplx_params_gt = {'global_orient': global_orient_gt, 'body_pose': body_pose_gt, 'transl': transl_gt}
        world_posed_data_smplx = smplx_helper.smplx_model(**smplx_params)
        world_posed_data_smplx_gt = smplx_helper.smplx_model(**smplx_params_gt)
        joints = world_posed_data_smplx.joints[:, :22]
        joints_gt = world_posed_data_smplx_gt.joints[:, :22]
        err = mpjpe(joints, joints_gt)
        total_err += err.item()
        if i % 10 == 0:
            print('Iteration: %d/%d' % (i, len(test_loader)))
        if i >500:
            break
    print('MPJPE:', total_err / 500)
