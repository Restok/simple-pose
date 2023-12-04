#load model mobilenet_v2_224x224.pth
import torch
from torchvision.models import mobilenet_v2

from preprocess_data import get_datasets, getCameraParams, getSMPLXParams
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from metrics.losses import mpjpe
from smplx_util import SMPLXHelper
import imageio
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

model = mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(1280, 201)
model = model.to('cuda')

model.load_state_dict(torch.load('mobilenet_v2_224x224_smplx.pth'))
model.eval()

# load test data
train_data, test_data = get_datasets(reload_data=False, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
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
        smplx_params = {'global_orient': global_orient, 'body_pose': body_pose, 'transl': transl}
        global_orient_gt, body_pose_gt, transl_gt = getSMPLXParams(labels)
        smplx_params_gt = {'global_orient': global_orient_gt, 'body_pose': body_pose_gt, 'transl': transl_gt}
        world_posed_data_smplx = smplx_helper.smplx_model(**smplx_params)
        world_posed_data_smplx_gt = smplx_helper.smplx_model(**smplx_params_gt)
        joints = world_posed_data_smplx.joints[:, :22]
        joints_gt = world_posed_data_smplx_gt.joints[:, :22]
        err = mpjpe(joints, joints_gt)
        total_err += err.item()
        if i % 10 == 0:
            print('Iteration: %d/%d' % (i, len(test_loader)))
    print('MPJPE:', total_err / len(test_loader))
