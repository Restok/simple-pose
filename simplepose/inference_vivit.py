
import numpy as np
import torch
from transformers import VivitConfig
from preprocess_data import euler_output_to_rot_tensor, get_datasets, get_random_video, getCameraParams, getSMPLXParams, get_video_datasets, getSMPLXParamsMat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from smplx_util import SMPLXHelper
import imageio
from PIL import Image
from model.vivit import VivitPose
from torch.utils.data import DataLoader
from euler_to_rot import euler_to_rotation_matrix_zyz, euler_to_rotation_matrix_zyz_tensor
from torchvision import transforms
import io
import random
device = 'cpu'
configuration = VivitConfig()
configuration.hidden_size
vivit_num_frames = 3
configuration.num_labels = 69
configuration.num_attention_heads = 6
configuration.num_hidden_layers = 6
configuration.num_frames=vivit_num_frames
model = VivitPose(configuration).to(device)

model = model.to(device)

model.load_state_dict(torch.load('checkpoints/vivit_mobilenet_MSE.pth'))

SMPLX_Models_Path = '../smplx/models'
smplx_helper = SMPLXHelper(SMPLX_Models_Path, device=device)
gt_is_euler = True

frames, labels, camera_params = get_random_video(euler_gt=gt_is_euler)
# frames is numpy array
# inference
model.eval()
t = transforms.Compose([
    transforms.Resize((224, 224)),    # Resize the image to 224x224
    transforms.ToTensor(),            # Convert PIL images to tensors
])

gt = False
frames = np.stack([t(Image.fromarray(x)) for x in frames])
apply_transl = False
with torch.no_grad():
    # get a sample from test data
    gif_frames = []
    saved = False
    for i in range(vivit_num_frames+80, vivit_num_frames+81):
        print(i)
        frame_history = frames[i-(vivit_num_frames-1):i+1]
        sample_label = labels[i]
        sample_label = torch.tensor(sample_label, dtype=torch.float32).unsqueeze(0)
        sample_frame = torch.tensor(frame_history, dtype=torch.float32).unsqueeze(0)
        sample_frame = sample_frame.to(device)
        sample_label = sample_label.to(device)
        output = model(sample_frame)
        # print('loss:', torch.nn.MSELoss()(output, sample_label))
        # print(camera_params)
        sample_frame = sample_frame.squeeze()[-1]
        global_orient_gt, body_pose_gt, transl_gt = None, None, None
        if gt_is_euler:
            global_orient_gt, body_pose_gt, transl_gt = getSMPLXParams(sample_label)
            global_orient_gt = euler_to_rotation_matrix_zyz_tensor(global_orient_gt)
            body_pose_gt = euler_to_rotation_matrix_zyz_tensor(body_pose_gt)
        else:
            global_orient_gt, body_pose_gt, transl_gt = getSMPLXParamsMat(sample_label)
        
        global_orient, body_pose, transl = getSMPLXParams(output)
        global_orient = euler_to_rotation_matrix_zyz_tensor(global_orient)
        body_pose = euler_to_rotation_matrix_zyz_tensor(body_pose)
        outputs_rot = euler_output_to_rot_tensor(output)
        smplx_params = {'global_orient': global_orient, 'body_pose': body_pose, 'transl': transl_gt}
        camera_smplx_params = smplx_helper.get_world_smplx_params(smplx_params)
        camera_posed_data_smplx = smplx_helper.smplx_model(**camera_smplx_params)
        vertices = camera_posed_data_smplx.vertices.cpu().detach().numpy().squeeze()
        # if gt:
        smplx_params_gt = {'global_orient': global_orient_gt, 'body_pose': body_pose_gt, 'transl': transl_gt}
        camera_smplx_params_gt = smplx_helper.get_world_smplx_params(smplx_params_gt)
        camera_posed_data_smplx_gt = smplx_helper.smplx_model(**camera_smplx_params_gt)
        vertices_gt = camera_posed_data_smplx_gt.vertices.cpu().detach().numpy().squeeze()
        joints_gt = camera_posed_data_smplx_gt.joints.cpu().squeeze()[:22].detach().numpy().squeeze()
        rendered_image_gt = smplx_helper.render(vertices_gt, sample_frame.cpu(), camera_params, vertices_in_world=True, color=[0.0, 1.0, 0.0, 1])
        
        rendered_image = smplx_helper.render(vertices, sample_frame.cpu(), camera_params, vertices_in_world=True) #tensor
        rendered_image = rendered_image.cpu().detach().numpy().squeeze()
        rendered_image = (rendered_image*255).astype('uint8')
        #plot rendered and gt side by side
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        #title
        plt.title('Predicted')
        ax1.imshow(rendered_image)
        ax2 = fig.add_subplot(122)
        plt.title('Ground Truth')
        ax2.imshow(rendered_image_gt)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_as_np_array = np.array(Image.open(buf))
        #close figure
        buf.close()
        #save plot
        gif_frames.append(image_as_np_array)
        if not saved:
            #save first frame
            saved = True
            plt.imsave(f'output/vivit_sample_images/{random.randrange(1,10000)}.png', image_as_np_array)
        plt.close()
        #gif_frames.append(rendered_image)
        print(i)
    # save gif
    imageio.mimsave('inference.gif', gif_frames, fps=30)