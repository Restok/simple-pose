#load model mobilenet_v2_224x224.pth
import torch
from torchvision.models import mobilenet_v2

from preprocess_data import euler_output_to_rot_tensor, get_datasets, get_random_video, getCameraParams, getSMPLXParams, getSMPLXParamsMat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from euler_to_rot import euler_to_rotation_matrix_zyz_tensor
from smplx_util import SMPLXHelper
import imageio
from PIL import Image
from torchvision import transforms
import numpy as np
import io
import random
model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v3_small', pretrained=True)
model.classifier = torch.nn.Linear(576, 69)
device = 'cpu'
model = model.to(device)
model.load_state_dict(torch.load('checkpoints/mobilenetv3mse/mobilenet_v3_224x224_smplx.pth'))

model.eval()
gt_is_euler = False
gt = True
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
gif_frames = []
SMPLX_Models_Path = '../smplx/models'
smplx_helper = SMPLXHelper(SMPLX_Models_Path, device=device)
#
# inference
with torch.no_grad():
    # get a sample from test data
    test_losses = 0
    saved = False
    for i in range(100, 101):
        sample_frame = frames[i]
        sample_label = labels[i]
        sample_label = torch.tensor(sample_label, dtype=torch.float32).unsqueeze(0)
        sample_frame = torch.tensor(sample_frame, dtype=torch.float32).unsqueeze(0)
        sample_frame = sample_frame.to(device)
        sample_label = sample_label.to(device)
        output = model(sample_frame) 
        # if gt:
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
            plt.imsave(f'output/mobilenet_sample_images/{random.randrange(1,10000)}.png', image_as_np_array)
        plt.close()
        #gif_frames.append(rendered_image)
        print(i)
    # save gif
    imageio.mimsave('inference.gif', gif_frames, fps=30)
