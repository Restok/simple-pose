
import torch
from transformers import VivitConfig
from preprocess_data import get_datasets, getCameraParams, getSMPLXParams, get_video_datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from smplx_util import SMPLXHelper
import imageio
from PIL import Image
from model.vivit import VivitPose
from torch.utils.data import DataLoader
from euler_to_rot import euler_to_rotation_matrix_zyz, euler_to_rotation_matrix_zyz_tensor
device = 'cpu'
configuration = VivitConfig()
vivit_num_frames = 3
configuration.num_labels = 69
configuration.num_frames=vivit_num_frames
train_data, test_data = get_video_datasets(slice_len=vivit_num_frames)
model = VivitPose(configuration).to(device)

model = model.to(device)

model.load_state_dict(torch.load('checkpoints/vivit_0.pth'))

# load test data
train_data, test_data = get_video_datasets(slice_len=3)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

SMPLX_Models_Path = '../smplx/models'
smplx_helper = SMPLXHelper(SMPLX_Models_Path, device=device)
ax, fig = plt.subplots(1, 1)
# inference
with torch.no_grad():
    # get a sample from test data
    gif_frames = []
    for i, v in enumerate(test_loader):
        sample_frame, sample_label, metadata = v
        sample_frame = torch.tensor(sample_frame, dtype=torch.float32)
        sample_frame = sample_frame.to(device)
        sample_label = sample_label.to(device)
        sample_frame = sample_frame # add batch dimension
        sample_label = sample_label # add batch dimension
        output = model(sample_frame)
        # print('loss:', torch.nn.MSELoss()(output, sample_label))

        camera_params = getCameraParams(metadata)
        # print(camera_params)
        image_id = metadata['i']
        # global_orient, body_pose, transl = getSMPLXParams(output.cpu().numpy())
        global_orient, body_pose, transl = getSMPLXParams(sample_label.cpu())
        print(global_orient.shape)
        print(body_pose.shape)
        global_orient = euler_to_rotation_matrix_zyz_tensor(global_orient)
        print(global_orient.shape)
        print(global_orient)
        body_pose = euler_to_rotation_matrix_zyz_tensor(body_pose)
        # smplx_params = {'global_orient': global_orient, 'body_pose': body_pose, 'transl': transl}
        smplx_params_gt = {'global_orient': global_orient, 'body_pose': body_pose, 'transl': transl}
        
        # joined = {**camera_params, **smplx_params}
        camera_smplx_params = smplx_helper.get_world_smplx_params(smplx_params_gt)
        camera_posed_data_smplx = smplx_helper.smplx_model(**camera_smplx_params)
        vertices = camera_posed_data_smplx.vertices.cpu().detach().numpy().squeeze()
        # joints = camera_posed_data_smplx.joints.cpu().detach()[:22].numpy()
        # print(joints.shape)
        # print(sample_frame[-].cpu().squeeze().shape)
        print(sample_frame.squeeze()[-1].cpu().shape)
        rendered_image = smplx_helper.render(vertices,sample_frame.squeeze()[-1].cpu(), camera_params[0], vertices_in_world=True) #tensor
        # rendered_image = rendered_image.cpu().detach().numpy().squeeze()
        # rendered_image = (rendered_image*255).astype('uint8')
        gif_frames.append(rendered_image)
        plt.imshow(rendered_image)
        # if i > 5:
        plt.show()
        break
    # save gif
    imageio.mimsave('inference.gif', gif_frames, fps=5)