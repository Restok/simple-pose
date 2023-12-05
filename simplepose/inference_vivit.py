
import torch
from transformers import VivitConfig
from preprocess_data import get_datasets, getCameraParams, getSMPLXParams, get_video_datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from smplx_util import SMPLXHelper
import imageio
from PIL import Image
from model.vivit import VivitPose


device = 'cuda'
configuration = VivitConfig()
vivit_num_frames = 3
configuration.num_labels = 69
configuration.num_frames=vivit_num_frames
train_data, test_data = get_video_datasets(slice_len=vivit_num_frames)
model = VivitPose(configuration).to(device)

model = model.to('cuda')

model.load_state_dict(torch.load('checkpoints/vivit_0.pth'))

# load test data
train_data, test_data = get_video_datasets(slice_len=3)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# inference
with torch.no_grad():
    # get a sample from test data
    gif_frames = []
    for i, v in enumerate(test_data):
        sample_frame, sample_label, metadata = test_data[i]
        sample_frame = torch.tensor(sample_frame, dtype=torch.float32)
        sample_frame = sample_frame.to('cuda')
        sample_label = sample_label.to('cuda')
        sample_frame = sample_frame.unsqueeze(0) # add batch dimension
        sample_label = sample_label.unsqueeze(0) # add batch dimension
        
        output = model(sample_frame)
        # print('loss:', torch.nn.MSELoss()(output, sample_label))
        SMPLX_Models_Path = '../smplx/models'
        smplx_helper = SMPLXHelper(SMPLX_Models_Path)
        camera_params = getCameraParams(metadata)
        # print(camera_params)
        image_id = metadata['i']
        global_orient, body_pose, transl = getSMPLXParams(output.cpu().numpy())
        smplx_params = {'global_orient': global_orient, 'body_pose': body_pose, 'transl': transl}
        global_orient, body_pose, transl = getSMPLXParams(sample_label.cpu().numpy())
        smplx_params_gt = {'global_orient': global_orient, 'body_pose': body_pose, 'transl': transl}

        # joined = {**camera_params, **smplx_params}
        camera_smplx_params = smplx_helper.get_world_smplx_params(smplx_params)
        camera_posed_data_smplx = smplx_helper.smplx_model(**camera_smplx_params)
        vertices = camera_posed_data_smplx.vertices.cpu().detach().numpy().squeeze()
        joints = camera_posed_data_smplx.joints.cpu().detach()[:22].numpy()
        joints_gt = camera_posed_data_smplx.joints.cpu().detach()[22:].numpy()
        print(joints.shape)
        print(joints_gt.shape)
        vertices_gt = camera_posed_data_smplx.vertices.cpu().detach().numpy().squeeze()
        rendered_image = smplx_helper.render(vertices, sample_frame.cpu(), camera_params, vertices_in_world=True) #tensor
        rendered_image = rendered_image.cpu().detach().numpy().squeeze()
        rendered_image = (rendered_image*255).astype('uint8')
        gif_frames.append(rendered_image)
        plt.imshow(rendered_image)
        # if i > 5:
        break
        plt.show()
    # save gif
    imageio.mimsave('inference.gif', gif_frames, fps=5)