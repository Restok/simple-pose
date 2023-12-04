#load model mobilenet_v2_224x224.pth
import torch
from torchvision.models import mobilenet_v2

from preprocess_data import get_datasets, getCameraParams, getSMPLXParams
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from smplx_util import SMPLXHelper
import imageio
model = mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(1280, 201)
model = model.to('cuda')

model.load_state_dict(torch.load('mobilenet_v2_224x224_smplx.pth'))
model.eval()

# load test data
train_data, test_data = get_datasets(reload_data=False, shuffle=False)



# inference
with torch.no_grad():
    # get a sample from test data
    gif_frames = []
    for i, v in enumerate(test_data):
        sample_frame, sample_label, metadata = test_data[i]
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
        
        vertices_gt = camera_posed_data_smplx.vertices.cpu().detach().numpy().squeeze()
        
        rendered_image = smplx_helper.render(vertices, sample_frame.cpu(), camera_params, vertices_in_world=True) #numpy

        
        gif_frames.append(rendered_image)
        if i > 5:
            break
        # plt.show()
    # save gif
    imageio.mimsave('inference.gif', gif_frames, fps=5)