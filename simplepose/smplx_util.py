from smplx import build_layer
from smplx.lbs import batch_rodrigues
import torch
from torchvision import transforms
import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import trimesh
# import tensorflow.compat.v2 as tf
import numpy as np
import copy

smplx_cfg = {'ext': 'npz',
             'extra_joint_path': '',
             'folder': 'transfer_data/body_models',
             'gender': 'neutral',
             'joint_regressor_path': '',
             'model_type': 'smplx',
             'num_expression_coeffs': 10,
             'smplx': {'betas': {'create': True, 'num': 10, 'requires_grad': True},
                       'body_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'expression': {'create': True, 'num': 10, 'requires_grad': True},
                       'global_rot': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'jaw_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'left_hand_pose': {'create': True,
                                          'pca': {'flat_hand_mean': False, 'num_comps': 12},
                                          'requires_grad': True,
                                          'type': 'aa'},
                       'leye_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'reye_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'right_hand_pose': {'create': True,
                                           'pca': {'flat_hand_mean': False,
                                                   'num_comps': 12},
                                           'requires_grad': True,
                                           'type': 'aa'},
                       'translation': {'create': True, 'requires_grad': True}},
             'use_compressed': False,
             'use_face_contour': True}

class SMPLXHelper:
    def __init__(self, Models_Path=None, load_renderer=True, device='cuda'):
        self.Models_Path = Models_Path
        self.cfg = smplx_cfg
        self.smplx_model = build_layer(self.Models_Path, **self.cfg)
        self.smplx_model.eval()
        self.smplx_model.requires_grad_(False)
        self.smplx_model = self.smplx_model.to(device)
        self.image_shape = (900, 900)
        if load_renderer:
            self.mesh_rasterizer = self.get_smplx_rasterizer()
        else:
            self.mesh_rasterizer = None
           
    def get_world_smplx_params(self, smplx_params):
        world_smplx_params = {key: torch.from_numpy(np.array(smplx_params[key]).astype(np.float32)) for key in smplx_params}
        return world_smplx_params
    
    def get_camera_smplx_params(self, smplx_params, cam_params):
        pelvis = self.smplx_model(betas=torch.from_numpy(np.array(smplx_params['betas']).astype(np.float32))).joints[:, 0, :].numpy()
        camera_smplx_params = copy.deepcopy(smplx_params)
        camera_smplx_params['global_orient'] = np.matmul(np.array(smplx_params['global_orient']).transpose(0, 1, 3, 2), np.transpose(cam_params['extrinsics']['R'])).transpose(0, 1, 3, 2)
        camera_smplx_params['transl'] = np.matmul(smplx_params['transl'] + pelvis - cam_params['extrinsics']['T'], np.transpose(cam_params['extrinsics']['R'])) - pelvis
        camera_smplx_params = {key: torch.from_numpy(np.array(camera_smplx_params[key]).astype(np.float32)) for key in camera_smplx_params}
        return camera_smplx_params
        
    def get_smplx_rasterizer(self):
        import pyrender
        return pyrender.OffscreenRenderer(viewport_width=self.image_shape[0], 
                                          viewport_height=self.image_shape[1], 
                                          point_size=1.0)
    
    def get_template_params(self, batch_size=1):
        smplx_params = {}
        smplx_params_all = self.smplx_model()
        for key1 in ['transl', 'global_orient', 'body_pose', 'betas', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression', 'leye_pose', 'reye_pose']:
            key2 = key1 if key1 in smplx_params_all else 'jaw_pose'
            smplx_params[key1] = np.repeat(smplx_params_all[key2].cpu().detach().numpy(), batch_size, axis=0)
        smplx_params['transl'][:, 2] = 3
        smplx_params['global_orient'][:, :, 1, 1] = -1
        smplx_params['global_orient'][:, :, 2, 2] = -1
        return smplx_params
    
    def get_template(self):
        smplx_posed_data = self.smplx_model()
        smplx_template = {'vertices': smplx_posed_data.vertices[0].cpu().detach().numpy(), 'triangles': self.smplx_model.faces}
        return smplx_template



    def render(self, vertices, frame, cam_params, vertices_in_world=True, gt_vertices=None, joints=None, color = [0.3, 0.3, 0.3, 1.0]):
        import pyrender
        blending_weight=1.0
        if vertices_in_world:
            vertices = np.matmul(vertices - cam_params['extrinsics']['T'], np.transpose(cam_params['extrinsics']['R'])) # transform vertices to camera space
            if gt_vertices is not None:
                gt_vertices = np.matmul(gt_vertices - cam_params['extrinsics']['T'], np.transpose(cam_params['extrinsics']['R']))
        vertices_to_render = vertices
        intrinsics = cam_params['intrinsics_wo_distortion']['f'] + cam_params['intrinsics_wo_distortion']['c'] 
        background_image = frame

        vertex_colors = np.ones([vertices_to_render.shape[0], 4]) * color # gray
        tri_mesh = trimesh.Trimesh(vertices_to_render, self.smplx_model.faces, 
                                   vertex_colors=vertex_colors) # create mesh

        mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True) # create mesh object
        scene = pyrender.Scene(ambient_light=(0.0, 0.0, 0.0)) # create scene
        scene.add(mesh, 'mesh')
        #joints are 
        """"""
        if joints is not None:
            if vertices_in_world:
                joints = np.matmul(joints - cam_params['extrinsics']['T'], np.transpose(cam_params['extrinsics']['R']))
            sm = trimesh.creation.uv_sphere(radius=0.1)
            #red
            sm.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        if gt_vertices is not None:
            vertex_colors = np.ones([gt_vertices.shape[0], 4]) * [0.0, 1.0, 0.0, 1] # green
            tri_mesh = trimesh.Trimesh(gt_vertices, self.smplx_model.faces,
                                        vertex_colors=vertex_colors)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
            scene.add(mesh, 'gt_mesh')


        camera_pose = np.eye(4) # create camera pose from extrinsics
        rot = trimesh.transformations.euler_matrix(0, np.pi, np.pi, 'rxyz')
        camera_pose[:3, :3] = rot[:3, :3]

        camera = pyrender.IntrinsicsCamera(
          fx=intrinsics[0],
          fy=intrinsics[1],
          cx=intrinsics[2],
          cy=intrinsics[3])

        scene.add(camera, pose=camera_pose)

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10)

        scene.add(light, pose=camera_pose)
        color, rend_depth = self.mesh_rasterizer.render(scene, flags=pyrender.RenderFlags.RGBA)
        img = color.astype(np.float32) / 255.0
        blended_image = img[:, :, :3]
        if background_image is not None:
            background_image = background_image.squeeze()
            #change to be (H, W, 3)
            background_image = background_image.permute(1, 2, 0)
            
            # Permute the dimensions to (C, H, W)
            img = torch.from_numpy(img).permute(2, 0, 1)
            resize_transform = transforms.Resize((background_image.shape[0], background_image.shape[1]), antialias=True)
            img = resize_transform(img).permute(1, 2, 0)
            
            # Blend the rendering result with the background image.
            foreground = (rend_depth > 0)[:, :, None] * blending_weight
            foreground = resize_transform(torch.from_numpy(foreground).permute(2, 0, 1)).permute(1, 2, 0)
            blended_image = (foreground * img[:, :, :3]
                             + (1. - foreground) * background_image)
        # pyrender.Viewer(scene, use_raymond_lighting=True)
        return blended_image