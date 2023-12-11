import torch

from preprocess_data import euler_output_to_rot_tensor, getSMPLXParams, getSMPLXParamsMat



def mpjpe(predicted, target):
    """
    Mean Per Joint Position Error
    :param predicted: (batch, 21, 3) predicted joint positions
    :param target: (batch, 21, 3) target joint positions
    """
    #First align the poses
    predicted = predicted - predicted[:, 0:1, :]
    target = target - target[:, 0:1, :]
    diff_squared = (predicted-target)
    error = torch.linalg.norm(diff_squared, dim=-1)
    error = torch.mean(error, dim=-1).mean()
    return error

def mpjpe_from_logits(output, labels, smplx_helper):
    output_rot = euler_output_to_rot_tensor(output)
    global_orient, body_pose, transl = getSMPLXParamsMat(output_rot)
    smplx_params = {'global_orient': global_orient, 'body_pose': body_pose, 'transl': transl}
    global_orient_gt, body_pose_gt, transl_gt = getSMPLXParamsMat(labels)
    smplx_params_gt = {'global_orient': global_orient_gt, 'body_pose': body_pose_gt, 'transl': transl_gt}
    world_posed_data_smplx = smplx_helper.smplx_model(**smplx_params)
    world_posed_data_smplx_gt = smplx_helper.smplx_model(**smplx_params_gt)
    joints = world_posed_data_smplx.joints[:, :22]
    joints_gt = world_posed_data_smplx_gt.joints[:, :22]
    #check if joints contains nan

    err = mpjpe(joints, joints_gt)
    return err