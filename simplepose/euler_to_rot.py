import numpy as np
import torch
# def euler_to_rotation_matrix_xyz(angles):
#     """
#     Convert multiple batches of Euler angles (roll, pitch, yaw) to rotation matrices using XYZ order.
#     This function is vectorized for better performance and supports multiple batches.

#     :param angles: An array of shape (batch, num_angles, 3) where each entry contains (roll, pitch, yaw) in radians
#     :return: An array of shape (batch, num_angles, 3, 3) containing rotation matrices
#     """
#     batch_size, num_angles, _ = angles.shape

#     # Reshape angles for vectorized operations
#     angles_reshaped = angles.reshape(-1, 3)
#     roll, pitch, yaw = angles_reshaped[:, 0], angles_reshaped[:, 1], angles_reshaped[:, 2]

#     # Compute cosine and sine of the angles
#     cr = np.cos(roll)
#     sr = np.sin(roll)
#     cp = np.cos(pitch)
#     sp = np.sin(pitch)
#     cy = np.cos(yaw)
#     sy = np.sin(yaw)

#     # Compute the rotation matrices for XYZ order
#     rotation_matrices = np.array([
#         [cp * cy, cp * sy, -sp],
#         [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
#         [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp]
#     ]).transpose(2, 0, 1)

#     # Reshape back to the desired output shape
#     rotation_matrices_reshaped = rotation_matrices.reshape(batch_size, num_angles, 3, 3)

#     return rotation_matrices_reshaped

def euler_to_rotation_matrix_zyz(angles):
    """
    Convert multiple batches of Euler angles (roll, pitch, yaw) to rotation matrices using ZYX order.
    This function is vectorized for better performance and supports multiple batches.

    :param angles: An array of shape (batch, num_angles, 3) where each entry contains (roll, pitch, yaw) in radians
    :return: An array of shape (batch, num_angles, 3, 3) containing rotation matrices
    """
    batch_size, num_angles, _ = angles.shape

    # Reshape angles for vectorized operations
    angles_reshaped = angles.reshape(-1, 3)
    roll, pitch, yaw = angles_reshaped[:, 0], angles_reshaped[:, 1], angles_reshaped[:, 2]

    # Compute cosine and sine of the angles
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # Compute the rotation matrices for ZYX order
    rotation_matrices = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ]).transpose(2, 0, 1)

    # Reshape back to the desired output shape
    rotation_matrices_reshaped = rotation_matrices.reshape(batch_size, num_angles, 3, 3)

    return rotation_matrices_reshaped

def rotation_matrix_to_euler_zyz(rot_matrices):
    """
    Convert multiple batches of rotation matrices to Euler angles in the XYZ order, vectorized implementation.

    :param rot_matrices: An array of shape (batch, n, 3, 3) containing rotation matrices
    :return: An array of shape (batch, n, 3) where each row contains (roll, pitch, yaw) in radians
    """
    batch_size, n, _, _ = rot_matrices.shape

    # Reshape for vectorized operations
    rot_matrices_reshaped = rot_matrices.reshape(-1, 3, 3)

    # Extract the necessary matrix elements for computation
    m00, m01, m02 = rot_matrices_reshaped[:, 0, 0], rot_matrices_reshaped[:, 0, 1], rot_matrices_reshaped[:, 0, 2]
    m10, m11, m12 = rot_matrices_reshaped[:, 1, 0], rot_matrices_reshaped[:, 1, 1], rot_matrices_reshaped[:, 1, 2]
    m20, m21, m22 = rot_matrices_reshaped[:, 2, 0], rot_matrices_reshaped[:, 2, 1], rot_matrices_reshaped[:, 2, 2]

    # Calculate the Euler angles
    pitch = np.arcsin(-m20)
    roll = np.where(np.abs(m20) < 1, np.arctan2(m21, m22), 0)
    yaw = np.where(np.abs(m20) < 1, np.arctan2(m10, m00), 0)

    # Reshape back to the desired output shape
    euler_angles = np.vstack((roll, pitch, yaw)).T
    euler_angles_reshaped = euler_angles.reshape(batch_size, n, 3)

    return euler_angles_reshaped

# def euler_to_rotation_matrix_xyz_tensor(angles):
#     """
#     Convert multiple batches of Euler angles (roll, pitch, yaw) to rotation matrices using XYZ order.
#     This function uses PyTorch tensors for better performance and supports multiple batches.

#     :param angles: A tensor of shape (batch, num_angles, 3) where each entry contains (roll, pitch, yaw) in radians
#     :return: A tensor of shape (batch, num_angles, 3, 3) containing rotation matrices
#     """
#     batch_size, num_angles, _ = angles.shape

#     # Extract roll, pitch, yaw components
#     roll, pitch, yaw = angles[..., 0], angles[..., 1], angles[..., 2]

#     # Compute cosine and sine of the angles
#     cr, sr = torch.cos(roll), torch.sin(roll)
#     cp, sp = torch.cos(pitch), torch.sin(pitch)
#     cy, sy = torch.cos(yaw), torch.sin(yaw)

#     # Compute the rotation matrices for XYZ order
#     r00, r01, r02 = cp * cy, cp * sy, -sp
#     r10, r11, r12 = sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp
#     r20, r21, r22 = cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp

#     # Stack the components into rotation matrices
#     rotation_matrices = torch.stack([
#         torch.stack([r00, r01, r02], dim=-1),
#         torch.stack([r10, r11, r12], dim=-1),
#         torch.stack([r20, r21, r22], dim=-1)
#     ], dim=-2)

#     return rotation_matrices

def euler_to_rotation_matrix_zyz_tensor(angles):
    """
    Convert multiple batches of Euler angles (roll, pitch, yaw) to rotation matrices using XYZ order.
    This function uses PyTorch tensors for better performance and supports multiple batches.

    :param angles: A tensor of shape (batch, num_angles, 3) where each entry contains (roll, pitch, yaw) in radians
    :return: A tensor of shape (batch, num_angles, 3, 3) containing rotation matrices
    """
    batch_size, num_angles, _ = angles.shape

    # Extract roll, pitch, yaw components
    roll, pitch, yaw = angles[..., 0], angles[..., 1], angles[..., 2]

    # Compute cosine and sine of the angles
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    # Compute the rotation matrices for XYZ order
    r00, r01, r02 = cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr
    r10, r11, r12 = sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr
    r20, r21, r22 = -sp, cp * sr, cp * cr

    # Stack the components into rotation matrices
    rotation_matrices = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)

    return rotation_matrices

