import os
import torch
import numpy as np


def listdir_nohidden(path):
    mylist = [f for f in os.listdir(path) if not f.startswith('.')]
    return mylist



def compute_inv_homo_matrix(matrix):

    R_imu_cam2 = matrix[:3,:3]
    d_imu_cam2 = matrix[:3,3] #torch.unsqueeze(,axis=1)
    inv_R_imu_cam2 = torch.inverse(R_imu_cam2)  
    # print(R_imu_cam2.shape, d_imu_cam2.shape, inv_R_imu_cam2.shape)
    d_cam2_imu = torch.matmul(-inv_R_imu_cam2, d_imu_cam2)

    T_imu_cam2 = torch.eye(4, dtype=torch.float32).to(matrix.device)
    T_imu_cam2[:3,:3] = inv_R_imu_cam2
    T_imu_cam2[:3,3] = d_cam2_imu
    return T_imu_cam2


def batch_compute_inv_homo_matrix(matrix):

    inverse_matrix = torch.zeros_like(matrix, device=matrix.device)
    if(len(list(inverse_matrix.shape))) == 3:
        for i in range(inverse_matrix.shape[0]):
                inverse_matrix[i,:,:] = compute_inv_homo_matrix(matrix[i,:,:])

    elif(len(list(inverse_matrix.shape))) == 4:
        for i in range(inverse_matrix.shape[0]):
            for j in range(inverse_matrix.shape[1]):
                inverse_matrix[i,j,:,:] = compute_inv_homo_matrix(matrix[i,j,:,:])
    
    elif(len(list(inverse_matrix.shape))) == 5:
        for i in range(inverse_matrix.shape[0]):
            for j in range(inverse_matrix.shape[1]):
                for k in range(inverse_matrix.shape[2]):
                    inverse_matrix[i,j,k, :,:] = compute_inv_homo_matrix(matrix[i,j, k, :,:])

    else:
        raise NotImplementedError("Shape of matrix: ", matrix.shape)

    return inverse_matrix


def normalize_trajectory(Rt, center='first', normalize_rotation=True):
    assert center in ['first', 'mid'], 'center must be either "first" or "mid", got {}'.format(center)

    seq_len = Rt.shape[1]

    if center == 'first':
        origin_frame = 0
    elif center == 'mid':
        origin_frame = seq_len // 2
    else:
        # return unmodified Rt
        return Rt

    if normalize_rotation:
        origins = Rt[:, origin_frame : origin_frame + 1].expand_as(Rt).reshape(-1, 4, 4).inverse()
        # origins = Rt[:, origin_frame : origin_frame + 1].expand_as(Rt).reshape(-1, 4, 4)
        # origins = batch_compute_inv_homo_matrix(origins)
        normalized_Rt = torch.bmm(Rt.view(-1, 4, 4), origins)
        normalized_Rt = normalized_Rt.view(-1, seq_len, 4, 4)
    else:
        # camera_pose = batch_compute_inv_homo_matrix(Rt)

        camera_pose = Rt. inverse()
        origins = camera_pose[:, origin_frame : origin_frame + 1, :3, 3]
        camera_pose[:, :, :3, 3] = camera_pose[:, :, :3, 3] - origins
        normalized_Rt = camera_pose.inverse()

        # normalized_Rt = batch_compute_inv_homo_matrix(camera_pose)



    return normalized_Rt


def random_rotation_augment(trajectory_Rt):
    # given a trajectory, apply a random rotation
    angle = np.random.randint(-180, 180)
    angle = np.deg2rad(angle)
    _rand_rot = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    rand_rot = torch.eye(4)
    rand_rot[:3, :3] = torch.from_numpy(_rand_rot).float()

    for i in range(len(trajectory_Rt)):
        trajectory_Rt[i] = trajectory_Rt[i].mm(rand_rot)

    return trajectory_Rt
