import numpy as np
import cv2
import os
import pykitti
from point_viz.converter import PointvizConverter

import matplotlib; matplotlib.use('Agg')
from copy import deepcopy

basedir = '../dataset'
sequence = '05'

output_path = '/media/data1/kitti/python/artifacts'

# The 'frames' argument is optional - default: None, which loads the whole dataset.
# Calibration, timestamps, and IMU data are read automatically. 
# Camera and velodyne data are available via properties that create generators
# when accessed, or through getter methods that provide random access.
data = pykitti.odometry(basedir, sequence, frames=range(0, 10, 1))

# dataset.calib:         Calibration data are accessible as a named tuple
# dataset.timestamps:    Timestamps are parsed into a list of datetime objects
# dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
# dataset.camN:          Returns a generator that loads individual images from camera N
# dataset.get_camN(idx): Returns the image from camera N at idx  
# dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
# dataset.get_gray(idx): Returns the monochrome stereo pair at idx  
# dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
# dataset.get_rgb(idx):  Returns the RGB stereo pair at idx  
# dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
# dataset.get_velo(idx): Returns the velodyne scan at idx  

# point_velo = np.array([0,0,0,1])
# point_cam0 = data.calib.T_cam0_velo.dot(point_velo)

# point_imu = np.array([0,0,0,1])
# point_w = [o.T_w_imu.dot(point_imu) for o in data.oxts]
print(data.poses)

point_pose = np.array([0,0,0,1])
# point_w = [o.T_w_imu.dot(point_imu) for o in data.oxts]

point_pose_list = []

for idx, pose in enumerate(data.poses):
    print(pose.shape)
    point_pose_list.append(pose.dot(point_pose))


point_pose_np = np.stack(point_pose_list, axis=0)

print(point_pose_np.shape)
color_map = matplotlib.cm.get_cmap('jet')

t = np.arange(len(point_pose_np)).astype(np.float32)

t /= float(len(point_pose_np))
t_color = color_map(t)[:,:3] * 255.0

Converter = PointvizConverter(home=output_path + '/eda_kitti_odometry')
Converter.compile(task_name="Pc_Generator_valid",
                coors=point_pose_np[:,[0,1,2]],
                intensity=point_pose_np[:,2],
                default_rgb = t_color)

def compute_inv_homo_matrix(matrix):

    R_imu_cam2 = matrix[:3,:3]
    d_imu_cam2 = matrix[:3,3]
    inv_R_imu_cam2 = np.linalg.inv(R_imu_cam2)  
    d_cam2_imu = -inv_R_imu_cam2.dot(d_imu_cam2)

    T_imu_cam2 = np.eye(4, dtype=np.float32)
    T_imu_cam2[:3,:3] = inv_R_imu_cam2
    T_imu_cam2[:3,3] = d_cam2_imu
    return T_imu_cam2


point_pose = np.array([0,0,0,1])
# point_w = [o.T_w_imu.dot(point_imu) for o in data.oxts]

point_pose_list = []
inv_point_pose_list = []
inv_pose_list = []

for idx, pose in enumerate(data.poses):
    print(pose.shape)

    point_pose_list.append(pose.dot(point_pose))
    inv_pose = compute_inv_homo_matrix(pose)
    inv_pose_list.append(inv_pose)
    inv_point_pose_list.append(inv_pose.dot(point_pose))


inv_point_pose_np = np.stack(inv_point_pose_list, axis=0)

print(inv_point_pose_np.shape)

color_map = matplotlib.cm.get_cmap('jet')

t = np.arange(len(inv_point_pose_np)).astype(np.float32)

t /= float(len(inv_point_pose_np))
# t_color = []
# for i in range(len(t)):
t_color = color_map(t)[:,:3] * 255.0



Converter = PointvizConverter(home=output_path + '/eda_kitti_odometry')
Converter.compile(task_name="inverse_pose",
                coors=inv_point_pose_np[:,[0,1,2]],
                intensity=inv_point_pose_np[:,2],
                default_rgb=t_color)


# project lidar to image plane


def get_union_sets(conditions):
    output = conditions[0]
    for i in np.arange(1, len(conditions)):
        output = np.logical_and(output, conditions[i])
    return output
range_x = [0., 70.4]
range_y = [-40., 40.]
range_z = [-3., 1.]
T_cam2_velo = data.calib.T_cam2_velo
P_rect_20 = data.calib.P_rect_20
T_cam0_velo = data.calib.T_cam0_velo
K_cam2 = data.calib.K_cam2

pc_gen = data.velo

pc0 = next(pc_gen)

lidar_intensity = deepcopy(pc0[:, -1])
pc0[:, -1] = np.ones(pc0.shape[0])
img = data.get_rgb(0)[0]
img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
rows, cols = img.shape[:2]
img_size = [rows, cols]

# trans_lidar_points = P_rect_20.dot(T_cam0_velo.dot(pc0.T)).T
trans_lidar_points = K_cam2.dot(T_cam2_velo.dot(pc0.T)[:3,:]).T

# trans_lidar_points = np.transpose(T_cam2_velo.dot(pc0.transpose()))  # [n, 4]
proj_lidar_points = trans_lidar_points / trans_lidar_points[:, 2:3]
# keep_idx = get_union_sets([proj_lidar_points[:, 0] > 0,
#                             proj_lidar_points[:, 0] < cols,
#                             proj_lidar_points[:, 1] > 0,
#                             proj_lidar_points[:, 1] < rows])

keep_idx = get_union_sets([pc0[:, 0] > range_x[0],
                            pc0[:, 0] < range_x[1],
                            pc0[:, 1] > range_y[0],
                            pc0[:, 1] < range_y[1],
                            pc0[:, 2] > range_z[0],
                            pc0[:, 2] < range_z[1],
                            proj_lidar_points[:, 0] > 0,
                            proj_lidar_points[:, 0] < cols,
                            proj_lidar_points[:, 1] > 0,
                            proj_lidar_points[:, 1] < rows])

trim_trans_lidar_points = trans_lidar_points[keep_idx,:]

trim_proj_lidar_points = proj_lidar_points[keep_idx,:]

print(proj_lidar_points.shape)
print(trim_proj_lidar_points.shape)

depth = np.linalg.norm(trim_trans_lidar_points, axis=1)
normalize_depth = (depth - np.min(depth)) / (np.max(depth)- np.min(depth))
print(depth.shape)
print(depth)
# print(trim_proj_lidar_points[:4,:])

depth_map = np.zeros((rows, cols), dtype=np.float32)

pixel_index = trim_proj_lidar_points[:,:2].astype(np.int32)

# for i in range(len(pixel_index)):
depth_map[pixel_index[:,1], pixel_index[:,0]] = depth
# print(depth_map[:10,:10]) 

print(np.min(pixel_index, axis=0))
print(np.max(pixel_index, axis=0))

normalize_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map)- np.min(depth_map))
# normalize_depth_map = depth_map
filename = output_path + '/depthMap.jpg'
  
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, (normalize_depth_map*255.0).astype(np.uint8))
filename = output_path + '/cam2Odo.jpg'
  
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, img)


# merge depth and cam2 image

cam2_depth_merge = deepcopy(img)
print(cam2_depth_merge.shape)
# print(cam2_depth_merge[pixel_index[:,1], pixel_index[:,0], :3].shape)
print(np.repeat(np.array([[255.0, 0.0, 0.0]]), len(pixel_index), axis=0).shape)
# cam2_depth_merge[pixel_index[:,1], pixel_index[:,0],:] = normalize_depth[:,np.newaxis] * np.repeat(np.array([[0.0, 255.0, 0.0]]), len(pixel_index), axis=0)

cam2_depth_merge[pixel_index[:,1], pixel_index[:,0],:] = color_map(normalize_depth)[:,:3] * 255.0

filename = output_path + '/cam2_depth_Odo.jpg'
  
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, cam2_depth_merge.astype(np.uint8))

Converter = PointvizConverter(home=output_path + '/eda_kitti_odometry')
Converter.compile(task_name="trim_trans_lidar_points",
                coors=trim_trans_lidar_points[:,[0,1,2]],
                intensity=trim_trans_lidar_points[:,2])

