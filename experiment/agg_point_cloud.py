import numpy as np
import cv2
import os
import pykitti
from point_viz.converter import PointvizConverter


basedir = '../'
date = '2011_09_26'
drive = '0093'

output_path = '/media/data1/kitti/python/artifacts'

# The 'frames' argument is optional - default: None, which loads the whole dataset.
# Calibration, timestamps, and IMU data are read automatically. 
# Camera and velodyne data are available via properties that create generators
# when accessed, or through getter methods that provide random access.
frame_sequence = range(0, 100, 10)
data = pykitti.raw(basedir, date, drive, frames=frame_sequence)

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

point_velo = np.array([0,0,0,1])
point_cam0 = data.calib.T_cam0_velo.dot(point_velo)

point_imu = np.array([0,0,0,1])
point_w = [o.T_w_imu.dot(point_imu) for o in data.oxts]

for cam0_image in data.cam0:
    # do something
    pass

cam2_image, cam3_image = data.get_rgb(3)

# print(dir(cam2_image))
# print(cam2_image.size)
# print(type(cam2_image))
filename="cam2.png"
open_cv_image = np.array(cam2_image) 
open_cv_image = open_cv_image[:, :, ::-1].copy()
cv2.imwrite(os.path.join(output_path, filename), open_cv_image)



# print(len(data.oxts))
# print(data.timestamps)

# print(data.oxts[0])


Converter = PointvizConverter(home='/media/data1/kitti/python/artifacts/agg_pointcoud')

point_imu = np.array([0,0,0,1])
point_w = [o.T_w_imu.dot(point_imu) for o in data.oxts]
point_w = np.array(point_w)
# point_w = np.concatenate(point_w,axis=0)
# print(point_w.shape)
# print(len(point_w))
# for i in range(len(point_w)):
#     print(point_w[i])
Converter.compile(task_name="point_w",
                coors=point_w[:,[2,0,1]],
                intensity=np.arange(point_w.shape[0]).astype(np.float32))
# pc_generator = data.velo
# for i in range(len(frame_sequence)):
#     pc = next(pc_generator)



    # Converter.compile(task_name="Pc_seq_valid_" + str(i),
    #                 coors=pc[:,[1,2,0]],
    #                 intensity=pc[:,3])


# point_cam2 = np.array([0,0,0,1])
## compute T_imu_cam2
T_cam2_imu = data.calib.T_cam2_imu

R_imu_cam2 = T_cam2_imu[:3,:3]
d_imu_cam2 = T_cam2_imu[:3,3]
# print(T_cam2_imu)
# print(R_imu_cam2)
# print(d_imu_cam2)
inv_R_imu_cam2 = np.linalg.inv(R_imu_cam2) 
# print(R_imu_cam2)
# print(inv_R_imu_cam2)
# T_imu_cam2 = 
print(inv_R_imu_cam2 @ R_imu_cam2 )

d_cam2_imu = -inv_R_imu_cam2.dot(d_imu_cam2)
print(d_cam2_imu)

T_imu_cam2 = np.eye(4, dtype=np.float32)
T_imu_cam2[:3,:3] = inv_R_imu_cam2
T_imu_cam2[:3,3] = d_cam2_imu

print(T_imu_cam2)
# print(T_cam2_imu)
# print(d_imu_cam2)

## compute T_imu_velo
T_velo_imu = data.calib.T_velo_imu

R_imu_velo = T_velo_imu[:3,:3]
d_imu_velo = T_velo_imu[:3,3]
print(T_velo_imu)
print(R_imu_velo)
print(d_imu_velo)
inv_R_imu_velo = np.linalg.inv(R_imu_velo) 
print(R_imu_velo)
print(inv_R_imu_velo)
# T_imu_cam2 = 
print(inv_R_imu_velo @ R_imu_velo )

d_velo_imu = -inv_R_imu_velo.dot(d_imu_velo.T)
print(d_velo_imu)

T_imu_velo = np.eye(4, dtype=np.float32)
T_imu_velo[:3,:3] = inv_R_imu_velo
T_imu_velo[:3,3] = d_velo_imu

print(T_imu_velo)

T_w_cam2_list = []
T_w_velo_list = []

point_w_world = []
velo_world = []

point_cam2 = np.array([0,0,0,1])
pc_gen = data.velo

origin = None

for idx, o in enumerate(data.oxts):
    T_w_cam2 = o.T_w_imu.dot(T_imu_cam2)
    T_w_cam2_list.append(T_w_cam2) 
    T_w_velo = o.T_w_imu.dot(T_imu_velo)
    T_w_velo_list.append(T_w_velo)
    point_w = T_w_cam2.dot(point_cam2)
    if origin is None:
        origin = point_w

    # diff_z = point_w[2] - origin[2]
    # point_w[2] = origin[2]
    point_w_world.append(point_w)
    pc_t = next(pc_gen)
    # print(T_w_cam2)
    homogenuous_velo_points = np.ones_like(pc_t)
    homogenuous_velo_points[:,:3] = pc_t[:,:3]
    # print(homogenuous_velo_points[:5,:])
    velo_w = T_w_velo.dot(homogenuous_velo_points.T).T
    # velo_w[:,2] -= diff_z
    velo_world.append(velo_w)

point_w_world = np.array(point_w_world)
velo_world_np = np.concatenate(velo_world,axis=0)
print(velo_world_np.shape)
print(point_w_world.shape)
# subsample_size = 1000000
# subsample_pt_ind = np.random.choice(velo_world_np.shape[0], size=subsample_size, replace=False)
# velo_world_np = velo_world_np[subsample_pt_ind,:]
all_points = np.concatenate([point_w_world, velo_world_np], axis=0)


Converter.compile(task_name="all_points",
                coors=all_points[:,[1,2,0]],
                intensity=np.arange(all_points.shape[0]).astype(np.float32))



# import glob
# import pykitti.utils as utils

# self_drive = date + '_drive_' + drive + '_' + 'sync'
# data_path = os.path.join(basedir, date, self_drive)

# oxts_files = sorted(glob.glob(
#     os.path.join(data_path, 'oxts', 'data', '*.txt')))

# # print(oxts_files)
# oxts_files = utils.subselect_files(
#                 oxts_files, range(0,10,1))

# def load_oxts_packets_and_poses(oxts_files):
#     """Generator to read OXTS ground truth data.

#        Poses are given in an East-North-Up coordinate system 
#        whose origin is the first GPS position.
#     """
#     # Scale for Mercator projection (from first lat value)
#     scale = None
#     # Origin of the global coordinate system (first GPS position)
#     origin = None

#     oxts = []
#     list_t = []

#     for filename in oxts_files:
#         with open(filename, 'r') as f:
#             for line in f.readlines():
#                 line = line.split()
#                 # Last five entries are flags and counts
#                 line[:-5] = [float(x) for x in line[:-5]]
#                 line[-5:] = [int(float(x)) for x in line[-5:]]

#                 packet = utils.OxtsPacket(*line)

#                 if scale is None:
#                     scale = np.cos(packet.lat * np.pi / 180.)

#                 R, t = utils.pose_from_oxts_packet(packet, scale)

#                 if origin is None:
#                     origin = t
                
#                 list_t.append( R.dot(t-origin))

#                 T_w_imu = utils.transform_from_rot_trans(R, t - origin)

#                 oxts.append(utils.OxtsData(packet, T_w_imu))

#     return oxts, list_t

# print(oxts_files)

# _, list_t = load_oxts_packets_and_poses(oxts_files)

# print(list_t)


# t_points = np.stack(list_t, axis=0)

# t_points = t_points - np.min(t_points,axis=0)

# Converter.compile(task_name="t_points",
#                 coors=t_points[:,[1,2,0]],
#                 intensity=np.arange(t_points.shape[0]).astype(np.float32))