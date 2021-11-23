import numpy as np
import cv2
import os
import pykitti
from point_viz.converter import PointvizConverter

import matplotlib; matplotlib.use('Agg')
from copy import deepcopy

import glob
import json

## helper function

def compute_inv_homo_matrix(matrix):

    R_imu_cam2 = matrix[:3,:3]
    d_imu_cam2 = matrix[:3,3]
    inv_R_imu_cam2 = np.linalg.inv(R_imu_cam2)  
    d_cam2_imu = -inv_R_imu_cam2.dot(d_imu_cam2)

    T_imu_cam2 = np.eye(4, dtype=np.float32)
    T_imu_cam2[:3,:3] = inv_R_imu_cam2
    T_imu_cam2[:3,3] = d_cam2_imu
    return T_imu_cam2


def get_union_sets(conditions):
    output = conditions[0]
    for i in np.arange(1, len(conditions)):
        output = np.logical_and(output, conditions[i])
    return output


BASEDIR = '/media/data1/kitti/dataset'

RANGE_X = [0., 70.4]
RANGE_Y = [-40., 40.]
RANGE_Z = [-3., 1.]

class OdometryKittiMlgsnConverter:

    def __init__(self, basedir, sequences:list, 
                    frame_step: int=1, 
                    start_frame: int=0,
                    max_frames:int=None,
                    n_digits_seq_name:int=2,
                    n_digits_img_name:int=3,
                    train_test_split:list=[0.8, 0.2]):
        super().__init__()

        assert isinstance(sequences, list)
        assert os.path.exists(basedir) 
        assert isinstance(frame_step, int) or (frame_step is None)
        assert isinstance(max_frames, int) or (max_frames is None)
        self.basedir = basedir
        self.frame_step = frame_step
        self.start_frame = start_frame
        self.max_frames = max_frames
        self.sequences = sequences
        self.imtype = 'png'
        self.n_digits_seq_name = n_digits_seq_name
        self.n_digits_img_name = n_digits_img_name


        assert np.sum(train_test_split).astype(np.int32) == 1

        self.generate_split(train_test_split)
        self.generate_data_map()

    def generate_data_map(self ):
        self.seq_output_id_map = {}

        for sid, sequence in enumerate(self.sequences):
            self.seq_output_id_map[sequence] = str(sid).zfill(self.n_digits_seq_name)
    
    def generate_split(self, train_test_split):
        self.splitted_sequences_map = {}

        if int(train_test_split[0]) == 1:
            self.splitted_sequences_map['train'] = deepcopy(self.sequences)
            self.splitted_sequences_map['test'] = deepcopy(self.sequences)
        else:
            n_sequences = len(self.sequences)
            train_sequences = int(n_sequences * train_test_split[0])
            self.splitted_sequences_map['train'] = deepcopy(self.sequences[:train_sequences])
            self.splitted_sequences_map['test'] = deepcopy(self.sequences[train_sequences:])

        assert len(self.splitted_sequences_map['train']) > 0
        assert len(self.splitted_sequences_map['test']) > 0



    def start_process(self, save_to_dir, crop_img_size=None):
        """
            <save_to_dir>
                - odokitti
                  - train
                    - 00
                      - 000_rgb.png
                      - 000__depth.tiff
                      - cameras.json
                        >> [{"K": 4x4, "Rt": 4x4, "CamPose": 4x4}]
                  - test
        """
        print(self.seq_output_id_map)

        if os.path.exists(save_to_dir):
            if len(os.listdir(save_to_dir)) > 0:
                raise FileExistsError(save_to_dir + " is found. Please delete or save to new path.")
        else:
            os.makedirs(save_to_dir)

        split_keys_list = list(self.splitted_sequences_map.keys())


        min_depth = 99999999
        max_depth = -9999999

        min_x = 99999999
        min_y = 99999999
        min_z = 99999999
        max_x = -99999999
        max_y = -99999999
        max_z = -99999999

        for split in split_keys_list:
            # generate the directory structure
            save_to_split_path = os.path.join(save_to_dir, split)
            if not os.path.exists(save_to_split_path):
                os.makedirs(save_to_split_path)
            for sid, sequence in enumerate(self.splitted_sequences_map[split]):
                sequence_path = os.path.join(self.basedir, 'sequences', sequence)
                cam2_files = sorted(glob.glob(
                os.path.join(sequence_path, 'image_2',
                            '*.{}'.format(self.imtype))))
                # print(len(cam2_files))
                frames = None
                if self.max_frames is not None:
                    frames = frames=range(self.start_frame, self.start_frame+self.max_frames, self.frame_step)
                
                else:
                    frames = frames=range(self.start_frame, len(cam2_files), self.frame_step)
                print("Split: ", split, "\t sequence: ", sequence, " has ", 
                    str(len(cam2_files)), "frames. Processed only ", len(frames), " frames.")

                assert len(frames) > 0

                save_to_seq_path = os.path.join(save_to_split_path, 
                    self.seq_output_id_map[sequence])
                if not os.path.exists(save_to_seq_path):
                    os.makedirs(save_to_seq_path)

                data = pykitti.odometry(self.basedir, sequence, frames=frames)
                

                ##  generate camera pose files
                self.camera_info_list = []

                # self.K_list = [] # list of intrinsic matrix
                # self.Rt_list = [] # list of extrinsic matrix (inverse of camera pose)
                # self.campose_list = [] # list of camera pose matrix (inverse of extrinsic matrix)
                

                P_rect_20 = data.calib.P_rect_20 # intrinsic
                T2 = np.eye(4)
                T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
                
                K_cam2 = P_rect_20[0:3, 0:3]

                if(crop_img_size is not None):
        
                    min_u = int(round(K_cam2[0,2] - crop_img_size[1]//2))
                    max_u = int(round(K_cam2[0,2] + crop_img_size[1]//2))
                    min_v = int(round(K_cam2[1,2] - crop_img_size[0]//2))
                    max_v = int(round(K_cam2[1,2] + crop_img_size[0]//2))
                    K_cam2[0,2] = crop_img_size[1]//2
                    K_cam2[1,2] = crop_img_size[0]//2

                    # print("K_cam2: ", K_cam2)

                for idx, pose in enumerate(data.poses):
                    # self.K_list.append(P_rect_20)
                    # self.campose_list.append(pose)
                    # print("pose: ", pose)
                    # print("P_rect_20: ", P_rect_20)
                    # print("T2: ", T2)
                    pose = T2.dot(pose)
                    # proj_offset = np.eye(4)
                    # proj_offset[:3,3] = P_rect_20[:3,3]
                    
                    # pose = np.dot(proj_offset, pose)
                    inv_pose = compute_inv_homo_matrix(pose)
                    # print("pose: ", pose)
                    # exit()
                    # self.Rt_list.append(inv_pose)
                    # print(P_rect_20.shape)
                    # print(inv_pose.shape)
                    # inv_pose[:3,3] += P_rect_20[:3,3]

                    # print("P_rect20: ", P_rect_20)
                    # print("pose: ", pose)
                    # pose[:3,3] += P_rect_20[:3,3]
                    # print("pose: ", pose)


                    # exit()
                    cam2_pose = compute_inv_homo_matrix(deepcopy(inv_pose))
                    self.camera_info_list.append({
                        "K": P_rect_20[:3,:3].tolist(),
                        "Rt": inv_pose.tolist(),
                        "CamPose": pose.tolist()
                        # "Rt": pose.tolist(),
                        # "CamPose": inv_pose.tolist()
                    })
                    
                    point_pose = np.array([0,0,0,1])
                    pos = pose.dot(point_pose)

                    min_x = min(min_x, pos[0])
                    min_y = min(min_y, pos[1])
                    min_z = min(min_z, pos[2])
                    max_x = max(max_x, pos[0])
                    max_y = max(max_y, pos[1])
                    max_z = max(max_z, pos[2])


                output_cameras_json_path = os.path.join(save_to_seq_path, 'cameras.json')
                with open(output_cameras_json_path, 'w') as f:
                    json.dump(self.camera_info_list,f)


                ## copy images files to the right directory and generate depth images

                P_rect_20 = data.calib.P_rect_20
                T_cam0_velo = data.calib.T_cam0_velo
                T_cam2_velo = data.calib.T_cam2_velo
                pc_gen = data.velo
                for idx in range(len(data.poses)):
                    pc0 = next(pc_gen)

                    pc0[:, -1] = np.ones(pc0.shape[0])
                    img = data.get_rgb(0)[0]
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    rows, cols = img.shape[:2]
                    img_size = [rows, cols]

                    cam0_velo_points = T_cam0_velo.dot(pc0.T).T
                    trans_lidar_points = P_rect_20.dot(cam0_velo_points.T).T

                    # trans_lidar_points = np.transpose(T_cam2_velo.dot(pc0.transpose()))  # [n, 4]
                    proj_lidar_points = trans_lidar_points / trans_lidar_points[:, 2:3]
                    # keep_idx = get_union_sets([proj_lidar_points[:, 0] > 0,
                    #                             proj_lidar_points[:, 0] < cols,
                    #                             proj_lidar_points[:, 1] > 0,
                    #                             proj_lidar_points[:, 1] < rows])

                    keep_idx = get_union_sets([pc0[:, 0] > RANGE_X[0],
                                                pc0[:, 0] < RANGE_X[1],
                                                pc0[:, 1] > RANGE_Y[0],
                                                pc0[:, 1] < RANGE_Y[1],
                                                pc0[:, 2] > RANGE_Z[0],
                                                pc0[:, 2] < RANGE_Z[1],
                                                proj_lidar_points[:, 0] > 0,
                                                proj_lidar_points[:, 0] < cols,
                                                proj_lidar_points[:, 1] > 0,
                                                proj_lidar_points[:, 1] < rows])

                    # trim_trans_lidar_points = trans_lidar_points[keep_idx,:]

                    trim_proj_lidar_points = proj_lidar_points[keep_idx,:]


                    # cam2proj_offset = np.eye(4)
                    # cam2proj_offset[:3,3] = P_rect_20[:3,3]

                    # print("cam0_velo_points: ", cam0_velo_points.shape)
                    # print("cam2proj_offset: ", cam2proj_offset)
                    
                    # cam2_velo_points = np.dot(cam2proj_offset, cam0_velo_points.T).T

                    cam2_velo_points = T_cam2_velo.dot(pc0.T).T
                    trim_cam2_velo_points = cam2_velo_points[keep_idx,: ]
                    # print("P_rect_20: ", P_rect_20)
                    # print("cam0_velo_points: ", cam0_velo_points)
                    # print("cam2_velo_points: ", cam2_velo_points)
                    # exit()

                    depth = np.linalg.norm(trim_cam2_velo_points, axis=1)
                    # normalize_depth = (depth - np.min(depth)) / (np.max(depth)- np.min(depth))
                    # depth = trim_cam2_velo_points[:, 2]

                    min_depth = min(np.min(depth), min_depth)
                    max_depth = max(np.max(depth), max_depth)

                    # depth = np.clip(depth, 0.0, 50.0)

                    depth_map = np.zeros((rows, cols), dtype=np.float32)

                    pixel_index = trim_proj_lidar_points[:,:2].astype(np.int32)
                    depth_map[pixel_index[:,1], pixel_index[:,0]] = depth

                    # print(np.min(pixel_index, axis=0))
                    # print(np.max(pixel_index, axis=0))

                    # normalize_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map)- np.min(depth_map))
                    # normalize_depth_map = depth_map
                    depth_map_filename = os.path.join(
                        save_to_seq_path, str(idx).zfill(self.n_digits_img_name)  + '_depth.tiff')
                    
                    # Using cv2.imwrite() method
                    # Saving the image
                    if(crop_img_size is not None):
                        # print("min_u: ", min_u, " max_u: ", max_u,
                        #     " min_v: ", min_v, " max_v: ", max_v
                        # )
                        # print("len(u): ", max_u-min_u)
                        # print("len(v): ", max_v-min_v)
                        cv2.imwrite(depth_map_filename, depth_map[min_v:max_v, min_u:max_u])
                    else:
                        cv2.imwrite(depth_map_filename, depth_map)

                    rgb_img_filename = os.path.join(
                        save_to_seq_path, str(idx).zfill(self.n_digits_img_name)  + '_rgb.png')
                    
                    # Using cv2.imwrite() method
                    # Saving the image
                    if(crop_img_size is not None):
                        cv2.imwrite(rgb_img_filename, img[min_v:max_v, min_u:max_u,:])
                    else:
                        cv2.imwrite(rgb_img_filename, img)
        print("Min depth: ", min_depth, " \t max_depth: ", max_depth)
        print("Min depth: ", min_depth, " \t max_depth: ", max_depth)

        print( "min_x ,min_y ,min_z ,max_x ,max_y ,max_z ,",
                    min_x ,
                    min_y ,
                    min_z ,
                    max_x ,
                    max_y ,
                    max_z 
        )
sequences = ['00', '01', '02',  '05']

converter = OdometryKittiMlgsnConverter(BASEDIR, sequences, frame_step=5, start_frame = 0, max_frames=500, train_test_split=[1.0, 0.0])

# converter.start_process('/media/data1/kitti/odokitti')
# converter.start_process('/media/data1/kitti/odokitti', crop_img_size=[360,360]) #(row, column)
# converter.start_process('/media/data1/kitti/test_odokitti_crop', crop_img_size=[360,360]) #(row, column)
# converter.start_process('/media/data1/kitti/odokitti_multi_seq')
# converter.start_process('/media/data1/kitti/odokitti_multi_seq', crop_img_size=[360,360]) #(row, column)
converter.start_process('/media/data1/kitti/odokitti_multi_5_seq', crop_img_size=[360,360]) #(row, column)