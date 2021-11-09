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



    def start_process(self, save_to_dir):
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
                    frames = frames=range(0, self.max_frames, self.frame_step)
                
                else:
                    frames = frames=range(0, len(cam2_files), self.frame_step)
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

                for idx, pose in enumerate(data.poses):
                    # self.K_list.append(P_rect_20)
                    # self.campose_list.append(pose)
                    inv_pose = compute_inv_homo_matrix(pose)
                    # self.Rt_list.append(inv_pose)

                    self.camera_info_list.append({
                        "K": P_rect_20.tolist(),
                        "Rt": inv_pose.tolist(),
                        "CamPose": pose.tolist()
                    })

                output_cameras_json_path = os.path.join(save_to_seq_path, 'cameras.json')
                with open(output_cameras_json_path, 'w') as f:
                    json.dump(self.camera_info_list,f)


                ## copy images files to the right directory and generate depth images

                P_rect_20 = data.calib.P_rect_20
                T_cam0_velo = data.calib.T_cam0_velo
                pc_gen = data.velo
                for idx in range(len(data.poses)):
                    pc0 = next(pc_gen)

                    pc0[:, -1] = np.ones(pc0.shape[0])
                    img = data.get_rgb(0)[0]
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    rows, cols = img.shape[:2]
                    img_size = [rows, cols]

                    trans_lidar_points = P_rect_20.dot(T_cam0_velo.dot(pc0.T)).T

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

                    trim_trans_lidar_points = trans_lidar_points[keep_idx,:]

                    trim_proj_lidar_points = proj_lidar_points[keep_idx,:]

                    depth = np.linalg.norm(trim_trans_lidar_points, axis=1)
                    # normalize_depth = (depth - np.min(depth)) / (np.max(depth)- np.min(depth))

                    depth_map = np.zeros((rows, cols), dtype=np.float32)

                    pixel_index = trim_proj_lidar_points[:,:2].astype(np.int32)
                    depth_map[pixel_index[:,1], pixel_index[:,0]] = depth

                    print(np.min(pixel_index, axis=0))
                    print(np.max(pixel_index, axis=0))

                    # normalize_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map)- np.min(depth_map))
                    # normalize_depth_map = depth_map
                    depth_map_filename = os.path.join(
                        save_to_seq_path, str(idx).zfill(self.n_digits_img_name)  + '_depth.tiff')
                    
                    # Using cv2.imwrite() method
                    # Saving the image
                    cv2.imwrite(depth_map_filename, depth_map)

                    rgb_img_filename = os.path.join(
                        save_to_seq_path, str(idx).zfill(self.n_digits_img_name)  + '_rgb.png')
                    
                    # Using cv2.imwrite() method
                    # Saving the image
                    cv2.imwrite(rgb_img_filename, img)


sequences = ['05']

converter = OdometryKittiMlgsnConverter(BASEDIR, sequences, frame_step=100, train_test_split=[1.0, 0.0])

converter.start_process('/media/data1/kitti/odokitti')