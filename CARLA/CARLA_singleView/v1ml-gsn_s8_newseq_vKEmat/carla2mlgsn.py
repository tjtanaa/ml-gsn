import os
import glob
import shutil
from typing import ContextManager
import pandas as pd
import numpy as np
import json
import cv2
import math


def create_json(train_scheme, train, fullposeslidar_file, lidar2cam_file, save_folder, filename='cameras.json'):
    org_json = '/disk2/kaho/projects_on_disk2/nerf/v1ml-gsn_s8_newseq_vKEmat/data/carla_sequence_1392x1024_554frames/cameras.json'
    f =open(org_json)
    data = json.load(f)
    f.close()
    content = []
    K = np.array([[330.33166091308163, 0.0, 240.0], [-0.0, -449.0446015537204, -240.0], [-0.0, -0.0, -1.0]], dtype='float')
    print('K')
    print(K)
    K = K.tolist()

    for i in range(0, 300):
        # fy = K[0,0] / (1024.0/480.0)
        # fx = 480.0 / (2.0*math.tan(72*math.pi/360))
        # K[0,0] = fx
        # # K[1,1] = fx   # should not be the same after resize
        # K[1,1] = fy
        # K[0,2] = 480.0/2
        # K[1,2] = 480.0/2
        # # print('Original K: ')
        # # print(K)
        # # (x, y ,z) -> (y, -z, x)
        # K[1,:] = -K[1,:]
        # K[2,:] = -K[2,:]
        # # print('Corrected twice K: ')
        # # print(K)
        # K = K.tolist()

        Rt = np.array(data[i]['Rt'], dtype='float')
        x = -(Rt[1,:].copy())
        y = Rt[2,:].copy()
        z = -(Rt[0,:].copy())
        Rt[0,:] = x
        Rt[1,:] = y
        Rt[2,:] = z
        Rt = Rt.tolist()
        content.append({'K':K, 'Rt':Rt})
        # assert (1==0)

    assert (2*len(content)==len(glob.glob(save_folder+'/*')))
    
    with open(os.path.join(save_folder, filename), 'w') as d:
        json.dump(content, d)
        
    return True

def preprocess_imgs(train_scheme, new_folder_list, org_folder_list):
    imgs_0 = sorted(glob.glob(org_folder_list[1]+'/*_rgb.png'))
    depths_0 = sorted(glob.glob(org_folder_list[2]+'/*_depth.png'))
    assert (len(imgs_0)==554)
    assert (len(depths_0)==554)

    if train_scheme==8:
        # return
        imgs_train_0, depths_train_0, imgs_test_0, depths_test_0 = [],[],[],[]

        imgs_train_0 = imgs_0[0:300:1]
        depths_train_0 = depths_0[0:300:1]
        assert (len(imgs_train_0)==300)
        assert (len(depths_train_0)==300)
        for i in range(len(imgs_train_0)):
            img = cv2.imread(imgs_train_0[i])#, cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(depths_train_0[i])
            img = cv2.resize(img,  (480, 480), interpolation = cv2.INTER_AREA)
            depth = cv2.resize(depth,  (480, 480), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(new_folder_list[-2], str(i).zfill(3)+'_rgb.png'), img)
            cv2.imwrite(os.path.join(new_folder_list[-2], str(i).zfill(3)+'_depth.png'), depth)
        # TODO create_json
        res = create_json(train_scheme, True, org_folder_list[3], org_folder_list[-2], new_folder_list[-2])

        imgs_test_0 = imgs_0[0:300:1]
        depths_test_0 = depths_0[0:300:1]
        assert (len(imgs_test_0)==len(depths_test_0))
        print('len(imgs_test_0): ', len(imgs_test_0))
        for i in range(len(imgs_test_0)):
            img = cv2.imread(imgs_test_0[i])#, cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(depths_test_0[i])
            img = cv2.resize(img,  (480, 480), interpolation = cv2.INTER_AREA)
            depth = cv2.resize(depth,  (480, 480), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(new_folder_list[-1], str(i).zfill(3)+'_rgb.png'), img)
            cv2.imwrite(os.path.join(new_folder_list[-1], str(i).zfill(3)+'_depth.png'), depth)
        # TODO create_json
        res = create_json(train_scheme, False, org_folder_list[3], org_folder_list[-2], new_folder_list[-1])


def create_folder(train_scheme, new_base_folder):
    new_folder_name = None
    if train_scheme==1:
        new_folder_name = 'new_Town02_fullleft'
    elif train_scheme==2:
        new_folder_name = 'new_Town02_fullleftresize'
    elif train_scheme==3:
        new_folder_name = 'new_Town02_left25'
    elif  train_scheme==5:
        new_folder_name = 'new_Town02_leftright50'
    elif train_scheme==8:
        new_folder_name = 'new_Town02_left25resize512'
    elif  train_scheme==10:
        new_folder_name = 'new_Town02_leftright50resize512'
    elif train_scheme==11:
        new_folder_name = 'new_Town02_left25crop512'
    elif  train_scheme==12:
        new_folder_name = 'new_Town02_leftright50crop512'
    assert (new_folder_name!=None)
    
    new_folder = os.path.join(new_base_folder, new_folder_name)
    new_train_folder = os.path.join(new_folder, 'train')
    new_train_folder_00 = os.path.join(new_train_folder, '00')
    
    new_test_folder = os.path.join(new_folder, 'test')
    new_test_folder_00 = os.path.join(new_test_folder, '00')
    new_folder_list = [new_folder, new_train_folder, new_test_folder, new_train_folder_00, new_test_folder_00]
    
    if train_scheme in [5,10,12]:
        new_train_folder_00 = os.path.join(new_train_folder, '01')
        new_test_folder_01 = os.path.join(new_test_folder, '01')
        new_folder_list = new_folder_list + [new_train_folder_00, new_test_folder_01]
        
    for folder in new_folder_list:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
    
    return new_folder_list


def main(train_scheme, new_base_folder, org_folder_list):
    new_folder_list = create_folder(train_scheme, new_base_folder)
    new_folder_list = preprocess_imgs(train_scheme, new_folder_list, org_folder_list)
                                               
if __name__ == '__main__':
    train_scheme = 8
    new_base_folder = '/disk2/kaho/projects_on_disk2/nerf/v1ml-gsn_s8_newseq_vKEmat/data'
    
    org_folder = '/disk2/kaho/projects_on_disk2/nerf/v1ml-gsn_s8_newseq_vKEmat/data/carla_sequence_1392x1024_554frames'
    org_imgs_folder = os.path.join(org_folder, 'rgb')
    org_depths_folder = os.path.join(org_folder, 'depth')
    org_full_poses_lidar = os.path.join(org_folder, 'full_poses_lidar.txt')
    org_full_ts_camera = os.path.join(org_folder, 'full_ts_camera.txt')
    org_lidar_to_cam0 = os.path.join(org_folder, 'lidar_to_cam0.txt')
    org_lidar_to_cam1 = os.path.join(org_folder, 'lidar_to_cam1.txt')
    org_folder_list = [org_folder, org_imgs_folder, org_depths_folder, org_full_poses_lidar, org_full_ts_camera, org_lidar_to_cam0, org_lidar_to_cam1]
    
    # train_test_ratio = 0.8
    
    # train_scheme_list = [1,2,3,5,8,10,11,12]
    # for i in range(len(train_scheme)):
        # main(train_scheme_list[i], new_base_folder, org_folder_list)

    main(train_scheme, new_base_folder, org_folder_list)