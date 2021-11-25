import os
import glob
import shutil
from typing import ContextManager
import pandas as pd
import numpy as np
import json
import cv2
import math


def parse_fullposeslidar(file):
    df = pd.read_csv(file, sep=" ")
    cols = df.columns.tolist()
    df.columns = cols[1:]+['none']
    del df['none']
    df['pos'] = df[df.columns.tolist()[:-1]].values.tolist()             # *lidar pos in array 
    # df['posstring'] = df['pos'].agg(lambda x: ','.join(map(str, x)))    # *array 2 posstring 2 cehck num of unique value 
    print('Parse full_poses_lidar.txt')
    print(df.head(1))
    return df

def parse_lidar2cam(file):
    df = pd.read_csv(file, sep=" ")
    cols = df.columns.tolist()
    df.columns = [cols[0][1:]] + cols[1:]
    df['pos'] = df[df.columns.tolist()[:]].values.tolist() 
    print('Parse lidar_to_cam.txt')
    print(df.head(1))
    return df



def create_json(train_scheme, train, fullposeslidar_file, lidar2cam_file, save_folder, filename='cameras.json'):
    df = parse_fullposeslidar(fullposeslidar_file)
    df_clip = df[0::100]
    df_cam = parse_lidar2cam(lidar2cam_file)
    lidar2cam = np.array(df_cam['pos'][0], dtype='float')
    lidar2cam = np.vstack((np.reshape(lidar2cam, (3,4)), [0.0, 0.0, 0.0, 1.0]))
    
    # *https://codeyarns.com/tech/2015-09-08-how-to-compute-intrinsic-camera-matrix-for-a-camera.html
    # *https://github.com/carla-simulator/carla/blob/a1b37f7f1cf34b0f6f77973c469926ea368d1507/PythonAPI/examples/lidar_to_camera.py
    # *line 122, In this case Fx and Fy are the same since the pixel aspect ratio is 1
    fx = 1392.0 / (2.0*math.tan(72*math.pi/360))
    # fy = 1024.0 / (2.0*tan(FOV_y*math.pi/360))
    K = np.array([[fx, 0.0, 1392.0/2.0], [0.0, fx, 1024.0/2.0], [0.0, 0.0, 1.0]], dtype='float')
    K_topleft, K_topright, K_botleft, K_botright = None, None, None, None
    
    if train_scheme==2:
        fy = fx / (1024.0/480.0)
        fx = 480.0 / (2.0*math.tan(72*math.pi/360))
        K[0,0] = fx
        # K[1,1] = fx   # should not be the same after resize
        K[1,1] = fy
        K[0,2] = 480.0/2
        K[1,2] = 480.0/2
    elif train_scheme==3:
        df_clip = df_clip[0::25] if train else df_clip[int(25/2)::25]
    elif train_scheme==5:
        df_clip = df_clip[0::50] if train else df_clip[int(50/2)::50]
    elif train_scheme==8 or 10:
        if train_scheme==8:
            df_clip = df_clip[0::25] if train else df_clip[int(25/2)::25]
        else:
            df_clip = df_clip[1110:1410:1] if train else df_clip[1110:1410:1]
        fy = fx / (1024.0/480.0)
        fx = 480.0 / (2.0*math.tan(72*math.pi/360))
        # print('fx: ',fx)
        # print(type(fx))
        # print(K[0,0])
        # print(type(K[0,0]))
        K[0,0] = fx
        # K[1,1] = fx   # should not be the same after resize
        K[1,1] = fy
        K[0,2] = 480.0/2
        K[1,2] = 480.0/2
    elif train_scheme==11 or train_scheme==12:
        if train_scheme==11:
            df_clip = df_clip[0::25] if train else df_clip[int(25/2)::25]
        else:
            df_clip = df_clip[0::50] if train else df_clip[int(50/2)::50]
        K_topleft = K.copy()
        K_topleft[0,2] = 480
        K_topleft[1,2] = 480
        K_topright = K.copy()
        K_topright[0,2] = 0
        K_topright[1,2] = 480
        K_botleft = K.copy()
        K_botleft[0,2] = 480
        K_botleft[1,2] = 0   
        K_botright = K.copy()
        K_botright[0,2] = 0
        K_botright[1,2] = 0

    
    content = []
    if train_scheme in [1,2,3,5,8,10]:
        print('save_folder', save_folder)
        # print(df_clip.shape[0], len(glob.glob(save_folder+'/*')))
        assert (2*df_clip.shape[0]==len(glob.glob(save_folder+'/*')))
        print('Original K: ')
        print(K)
        K[1,:] = -K[1,:]
        K[2,:] = -K[2,:]
        print('Corrected K: ')
        print(K)
        K = K.tolist()
        for index, row in df_clip.iterrows():
            lidar_pose = np.array(row['pos'], dtype='float')
            lidar_pose = np.vstack((np.reshape(lidar_pose, (3,4)), [0.0, 0.0, 0.0, 1.0]))
            world2lidar = np.linalg.inv(lidar_pose)
            # print('lidar_pose: ', lidar_pose)
            # print('world2lidar: ', world2lidar)
            # print('lidar2cam: ', lidar2cam)
            world2cam = lidar2cam.dot(world2lidar)
            x = -(world2cam[1,:].copy())
            y = world2cam[2,:].copy()
            z = -(world2cam[0,:].copy())
            world2cam[0,:] = x
            world2cam[1,:] = y
            world2cam[2,:] = z
            Rt = world2cam.tolist()
            content.append({'K':K, 'Rt':Rt})
    else: #*crop cases 11,12
        assert (4*2*df_clip.shape[0]==len(glob.glob(save_folder+'/*')))
        
        K_topleft = K_topleft.tolist()
        K_topright = K_topright.tolist()
        K_botleft = K_botleft.tolist()
        K_botright = K_botright.tolist()
        for index, row in df_clip.iterrows():
            lidar_pose = np.array(row['pos'], dtype='float')
            lidar_pose = np.vstack((np.reshape(lidar_pose, (3,4)), [0.0, 0.0, 0.0, 1.0]))
            world2lidar = np.linalg.inv(lidar_pose)
            # print('lidar_pose: ', lidar_pose)
            # print('world2lidar: ', world2lidar)
            # print('lidar2cam: ', lidar2cam)
            world2cam = lidar2cam.dot(world2lidar)
            Rt = world2cam.tolist()
            content.append({'K':K_topleft, 'Rt':Rt})
            content.append({'K':K_topright, 'Rt':Rt})
            content.append({'K':K_botleft, 'Rt':Rt})
            content.append({'K':K_botright, 'Rt':Rt})
   
    assert (2*len(content)==len(glob.glob(save_folder+'/*')))
    
    with open(os.path.join(save_folder, filename), 'w') as f:
        json.dump(content, f)
        
    return True



def preprocess_imgs(train_scheme, new_folder_list, org_folder_list):
    imgs_0 = sorted(glob.glob(org_folder_list[1]+'/*_0.png'))
    depths_0 = sorted(glob.glob(org_folder_list[2]+'/*_20.png'))
    assert (len(imgs_0)==5000)
    assert (len(depths_0)==5000)
    print('train_scheme: ', train_scheme)
    if train_scheme in [1,2,3,8,11]:
        imgs_train_0, depths_train_0, imgs_test_0, depths_test_0 = [],[],[],[]

        if train_scheme==1: # *full left imgs
            imgs_train_0 = imgs_0
            depths_train_0 = depths_0
            for i in range(len(imgs_train_0)):
                shutil.copy(imgs_train_0[i], os.path.join(new_folder_list[-2], str(i).zfill(3)+'_rgb.png'))
                shutil.copy(depths_train_0[i], os.path.join(new_folder_list[-2], str(i).zfill(3)+'_depth.png'))
            # TODO create_json
            res = create_json(train_scheme, True, org_folder_list[3], org_folder_list[-2], new_folder_list[-2])

            imgs_test_0 = imgs_0
            depths_test_0 = depths_0
            for i in range(len(imgs_test_0)):
                shutil.copy(imgs_test_0[i], os.path.join(new_folder_list[-1], str(i).zfill(3)+'_rgb.png'))
                shutil.copy(depths_test_0[i], os.path.join(new_folder_list[-1], str(i).zfill(3)+'_depth.png'))
            # TODO create_json
            res = create_json(train_scheme, False, org_folder_list[3], org_folder_list[-2], new_folder_list[-1])
            
        elif train_scheme==2: # *full left -> resize 480*480 visdoom
            imgs_train_0 = imgs_0
            depths_train_0 = depths_0
            for i in range(len(imgs_train_0)):
                img = cv2.imread(imgs_train_0[i])# cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depths_train_0[i], cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img,  (480, 480), interpolation = cv2.INTER_AREA)
                depth = cv2.resize(depth,  (480, 480), interpolation = cv2.INTER_AREA)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i).zfill(3)+'_rgb.png'), img)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i).zfill(3)+'_depth.png'), depth)
            # TODO create_json
            res = create_json(train_scheme, True, org_folder_list[3], org_folder_list[-2], new_folder_list[-2])

            imgs_test_0 = imgs_0
            depths_test_0 = depths_0
            for i in range(len(imgs_test_0)):
                img = cv2.imread(imgs_test_0[i])#, cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depths_test_0[i], cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img,  (480, 480), interpolation = cv2.INTER_AREA)
                depth = cv2.resize(depth,  (480, 480), interpolation = cv2.INTER_AREA)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i).zfill(3)+'_rgb.png'), img)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i).zfill(3)+'_depth.png'), depth)
            # TODO create_json
            res = create_json(train_scheme, False, org_folder_list[3], org_folder_list[-2], new_folder_list[-1])

        elif train_scheme==3: # *left every 25
            imgs_train_0 = imgs_0[0::25]
            depths_train_0 = depths_0[0::25]
            assert (len(imgs_train_0)==200)
            assert (len(depths_train_0)==200)
            for i in range(len(imgs_train_0)):
                shutil.copy(imgs_train_0[i], os.path.join(new_folder_list[-2], str(i).zfill(3)+'_rgb.png'))
                shutil.copy(depths_train_0[i], os.path.join(new_folder_list[-2], str(i).zfill(3)+'_depth.png'))
            # TODO create_json
            res = create_json(train_scheme, True, org_folder_list[3], org_folder_list[-2], new_folder_list[-2])

            imgs_test_0 = imgs_0[int(25/2)::25]
            depths_test_0 = depths_0[int(25/2)::25]
            assert (len(imgs_test_0)==len(depths_test_0))
            print('len(imgs_test_0): ', len(imgs_test_0))
            for i in range(len(imgs_test_0)):
                shutil.copy(imgs_test_0[i], os.path.join(new_folder_list[-1], str(i).zfill(3)+'_rgb.png'))
                shutil.copy(depths_test_0[i], os.path.join(new_folder_list[-1], str(i).zfill(3)+'_depth.png'))
            # TODO create_json
            res = create_json(train_scheme, False, org_folder_list[3], org_folder_list[-2], new_folder_list[-1])

        elif train_scheme==8: # *left every 25 -> resize 480*480
            imgs_train_0 = imgs_0[0::25]
            depths_train_0 = depths_0[0::25]
            assert (len(imgs_train_0)==200)
            assert (len(depths_train_0)==200)
            for i in range(len(imgs_train_0)):
                img = cv2.imread(imgs_train_0[i])#, cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depths_train_0[i], cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img,  (480, 480), interpolation = cv2.INTER_AREA)
                depth = cv2.resize(depth,  (480, 480), interpolation = cv2.INTER_AREA)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i).zfill(3)+'_rgb.png'), img)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i).zfill(3)+'_depth.png'), depth)
            # TODO create_json
            res = create_json(train_scheme, True, org_folder_list[3], org_folder_list[-2], new_folder_list[-2])

            imgs_test_0 = imgs_0[int(25/2)::25]
            depths_test_0 = depths_0[int(25/2)::25]
            assert (len(imgs_test_0)==len(depths_test_0))
            print('len(imgs_test_0): ', len(imgs_test_0))
            for i in range(len(imgs_test_0)):
                img = cv2.imread(imgs_test_0[i])#, cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depths_test_0[i], cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img,  (480, 480), interpolation = cv2.INTER_AREA)
                depth = cv2.resize(depth,  (480, 480), interpolation = cv2.INTER_AREA)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i).zfill(3)+'_rgb.png'), img)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i).zfill(3)+'_depth.png'), depth)
            # TODO create_json
            res = create_json(train_scheme, False, org_folder_list[3], org_folder_list[-2], new_folder_list[-1])

        elif train_scheme==11: # *left every 25 -> crop 480*480
            imgs_train_0 = imgs_0[0::25]
            depths_train_0 = depths_0[0::25]
            assert (len(imgs_train_0)==200)
            assert (len(depths_train_0)==200)
            img_x, img_y, crop_size = 1392, 1024, 480
            start_x = int((img_x - crop_size*2)/2)
            start_y = int((img_y - crop_size*2)/2)
            for i in range(len(imgs_train_0)):
                img = cv2.imread(imgs_train_0[i])#, cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depths_train_0[i], cv2.IMREAD_UNCHANGED)
                img_topleft = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
                img_topright = img[start_y:start_y+crop_size, start_x+crop_size:start_x+2*crop_size]
                img_botleft = img[start_y+crop_size:start_y+2*crop_size, start_x:start_x+crop_size]
                img_botright = img[start_y+crop_size:start_y+2*crop_size, start_x+crop_size:start_x+2*crop_size]
                depth_topleft = depth[start_y:start_y+crop_size, start_x:start_x+crop_size]
                depth_topright = depth[start_y:start_y+crop_size, start_x+crop_size:start_x+2*crop_size]
                depth_botleft = depth[start_y+crop_size:start_y+2*crop_size, start_x:start_x+crop_size]
                depth_botright = depth[start_y+crop_size:start_y+2*crop_size, start_x+crop_size:start_x+2*crop_size]
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8).zfill(3)+'_rgb.png'), img_topleft)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8+1).zfill(3)+'_rgb.png'), img_topright)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8+2).zfill(3)+'_rgb.png'), img_botleft)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8+3).zfill(3)+'_rgb.png'), img_botright)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8).zfill(3)+'_depth.png'), depth_topleft)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8+1).zfill(3)+'_depth.png'), depth_topright)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8+2).zfill(3)+'_depth.png'), depth_botleft)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8+3).zfill(3)+'_depth.png'), depth_botright)
            # TODO create_json
            res = create_json(train_scheme, True, org_folder_list[3], org_folder_list[-2], new_folder_list[-2])

            imgs_test_0 = imgs_0[int(25/2)::25]
            depths_test_0 = depths_0[int(25/2)::25]
            assert (len(imgs_test_0)==len(depths_test_0))
            print('len(imgs_test_0): ', len(imgs_test_0))
            for i in range(len(imgs_test_0)):
                img = cv2.imread(imgs_test_0[i])#, cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depths_test_0[i], cv2.IMREAD_UNCHANGED)
                img_topleft = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
                img_topright = img[start_y:start_y+crop_size, start_x+crop_size:start_x+2*crop_size]
                img_botleft = img[start_y+crop_size:start_y+2*crop_size, start_x:start_x+crop_size]
                img_botright = img[start_y+crop_size:start_y+2*crop_size, start_x+crop_size:start_x+2*crop_size]
                depth_topleft = depth[start_y:start_y+crop_size, start_x:start_x+crop_size]
                depth_topright = depth[start_y:start_y+crop_size, start_x+crop_size:start_x+2*crop_size]
                depth_botleft = depth[start_y+crop_size:start_y+2*crop_size, start_x:start_x+crop_size]
                depth_botright = depth[start_y+crop_size:start_y+2*crop_size, start_x+crop_size:start_x+2*crop_size]
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8).zfill(3)+'_rgb.png'), img_topleft)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8+1).zfill(3)+'_rgb.png'), img_topright)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8+2).zfill(3)+'_rgb.png'), img_botleft)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8+3).zfill(3)+'_rgb.png'), img_botright)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8).zfill(3)+'_depth.png'), depth_topleft)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8+1).zfill(3)+'_depth.png'), depth_topright)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8+2).zfill(3)+'_depth.png'), depth_botleft)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8+3).zfill(3)+'_depth.png'), depth_botright)
            # TODO create_json
            res = create_json(train_scheme, False, org_folder_list[3], org_folder_list[-2], new_folder_list[-1])
                
        return [imgs_train_0, depths_train_0, imgs_test_0, depths_test_0]
    else:
        imgs_1 = sorted(glob.glob(org_folder_list[1]+'/*_1.png'))
        depths_1 = sorted(glob.glob(org_folder_list[2]+'/*_21.png'))
        assert (len(imgs_1)==5000)
        assert (len(depths_1)==5000)
        
        imgs_train_0, depths_train_0, imgs_train_1, depths_train_1, imgs_test_0, depths_test_0, imgs_test_1, depths_test_1 = [],[],[],[],[],[],[],[]
        
        imgs_train_0 = imgs_0[0::50]
        depths_train_0 = depths_0[0::50]
        imgs_train_1 = imgs_1[0::50]
        depths_train_1 = depths_1[0::50]
        assert (len(imgs_train_0)==100)
        assert (len(depths_train_0)==100)
        assert (len(imgs_train_1)==100)
        assert (len(depths_train_1)==100)
        
        
        if train_scheme==5: # *left, right every 50
            for i in range(len(imgs_train_0)):
                shutil.copy(imgs_train_0[i], os.path.join(new_folder_list[-4], str(i).zfill(3)+'_rgb.png'))
                shutil.copy(depths_train_0[i], os.path.join(new_folder_list[-4], str(i).zfill(3)+'_depth.png'))
            # TODO create_json
            res = create_json(train_scheme, True, org_folder_list[3], org_folder_list[-2], new_folder_list[-4])

            for i in range(len(imgs_train_1)):
                shutil.copy(imgs_train_1[i], os.path.join(new_folder_list[-2], str(i).zfill(3)+'_rgb.png'))
                shutil.copy(depths_train_1[i], os.path.join(new_folder_list[-2], str(i).zfill(3)+'_depth.png'))
            # TODO create_json
            res = create_json(train_scheme, True, org_folder_list[3], org_folder_list[-1], new_folder_list[-2])
                
            imgs_test_0 = imgs_0[int(50/2)::50]
            depths_test_0 = depths_0[int(50/2)::50]
            imgs_test_1 = imgs_1[int(50/2)::50]
            depths_test_1 = depths_1[int(50/2)::50]
            assert (len(imgs_test_0)==len(depths_test_0))
            print('len(imgs_test_0): ', len(imgs_test_0))
            assert (len(imgs_test_1)==len(depths_test_1))
            print('len(imgs_test_1): ', len(imgs_test_1))
            for i in range(len(imgs_test_0)):
                shutil.copy(imgs_test_0[i], os.path.join(new_folder_list[-3], str(i).zfill(3)+'_rgb.png'))
                shutil.copy(depths_test_0[i], os.path.join(new_folder_list[-3], str(i).zfill(3)+'_depth.png'))
            # TODO create_json
            res = create_json(train_scheme, False, org_folder_list[3], org_folder_list[-2], new_folder_list[-3])

            for i in range(len(imgs_test_1)):
                shutil.copy(imgs_test_1[i], os.path.join(new_folder_list[-1], str(i).zfill(3)+'_rgb.png'))
                shutil.copy(depths_test_1[i], os.path.join(new_folder_list[-1], str(i).zfill(3)+'_depth.png'))
            # TODO create_json
            res = create_json(train_scheme, False, org_folder_list[3], org_folder_list[-1], new_folder_list[-1])
            
        elif train_scheme==10: # *left, right every 50 -> resize 480*480
            print('In s10')
            imgs_train_0 = imgs_0[1110:1410:1]
            depths_train_0 = depths_0[1110:1410:1]
            imgs_train_1 = imgs_1[1110:1410:1]
            depths_train_1 = depths_1[1110:1410:1]
            assert (len(imgs_train_0)==300)
            assert (len(depths_train_0)==300)
            assert (len(imgs_train_1)==300)
            assert (len(depths_train_1)==300)
            for i in range(len(imgs_train_0)):
                img = cv2.imread(imgs_train_0[i])#, cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depths_train_0[i], cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img,  (480, 480), interpolation = cv2.INTER_AREA)
                depth = cv2.resize(depth,  (480, 480), interpolation = cv2.INTER_AREA)
                cv2.imwrite(os.path.join(new_folder_list[-4], str(i).zfill(3)+'_rgb.png'), img)
                cv2.imwrite(os.path.join(new_folder_list[-4], str(i).zfill(3)+'_depth.png'), depth)
            # TODO create_json
            res = create_json(train_scheme, True, org_folder_list[3], org_folder_list[-2], new_folder_list[-4])
            
            for i in range(len(imgs_train_1)):
                img = cv2.imread(imgs_train_1[i])#, cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depths_train_1[i], cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img,  (480, 480), interpolation = cv2.INTER_AREA)
                depth = cv2.resize(depth,  (480, 480), interpolation = cv2.INTER_AREA)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i).zfill(3)+'_rgb.png'), img)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i).zfill(3)+'_depth.png'), depth)
            # TODO create_json
            res = create_json(train_scheme, True, org_folder_list[3], org_folder_list[-1], new_folder_list[-2])
                
            imgs_test_0 = imgs_0[1110:1410:1]
            depths_test_0 = depths_0[1110:1410:1]
            imgs_test_1 = imgs_1[1110:1410:1]
            depths_test_1 = depths_1[1110:1410:1]
            assert (len(imgs_test_0)==len(depths_test_0))
            print('len(imgs_test_0): ', len(imgs_test_0))
            assert (len(imgs_test_1)==len(depths_test_1))
            print('len(imgs_test_1): ', len(imgs_test_1))
            for i in range(len(imgs_test_0)):
                img = cv2.imread(imgs_test_0[i])#, cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depths_test_0[i], cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img,  (480, 480), interpolation = cv2.INTER_AREA)
                depth = cv2.resize(depth,  (480, 480), interpolation = cv2.INTER_AREA)
                cv2.imwrite(os.path.join(new_folder_list[-3], str(i).zfill(3)+'_rgb.png'), img)
                cv2.imwrite(os.path.join(new_folder_list[-3], str(i).zfill(3)+'_depth.png'), depth)
            # TODO create_json
            res = create_json(train_scheme, False, org_folder_list[3], org_folder_list[-2], new_folder_list[-3])
            
            for i in range(len(imgs_test_1)):
                img = cv2.imread(imgs_test_1[i])#, cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depths_test_1[i], cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img,  (480, 480), interpolation = cv2.INTER_AREA)
                depth = cv2.resize(depth,  (480, 480), interpolation = cv2.INTER_AREA)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i).zfill(3)+'_rgb.png'), img)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i).zfill(3)+'_depth.png'), depth)
            # TODO create_json
            res = create_json(train_scheme, False, org_folder_list[3], org_folder_list[-1], new_folder_list[-1])

        elif train_scheme==12: # *left, right every 50 -> crop 480*480
            img_x, img_y, crop_size = 1392, 1024, 480
            start_x = int((img_x - crop_size*2)/2)
            start_y = int((img_y - crop_size*2)/2)
            for i in range(len(imgs_train_0)):
                img = cv2.imread(imgs_train_0[i])#, cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depths_train_0[i], cv2.IMREAD_UNCHANGED)
                img_topleft = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
                img_topright = img[start_y:start_y+crop_size, start_x+crop_size:start_x+2*crop_size]
                img_botleft = img[start_y+crop_size:start_y+2*crop_size, start_x:start_x+crop_size]
                img_botright = img[start_y+crop_size:start_y+2*crop_size, start_x+crop_size:start_x+2*crop_size]
                depth_topleft = depth[start_y:start_y+crop_size, start_x:start_x+crop_size]
                depth_topright = depth[start_y:start_y+crop_size, start_x+crop_size:start_x+2*crop_size]
                depth_botleft = depth[start_y+crop_size:start_y+2*crop_size, start_x:start_x+crop_size]
                depth_botright = depth[start_y+crop_size:start_y+2*crop_size, start_x+crop_size:start_x+2*crop_size]
                cv2.imwrite(os.path.join(new_folder_list[-4], str(i*8).zfill(3)+'_rgb.png'), img_topleft)
                cv2.imwrite(os.path.join(new_folder_list[-4], str(i*8+1).zfill(3)+'_rgb.png'), img_topright)
                cv2.imwrite(os.path.join(new_folder_list[-4], str(i*8+2).zfill(3)+'_rgb.png'), img_botleft)
                cv2.imwrite(os.path.join(new_folder_list[-4], str(i*8+3).zfill(3)+'_rgb.png'), img_botright)
                cv2.imwrite(os.path.join(new_folder_list[-4], str(i*8).zfill(3)+'_depth.png'), depth_topleft)
                cv2.imwrite(os.path.join(new_folder_list[-4], str(i*8+1).zfill(3)+'_depth.png'), depth_topright)
                cv2.imwrite(os.path.join(new_folder_list[-4], str(i*8+2).zfill(3)+'_depth.png'), depth_botleft)
                cv2.imwrite(os.path.join(new_folder_list[-4], str(i*8+3).zfill(3)+'_depth.png'), depth_botright)
            # TODO create_json
            res = create_json(train_scheme, True, org_folder_list[3], org_folder_list[-2], new_folder_list[-4])

            for i in range(len(imgs_train_1)):
                img = cv2.imread(imgs_train_1[i])#, cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depths_train_1[i], cv2.IMREAD_UNCHANGED)
                img_topleft = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
                img_topright = img[start_y:start_y+crop_size, start_x+crop_size:start_x+2*crop_size]
                img_botleft = img[start_y+crop_size:start_y+2*crop_size, start_x:start_x+crop_size]
                img_botright = img[start_y+crop_size:start_y+2*crop_size, start_x+crop_size:start_x+2*crop_size]
                depth_topleft = depth[start_y:start_y+crop_size, start_x:start_x+crop_size]
                depth_topright = depth[start_y:start_y+crop_size, start_x+crop_size:start_x+2*crop_size]
                depth_botleft = depth[start_y+crop_size:start_y+2*crop_size, start_x:start_x+crop_size]
                depth_botright = depth[start_y+crop_size:start_y+2*crop_size, start_x+crop_size:start_x+2*crop_size]
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8).zfill(3)+'_rgb.png'), img_topleft)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8+1).zfill(3)+'_rgb.png'), img_topright)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8+2).zfill(3)+'_rgb.png'), img_botleft)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8+3).zfill(3)+'_rgb.png'), img_botright)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8).zfill(3)+'_depth.png'), depth_topleft)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8+1).zfill(3)+'_depth.png'), depth_topright)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8+2).zfill(3)+'_depth.png'), depth_botleft)
                cv2.imwrite(os.path.join(new_folder_list[-2], str(i*8+3).zfill(3)+'_depth.png'), depth_botright)
            # TODO create_json
            res = create_json(train_scheme, True, org_folder_list[3], org_folder_list[-1], new_folder_list[-2])
                
            imgs_test_0 = imgs_0[int(50/2)::50]
            depths_test_0 = depths_0[int(50/2)::50]
            imgs_test_1 = imgs_1[int(50/2)::50]
            depths_test_1 = depths_1[int(50/2)::50]
            assert (len(imgs_test_0)==len(depths_test_0))
            print('len(imgs_test_0): ', len(imgs_test_0))
            assert (len(imgs_test_1)==len(depths_test_1))
            print('len(imgs_test_1): ', len(imgs_test_1))
            for i in range(len(imgs_test_0)):
                img = cv2.imread(imgs_test_0[i])#, cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depths_test_0[i], cv2.IMREAD_UNCHANGED)
                img_topleft = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
                img_topright = img[start_y:start_y+crop_size, start_x+crop_size:start_x+2*crop_size]
                img_botleft = img[start_y+crop_size:start_y+2*crop_size, start_x:start_x+crop_size]
                img_botright = img[start_y+crop_size:start_y+2*crop_size, start_x+crop_size:start_x+2*crop_size]
                depth_topleft = depth[start_y:start_y+crop_size, start_x:start_x+crop_size]
                depth_topright = depth[start_y:start_y+crop_size, start_x+crop_size:start_x+2*crop_size]
                depth_botleft = depth[start_y+crop_size:start_y+2*crop_size, start_x:start_x+crop_size]
                depth_botright = depth[start_y+crop_size:start_y+2*crop_size, start_x+crop_size:start_x+2*crop_size]
                cv2.imwrite(os.path.join(new_folder_list[-3], str(i*8).zfill(3)+'_rgb.png'), img_topleft)
                cv2.imwrite(os.path.join(new_folder_list[-3], str(i*8+1).zfill(3)+'_rgb.png'), img_topright)
                cv2.imwrite(os.path.join(new_folder_list[-3], str(i*8+2).zfill(3)+'_rgb.png'), img_botleft)
                cv2.imwrite(os.path.join(new_folder_list[-3], str(i*8+3).zfill(3)+'_rgb.png'), img_botright)
                cv2.imwrite(os.path.join(new_folder_list[-3], str(i*8).zfill(3)+'_depth.png'), depth_topleft)
                cv2.imwrite(os.path.join(new_folder_list[-3], str(i*8+1).zfill(3)+'_depth.png'), depth_topright)
                cv2.imwrite(os.path.join(new_folder_list[-3], str(i*8+2).zfill(3)+'_depth.png'), depth_botleft)
                cv2.imwrite(os.path.join(new_folder_list[-3], str(i*8+3).zfill(3)+'_depth.png'), depth_botright)
            # TODO create_json
            res = create_json(train_scheme, False, org_folder_list[3], org_folder_list[-2], new_folder_list[-3])

            for i in range(len(imgs_test_1)):
                img = cv2.imread(imgs_test_1[i])#, cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depths_test_1[i], cv2.IMREAD_UNCHANGED)
                img_topleft = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
                img_topright = img[start_y:start_y+crop_size, start_x+crop_size:start_x+2*crop_size]
                img_botleft = img[start_y+crop_size:start_y+2*crop_size, start_x:start_x+crop_size]
                img_botright = img[start_y+crop_size:start_y+2*crop_size, start_x+crop_size:start_x+2*crop_size]
                depth_topleft = depth[start_y:start_y+crop_size, start_x:start_x+crop_size]
                depth_topright = depth[start_y:start_y+crop_size, start_x+crop_size:start_x+2*crop_size]
                depth_botleft = depth[start_y+crop_size:start_y+2*crop_size, start_x:start_x+crop_size]
                depth_botright = depth[start_y+crop_size:start_y+2*crop_size, start_x+crop_size:start_x+2*crop_size]
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8).zfill(3)+'_rgb.png'), img_topleft)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8+1).zfill(3)+'_rgb.png'), img_topright)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8+2).zfill(3)+'_rgb.png'), img_botleft)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8+3).zfill(3)+'_rgb.png'), img_botright)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8).zfill(3)+'_depth.png'), depth_topleft)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8+1).zfill(3)+'_depth.png'), depth_topright)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8+2).zfill(3)+'_depth.png'), depth_botleft)
                cv2.imwrite(os.path.join(new_folder_list[-1], str(i*8+3).zfill(3)+'_depth.png'), depth_botright)
            # TODO create_json
            res = create_json(train_scheme, False, org_folder_list[3], org_folder_list[-1], new_folder_list[-1])

        return [imgs_train_0, depths_train_0, imgs_train_1, depths_train_1, imgs_test_0, depths_test_0, imgs_test_1, depths_test_1]

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
            
    # *in folder 00: [095_depth.png, 095_rgb.png, ..., cameras.json]
    # org_folder = os.path.join(base_folder, 'Town02')
    # org_imgrgb_folder = os.path.join(org_folder, 'images_rgb')
    # # *in folder images_rgb: [0000_0.png, 0000_1.png, 0001_0.png, ..., 4999_1.png]
    # org_imgdepth_folder = os.path.join(org_folder, 'images_depth')
    # # *in folder images_depth: [0000_20.png, 0000_21.png, 0001_20.png, ..., 4999_21.png]
    # full_poses_lidar.txt
    # full_ts_camera.txt
    # lidar_to_cam0.txt
    # lidar_to_cam1.txt
    
    return new_folder_list


def main(train_scheme, new_base_folder, org_folder_list):
    new_folder_list = create_folder(train_scheme, new_base_folder)
    new_folder_list = preprocess_imgs(train_scheme, new_folder_list, org_folder_list)
                                               
if __name__ == '__main__':
    train_scheme = 10
    new_base_folder = '/disk2/kaho/projects_on_disk2/nerf/v1ml-gsn_s10_densedata_vKEmat/data'
    
    org_folder = '/disk2/kaho/projects_on_disk2/nerf/ml-gsn_s8/data/Town02'
    org_imgs_folder = os.path.join(org_folder, 'images_rgb')
    org_depths_folder = os.path.join(org_folder, 'images_depth')
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
    
    
    
# a=np.array([[-0.043350886553525925, 0.019534343853592873, -0.9988689422607422, 112.40690612792969], [0.00546253053471446, 0.9997985363006592, 0.019315456971526146, 1.4703837633132935], [0.9990449547767639, -0.00461900420486927, -0.04344886168837547, -36.58460998535156], [0.0, 0.0, 0.0, 1.0]], dtype='float')
# np.linalg.inv(a)
# "K": [[707.0912, 0.0, 180.0], [0.0, 707.0912, 180.0], [0.0, 0.0, 1.0]], 
# "Rt": [[-0.043350886553525925, 0.019534343853592873, -0.9988689422607422, 112.40690612792969], [0.00546253053471446, 0.9997985363006592, 0.019315456971526146, 1.4703837633132935], [0.9990449547767639, -0.00461900420486927, -0.04344886168837547, -36.58460998535156], [0.0, 0.0, 0.0, 1.0]], 
# "CamPose": [[-0.04335089, 0.005462525, 0.999045, 41.414580866264494], [0.01953435, 0.9997985, -0.00461901, -3.834868], [-0.9988689, 0.01931545, -0.04344886, 110.6618], [0.0, 0.0, 0.0, 1.0]]
