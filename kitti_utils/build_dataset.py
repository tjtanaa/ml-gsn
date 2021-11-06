import glob
import shutil
import os
import imageio as im
import numpy as np
import json

depth_path_lists = glob.glob('train/*/proj_depth/groundtruth/*/*.png')
depth_path_lists.sort()

# read camera matrix
def get_matrix(path):
    with open(path) as f:
        for line in f:
            matrix_split = line.split()
            if matrix_split[0] == 'K_01:':
                K_02 = [[float(matrix_split[1]), float(matrix_split[2]), float(matrix_split[1])],
                             [float(matrix_split[4]), float(matrix_split[5]), float(matrix_split[6])],
                             [float(matrix_split[7]), float(matrix_split[8]), float(matrix_split[9])]]
            elif matrix_split[0] == 'K_02:':
                K_03 = [[float(matrix_split[1]), float(matrix_split[2]), float(matrix_split[1])],
                             [float(matrix_split[4]), float(matrix_split[5]), float(matrix_split[6])],
                             [float(matrix_split[7]), float(matrix_split[8]), float(matrix_split[9])]]
            elif matrix_split[0] == 'R_01:':
                temp = matrix_split
                continue
            elif matrix_split[0] == 'T_01:':
                Rt_02 = [[float(temp[1]), float(temp[2]), float(temp[3]), float(matrix_split[1])],
                        [float(temp[4]), float(temp[5]), float(temp[6]), float(matrix_split[2])],
                        [float(temp[7]), float(temp[8]), float(temp[9]), float(matrix_split[3])],
                         [0.,0.,0.,1.]]
            elif matrix_split[0] == 'R_02:':
                temp = matrix_split
                continue
            elif matrix_split[0] == 'T_02:':
                Rt_03 = [[float(temp[1]), float(temp[2]), float(temp[3]), float(matrix_split[1])],
                        [float(temp[4]), float(temp[5]), float(temp[6]), float(matrix_split[2])],
                        [float(temp[7]), float(temp[8]), float(temp[9]), float(matrix_split[3])],
                         [0.,0.,0.,1.]]
    f.close()
    return {'K_02': K_02, 'K_03': K_03, 'Rt_02': Rt_02, 'Rt_03': Rt_03}

# for loop
previous_image_set = 'previous'
count = 0
matrix_json = []
for path in depth_path_lists:
    # ['train', '2011_09_30_drive_0028_sync', 'proj_depth', 'groundtruth', 'image_03', '0000003143.png']
    split = path.split('/')

    train_or_valid = split[0]
    date = split[1].split('_drive')[0] # '2011_09_30'
    date_second = split[1] # '2011_09_30_drive_0028_sync'
    image_set = split[4]
    image_name = split[5]

    # extract paths
    rgb_read = os.path.join('Kitti-Dataset', date)
    rgb_read = os.path.join(rgb_read, date_second)
    rgb_read = os.path.join(rgb_read, image_set)
    rgb_read = os.path.join(rgb_read, 'data')
    rgb_read = os.path.join(rgb_read, image_name)

    depth_read = path
    depth_write = os.path.join('Kitti_GSN_smalldataset', train_or_valid)
    depth_write = os.path.join(depth_write, date_second)
    matrix_write = depth_write
    #depth_write = os.path.join(depth_write, image_set)

    # 'Kitti-Dataset/2011_09_26/calib_cam_to_cam.txt'
    matrix_read = os.path.join('Kitti-Dataset', date)
    matrix_read = os.path.join(matrix_read, 'calib_cam_to_cam.txt')
    matrixes = get_matrix(matrix_read)

    if not os.path.exists(depth_write):
        os.makedirs(depth_write)
    rgb_write = depth_write

    if image_set == previous_image_set:
        count += 1
    else:
        # write json
        if previous_image_set == 'image_02':
            print(len(matrix_json))
            print(count)
            #matrix_write = os.path.join(matrix_write, previous_image_set)
            matrix_write = os.path.join(matrix_write, 'cameras.json')
            print(matrix_write)
            with open(matrix_write, 'w') as f:
                json.dump(matrix_json, f)

        matrix_json = []
        count = 0

    if image_set == 'image_02':
        matrix_json.append({"K": matrixes['K_02'], "Rt": matrixes['Rt_02']})
    elif image_set == 'image_03':
        matrix_json.append({"K": matrixes['K_03'], "Rt": matrixes['Rt_03']})

    previous_image_set = image_set


    depth_write = os.path.join(depth_write, '{0}_depth.png'.format(str(count).zfill(4)))
    rgb_write = os.path.join(rgb_write, '{0}_rgb.png'.format(str(count).zfill(4)))



    # read and modify and write Kitti depth
    '''
    if previous_image_set == 'image_02':
        depth_img = im.imread(depth_read)
        naive_mean_depth_img = np.zeros(depth_img.shape)
        naive_mean_depth_img = naive_mean_depth_img + np.mean(depth_img)
        im.imwrite(depth_write, naive_mean_depth_img)


        # write Kitti rgb
        shutil.copy(rgb_read, rgb_write)
    '''




