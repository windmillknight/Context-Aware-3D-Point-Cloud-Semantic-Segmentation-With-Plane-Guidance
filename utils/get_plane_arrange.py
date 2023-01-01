# import indoor_s3dis_utils
import helper_ply

import os
import sys
import pandas as pd
import numpy as np
from os.path import join, exists, dirname, abspath

#读入平面点云文件 和 平面参数文件 输出 房间分类 + 平面参数 + 范围
# Area_1_hallway_4_wall_3_Area_1_hallway_4_wall_3 - Cloud_RANSAC_DETECTED_SHAPES_2020-07-29_15h01_59_491
base_path = 'D:\plane\plane_normal_15'
plane_cloud_ply_path = 'plane_cloud'
plane_cloud_path = 'plane_cloud_names.csv'
plane_args_path = 'plane_args.csv'

plane_cloud_files = pd.read_csv(os.path.join(base_path, plane_cloud_path))
plane_names = plane_cloud_files['plane_name']
plane_cloud_files = plane_cloud_files['plane_cloud']
plane_args_file = pd.read_csv(os.path.join(base_path, plane_args_path), sep=';')

BASE_DIR = dirname(abspath(__file__))
# gt_class = [x.rstrip() for x in open(join(BASE_DIR, 'meta/class_names.txt'))]
# gt_class2label = {cls: i for i, cls in enumerate(gt_class)}

# xyz_min, xyz_max, label
plane_range_array = []
plane_rooms = []
for i in range(len(plane_cloud_files)):
    plane_cloud = helper_ply.read_ply_2(os.path.join(base_path, plane_cloud_ply_path, plane_cloud_files[i]))
    plane_room = '_'.join(plane_names[i].split('_')[:4])
    plane_rooms.append(plane_room)
    # label = np.ones(1) * gt_class2label[plane_cloud_files[i].split('_')[4]]
    xyz_min = np.amin(plane_cloud[:, 0:3], axis=0)
    xyz_max = np.amax(plane_cloud[:, 0:3], axis=0)
    diff = np.argmin(xyz_max - xyz_min)
    xyz_min[diff] = 0
    xyz_max[diff] = 0
    plane_range_array.append(np.concatenate([xyz_min, xyz_max], 0))
    print(i)

plane_range_array = np.concatenate(plane_range_array, 0).reshape(len(plane_cloud_files), -1)
plane_range_df = pd.DataFrame(plane_range_array, columns= ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'])
result = pd.concat([plane_names, plane_args_file[['Cx','Cy','Cz','Nx','Ny','Nz','Height','Width']], plane_range_df], 1)
result['plane_room'] = plane_rooms
result.to_csv(os.path.join(base_path,'test_result.csv'), index=False)
print('a')

result_path = 'test_result.csv'
result_file = pd.read_csv(os.path.join(base_path, result_path))
plane_rooms = set(result_file['plane_room'].to_list())
out_path = 'test_result'

if not os.path.exists(os.path.join(base_path, out_path)):
    os.mkdir(os.path.join(base_path, out_path))

for i in plane_rooms:
    file_name = i + '.csv'
    temp = result_file[result_file['plane_room'] == i]
    temp.to_csv(os.path.join(base_path, out_path, file_name), index=False)
















