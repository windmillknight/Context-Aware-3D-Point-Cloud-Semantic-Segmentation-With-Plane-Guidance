import numpy as np
import glob, os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import read_ply
import torch
import pandas as pd
from tqdm import tqdm


def get_nearby_plane_points(clouds, planes, label):
    label_result = []
    idx = torch.arange(len(clouds))
    # every plane
    for i in range(len(planes)):
        plane_args = planes.iloc[i]
        # export name
        center = torch.tensor(plane_args[1:4])
        normal = torch.tensor(plane_args[4:7])
        xyz_min = torch.tensor(plane_args[9:12])
        xyz_max = torch.tensor(plane_args[12:15])

        arange = torch.where(xyz_max != 0)[0]
        range_mask = torch.ge(clouds[:, arange[0]], xyz_min[arange[0]]) * torch.lt(clouds[:, arange[0]],
                                                                                 xyz_max[arange[0]]) \
                     * torch.ge(clouds[:, arange[1]], xyz_min[arange[1]]) * torch.lt(clouds[:, arange[1]],
                                                                                   xyz_max[arange[1]])

        dist = torch.abs(torch.mm((clouds - center), normal.view(3, 1)))
        dist_mask = dist < 0.02
        mask = dist_mask[:, 0] * range_mask

        idx_temp = idx[mask]
        if len(idx_temp) != 0:

            labels = np.argmax(np.bincount(label[mask].numpy()))
            label_result.append(labels)
        else:
            print(i)

    return label_result


if __name__ == '__main__':
    original_data_dir = '/data/scannet/test_input_0.040_train'
    plane_path = '/data/scannet/plane_10000_result/result'
    out_path = '/data/scannet/plane_10000_result/result/processed/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    data_path = glob.glob(os.path.join(original_data_dir, '*.ply'))
    data_path = np.sort(data_path)
    print(data_path)

    for file_name in tqdm(data_path):
        original_data = read_ply(os.path.join(original_data_dir, file_name.split('/')[-1][:-4] + '.ply'))
        # ori_labels = original_data['class']
        labels = torch.tensor(original_data['class'])
        points = torch.tensor(np.vstack((original_data['x'], original_data['y'], original_data['z'])).T)
        if os.path.exists(os.path.join(plane_path, file_name.split('/')[-1][:-4] + '.csv')):
            room_planes = pd.read_csv(os.path.join(plane_path, file_name.split('/')[-1][:-4] + '.csv'))
        else:
            print(file_name)
            continue
        label_result = get_nearby_plane_points(points, room_planes, labels)
        room_planes['label'] = label_result
        order = ['plane_name','Cx','Cy','Cz','Nx','Ny','Nz','Height','Width','xmin','ymin','zmin','xmax','ymax','zmax','label','plane_room']
        room_planes = room_planes[order]
        room_planes.to_csv(os.path.join(out_path, file_name.split('/')[-1][:-4] + '.csv'),index=False)