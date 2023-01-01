import sys
sys.path.append('../')
from S3DIS_dataset2 import S3DIS, ActiveLearningSampler
from torch.utils.data import DataLoader
import os
import pandas as pd
import torch
import helper_ply
import numpy as np

plane_path = '/data/wty/S3DIS/plane_result_5000/'

dataset = S3DIS(5)
train_sampler = ActiveLearningSampler(dataset, batch_size=6, split='training')
test_sampler = ActiveLearningSampler(dataset, batch_size=6, split='validation')
train_dataloder = DataLoader(train_sampler, batch_size=6, collate_fn=train_sampler.collate_fn)
test_dataloder = DataLoader(test_sampler, batch_size=6, collate_fn=test_sampler.collate_fn)

def get_nearby_plane_points(clouds, planes):
    planes_length = len(planes)
    room_result = {}
    idx = torch.arange(len(clouds))
    # every plane
    for i in range(len(planes)):
        plane_args = planes.iloc[i]
        # export name
        file_name = plane_args[0].split('.')[0]
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
        dist_mask = dist < 0.1
        # dist_mask_2 = dist < 2
        mask = dist_mask[:, 0] * range_mask

        idx_temp = idx[mask]
        if len(idx_temp) != 0:

            points = clouds[mask].numpy()
            colors = np.ones_like(points) * [255, 0, 0]
            field_names = ['x', 'y', 'z', 'r', 'g', 'b']
            # field_names = ['x', 'y', 'z']
            helper_ply.write_ply(os.path.join(plane_path, file_name + '.ply'), [points, colors], field_names)


if __name__ == '__main__':
    for batch_idx, batch_data in enumerate(test_dataloder):
        room_list = []
        cloud_names = [dataset.input_names['validation'][i] for i in batch_data['cloud_inds'].squeeze().numpy().tolist() ]
        # no sample
        print(cloud_names)
        xyz = batch_data['xyz'][0]
        point_centers = batch_data['cloud_centers']
        # every room
        for i in range(len(cloud_names)):
            center = point_centers[i]
            room_clouds = xyz[i] + center
            room_planes = pd.read_csv(os.path.join(plane_path, cloud_names[i] + '.csv'))
            get_nearby_plane_points(room_clouds, room_planes)




