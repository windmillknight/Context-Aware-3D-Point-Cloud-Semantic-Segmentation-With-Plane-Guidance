from collections import defaultdict
from os.path import join
import sys
sys.path.append('/home/wty/RandLA-Net/')

from helper_ply import read_ply
from helper_tool import ConfigScannet as cfg
from helper_tool import DataProcessing as DP
from helper_tool import Plot

import numpy as np
import time, pickle, argparse, glob, os
import torch.utils.data as torch_data
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler
import pandas as pd

class Scannet(torch_data.Dataset):
    def __init__(self, sub_grid_size=0.04, train=True):
        # TODO
        self.plane_path = '/file/wty/scannet/plane_10000_result/result/processed'

        self.train = train
        self.path = '/file/wty/scannet/'
        self.sub_grid_size = sub_grid_size
        self.train_path = 'utils/meta/scannetv2_train.txt'
        self.val_path = 'utils/meta/scannetv2_test.txt'

        self.label_to_names = {0: 'unclassified',
                               1: 'wall',
                               2: 'floor',
                               3: 'cabinet',
                               4: 'bed',
                               5: 'chair',
                               6: 'sofa',
                               7: 'table',
                               8: 'door',
                               9: 'window',
                               10: 'bookshelf',
                               11: 'picture',
                               12: 'counter',
                               14: 'desk',
                               16: 'curtain',
                               24: 'refridgerator',
                               28: 'shower curtain',
                               33: 'toilet',
                               34: 'sink',
                               36: 'bathtub',
                               39: 'otherfurniture'}

        # Initiate a bunch of variables converning class labels
        self.label_values = np.sort([key for key, values in self.label_to_names.items()])  # 0,1,2,3,...
        self.label_names = [self.label_to_names[key] for key in
                            self.label_values]  # unclassified, wall, floor, cabinet...
        # 如果label_to_names中的key值不连续，这一步可以让label的值变得连续
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}  # '0':0, '1':1, ...'14':13,'16':14
        self.name_to_idx = {name: i for i, name in enumerate(self.label_names)}
        self.ignored_labels = np.array([0])
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]

        self.all_files = glob.glob(join(self.path, 'original_test', '*.ply'))

        self.train_files = [line.rstrip() for line in open(self.train_path)]
        self.val_files = [line.rstrip() for line in open(self.val_path)]

        # TODO
        # cfg.class_weights = DP.get_class_weights('S3DIS')
        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.val_boundarys = []

        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.input_planes = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(self.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'test_input_{:.3f}_test'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if cloud_name in self.val_files:
                cloud_split = 'validation'
            elif cloud_name in self.train_files:
                cloud_split = 'training'
            else:
                continue

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))

            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))
            data = read_ply(sub_ply_file)

            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels_raw = np.zeros_like(data['red'])
            sub_labels = np.array([self.label_to_idx[label] for label in sub_labels_raw])

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)
            # Read plane

            # room_planes = pd.read_csv(os.path.join(self.plane_path, cloud_name + '.csv'))
            # 一个房间

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]
            # self.input_planes[cloud_split] += [room_planes]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

            print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices

            # Validation projection and labels
            if cloud_name in self.val_files:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                # data = read_ply(file_path)

                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                # self.val_boundarys += [data['is_boundary']]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))


    def __getitem__(self, item):

        pass


class ActiveLearningSampler(IterableDataset):

    def __init__(self, dataset, batch_size=10, split='training'):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.possibility = {}
        self.min_possibility = {}

        if split == 'training':
            self.n_samples = cfg.train_steps
        else:
            self.n_samples = cfg.val_steps

        # Random initialisation for weights
        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.dataset.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        return self.n_samples  # not equal to the actual size of the dataset, but enable nice progress bars


    def spatially_regular_gen(self):
        # Choosing the least known point as center of a new cloud each time.
        test = defaultdict(int)
        for i in range(self.n_samples * self.batch_size):  # num_per_epoch
            # t0 = time.time()

            # Generator loop

            # Choose a random cloud
            cloud_idx = int(np.argmin(self.min_possibility[self.split]))

            cloud_name = self.dataset.input_names[self.split][cloud_idx]
            print(cloud_name)
            # choose the point with the minimum of possibility as query point
            point_ind = np.argmin(self.possibility[self.split][cloud_idx])
            test[point_ind] += 1
            print(f'{point_ind} : {test[point_ind]}')
            # Get points from tree structure
            points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)

            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)

            # Planes of input room
            # planes = self.dataset.input_planes[self.split][cloud_idx]

            # Add noise to the center point
            noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
            pick_point = center_point + noise.astype(center_point.dtype)

            if len(points) < cfg.num_points:
                queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
            else:
                queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

            queried_idx = DP.shuffle_idx(queried_idx)
            # Collect points and colors
            queried_pc_xyz = points[queried_idx]

            queried_pc_xyz = queried_pc_xyz - pick_point
            queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
            queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

            dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
            delta = np.square(1 - dists / np.max(dists))
            self.possibility[self.split][cloud_idx][queried_idx] += delta
            self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

            if len(points) < cfg.num_points:
                queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                    DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)
            # Collect plane mask
            # TODO
            # plane_labels, plane_masks, plane_features = self.get_nearby_plane_points(torch.tensor(queried_pc_xyz + pick_point), pick_point,
            #                                                                           planes)
            queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()
            queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
            queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
            queried_idx = torch.from_numpy(queried_idx).float()  # keep float here?
            cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()
            queried_pick_point = torch.from_numpy(pick_point).float()

            # TODO 把读平面信息放到这

            yield queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cloud_idx, queried_pick_point
                  # plane_labels, plane_masks, plane_features

    def tf_map(self, batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
        batch_features = np.concatenate((batch_xyz, batch_features), axis=-1)

        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n)
            sub_points = batch_xyz[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
            up_i = DP.knn_search(sub_points, batch_xyz, 1)
            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_xyz = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

        return input_list

    def collate_fn(self, batch):

        selected_pc, selected_colors, selected_labels, selected_idx, cloud_ind, selected_center \
            = [], [], [], [], [], []
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_colors.append(batch[i][1])
            selected_labels.append(batch[i][2])
            selected_idx.append(batch[i][3])
            cloud_ind.append(batch[i][4])
            selected_center.append(batch[i][5])


        selected_pc = np.stack(selected_pc)
        selected_colors = np.stack(selected_colors)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)
        selected_center = np.stack(selected_center)

        # Completion plane with default


        flat_inputs = self.tf_map(selected_pc, selected_colors, selected_labels, selected_idx, cloud_ind)

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()
        inputs['cloud_centers'] = torch.from_numpy(selected_center).float()

        return inputs


if __name__ == '__main__':
    dataset = Scannet()
    datasample = ActiveLearningSampler(dataset, split='validation')
    dataloader = DataLoader(datasample, batch_size=6, collate_fn=datasample.collate_fn)
    for batch_idx, batch_data in enumerate(dataloader):
        print(batch_idx)
