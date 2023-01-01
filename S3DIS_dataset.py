from os.path import join
import sys

# sys.path.append('/home/wty/torch_points3d/RandLA-Net/')

from helper_ply import read_ply
from helper_tool import ConfigS3DIS as cfg
from helper_tool import DataProcessing as DP
from helper_tool import Plot

import numpy as np
import time, pickle, argparse, glob, os
import torch.utils.data as torch_data
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler


class S3DIS:
    def __init__(self, test_area_idx):
        self.name = 'S3DIS'
        self.path = '/data/wty/S3DIS'
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(join(self.path, 'boundary_original_ply_0.1', '*.ply'))


        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        cfg.class_weights = DP.get_class_weights('S3DIS')
        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.val_boundarys = []
        self.validation_b_c_0 = []
        self.validation_b_c_1 = []
        self.validation_b_c_2 = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                data = read_ply(file_path)

                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                self.val_boundarys += [data['is_boundary']]
                # self.validation_b_c_0 += [data['b_c_0']]
                # self.validation_b_c_1 += [data['b_c_1']]
                # self.validation_b_c_2 += [data['b_c_2']]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def __getitem__(self, item):

        pass


class ActiveLearningSampler(IterableDataset):

    def __init__(self, dataset, batch_size=6, split='training'):
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
        for i, tree in enumerate(self.dataset.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        return self.n_samples  # not equal to the actual size of the dataset, but enable nice progress bars

    def spatially_regular_gen(self):
        # Choosing the least known point as center of a new cloud each time.

        for i in range(self.n_samples * self.batch_size):  # num_per_epoch
            # t0 = time.time()

            # Generator loop

            # Choose a random cloud
            cloud_idx = int(np.argmin(self.min_possibility[self.split]))

            # cloud_name = self.dataset.input_names[self.split][cloud_idx]

            # choose the point with the minimum of possibility as query point
            point_ind = np.argmin(self.possibility[self.split][cloud_idx])

            # Get points from tree structure
            points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)

            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)

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

            queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()
            queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
            queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
            queried_idx = torch.from_numpy(queried_idx).float()  # keep float here?
            cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()
            queried_pick_point = torch.from_numpy(pick_point).float()
            # TODO 把读平面信息放到这

            yield queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cloud_idx, queried_pick_point

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

        selected_pc, selected_colors, selected_labels, selected_idx, cloud_ind, selected_center = [], [], [], [], [], []
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
    dataset = S3DIS(1)
    datasample = ActiveLearningSampler(dataset, split='validation')
    dataloader = DataLoader(datasample, batch_size=6, collate_fn=datasample.collate_fn)
    for batch_idx, batch_data in enumerate(dataloader):
        print(batch_idx)
