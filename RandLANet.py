import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_utils as pt_utils
from helper_tool import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix
from pytorch_utils import ResnetPointnet
import pandas as pd
import os


class Network(nn.Module):

    def __init__(self, config, dataset_names):
        super().__init__()

        self.config = config

        self.dataset_names = dataset_names

        self.class_weights = DP.get_class_weights('S3DIS')  #

        self.fc0 = pt_utils.Conv1d(6, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j < 4:
                d_in = d_out + 2 * self.config.d_out[-j - 2]
                d_out = 2 * self.config.d_out[-j - 2]
            else:
                d_in = 4 * self.config.d_out[-5]
                d_out = 2 * self.config.d_out[-5]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1, 1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        if self.config.plane_refine:
            self.plane_refine = Plane_refine_block(32, config)
            # TODO 76
            self.fc3 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        if self.config.plane_refine_2:
            self.plane_refine = Plane_refine_block_2(32, config)
            self.fc3 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        if self.config.plane_refine_3:
            self.plane_refine = Plane_refine_block_3(32, config)
            self.fc3 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        if self.config.plane_refine_4:
            self.plane_refine = Plane_refine_block_4(32, config)
            self.fc3 = pt_utils.Conv2d(74, 32, kernel_size=(1, 1), bn=True)
        if self.config.plane_refine_5:
            self.plane_refine = Plane_refine_block_5(32, config)
            self.fc3 = pt_utils.Conv2d(76, 32, kernel_size=(1, 1), bn=True)
        if self.config.plane_refine_6:
            self.decoder_2_blocks = nn.ModuleList()
            for j in range(self.config.num_layers):
                if j < 4:
                    d_in = d_out + 2 * self.config.d_out[-j - 2]
                    d_out = 2 * self.config.d_out[-j - 2]
                else:
                    d_in = 4 * self.config.d_out[-5]
                    d_out = 2 * self.config.d_out[-5]
                self.decoder_2_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))

            self.plane_refine = Plane_refine_block_6(d_out, config)
            self.fc3 = pt_utils.Conv2d(74, 32, kernel_size=(1, 1), bn=True)
        if self.config.plane_refine_7:
            self.decoder_2_blocks = nn.ModuleList()
            for j in range(self.config.num_layers):
                if j < 4:
                    d_in = d_out + 2 * self.config.d_out[-j - 2]
                    d_out = 2 * self.config.d_out[-j - 2]
                else:
                    d_in = 4 * self.config.d_out[-5]
                    d_out = 2 * self.config.d_out[-5]
                self.decoder_2_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))
            self.plane_refine = Plane_refine_block_7(d_out, config)
            self.fc3 = pt_utils.Conv2d(76, 32, kernel_size=(1, 1), bn=True)

        self.dropout = nn.Dropout(0.5)
        self.fc4 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1, 1), bn=False, activation=None)

    def forward(self, end_points, split):

        features = end_points['features']  # Batch*channel*npoints
        features = self.fc0(features)
        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        ############################ Upsample ##########################
        f_upsample_list = []
        for i in range(3):
            feature = f_encoder_list[i]
            if i == 0:
                f_upsample_list.append(feature)
            else:
                for l in range(i, -1, -1):
                    if l == 0:
                        f_upsample_list.append(feature)
                    else:
                        feature = self.nearest_interpolation(feature, end_points['interp_idx'][l-1])

        ############################ Upsample ##########################

        features = self.decoder_0(f_encoder_list[-1])
        features_to_geo = features
        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################
        features2 = self.fc1(features)  # 64
        features = self.fc2(features2)  # 32

        if self.config.plane_refine:
            dataset_cloud_names = np.array(self.dataset_names[split])
            batch_cloud_names = dataset_cloud_names[end_points['cloud_inds'][:, 0].cpu().numpy()]
            result_list = self.plane_refine(features, batch_cloud_names, end_points['cloud_centers'])

            plane_features = torch.ones([features.shape[0], 32, features.shape[2], features.shape[3]]).cuda()
            for k in range(len(result_list)):
                result = result_list[k]
                # every plane
                for l in range(len(result['on_plane_idx_list'])):
                    if len(result['on_plane_idx_list'][l]) != 0:
                        on_plane_idx = result['on_plane_idx_list'][l]
                        on_plane_feature = result['on_plane_feature_list'][l]
                        plane_features[k, :, on_plane_idx, :] = on_plane_feature

                    # FIXME if no off plane points
                    if len(result['off_plane_idx_list'][l]) != 0:
                        off_plane_idx = result['off_plane_idx_list'][l]
                        off_plane_feature = result['off_plane_feature_list'][l]
                        plane_features[k, :, off_plane_idx, :] = off_plane_feature

            features = torch.cat([features, plane_features], dim=1)

            features = self.fc3(features)
            end_points['room_results'] = result_list
        if self.config.plane_refine_2:
            dataset_cloud_names = np.array(self.dataset_names[split])
            batch_cloud_names = dataset_cloud_names[end_points['cloud_inds'][:, 0].cpu().numpy()]
            result_list = self.plane_refine(features, end_points['xyz'][0], batch_cloud_names, end_points['cloud_centers'])

            plane_features = torch.ones_like(features).float()
            for k in range(len(result_list)):
                result = result_list[k]
                # every plane
                for l in range(len(result['on_plane_idx_list'])):
                    if len(result['on_plane_idx_list'][l]) != 0:
                        on_plane_idx = result['on_plane_idx_list'][l]
                        on_plane_feature = result['on_plane_feature_list'][l]
                        plane_features[k, :, on_plane_idx, :] = on_plane_feature

            features = torch.cat([features, plane_features], dim=1)
            features = self.fc3(features)
        if self.config.plane_refine_3:
            dataset_cloud_names = np.array(self.dataset_names[split])
            batch_cloud_names = dataset_cloud_names[end_points['cloud_inds'][:, 0].cpu().numpy()]
            result_list = self.plane_refine(features, f_encoder_list[0], end_points['xyz'][0], end_points['labels'], batch_cloud_names,
                                            end_points['cloud_centers'])

            plane_features = torch.ones_like(features)
            for k in range(len(result_list)):
                result = result_list[k]
                # every plane
                for l in range(len(result['on_plane_idx_list'])):
                    if len(result['on_plane_idx_list'][l]) != 0:
                        on_plane_idx = result['on_plane_idx_list'][l]
                        on_plane_feature = result['on_plane_feature_list'][l]
                        plane_features[k, :, on_plane_idx, :] = on_plane_feature

                    if len(result['off_plane_idx_list'][l]) != 0:
                        off_plane_idx = result['off_plane_idx_list'][l]
                        off_plane_feature = result['off_plane_feature_list'][l]
                        plane_features[k, :, off_plane_idx, :] = off_plane_feature

            features = torch.cat([features, plane_features], dim=1)
            features = self.fc3(features)
        if self.config.plane_refine_4:
            dataset_cloud_names = np.array(self.dataset_names[split])
            batch_cloud_names = dataset_cloud_names[end_points['cloud_inds'][:, 0].cpu().numpy()]
            result_list = self.plane_refine(features, f_encoder_list[0], end_points['xyz'][0], batch_cloud_names,
                                            end_points['cloud_centers'])

            plane_features = torch.ones([features.shape[0], 42, features.shape[2], features.shape[3]]).cuda()
            for k in range(len(result_list)):
                result = result_list[k]
                # every plane
                for l in range(len(result['on_plane_idx_list'])):
                    if len(result['on_plane_idx_list'][l]) != 0:
                        on_plane_idx = result['on_plane_idx_list'][l]
                        on_plane_feature = result['on_plane_feature_list'][l]
                        plane_features[k, :, on_plane_idx, :] = on_plane_feature

                    # FIXME if no off plane points
                    if len(result['off_plane_idx_list'][l]) != 0:
                        off_plane_idx = result['off_plane_idx_list'][l]
                        off_plane_feature = result['off_plane_feature_list'][l]
                        plane_features[k, :, off_plane_idx, :] = off_plane_feature

            features = torch.cat([features, plane_features], dim=1)
            features = self.fc3(features)
            end_points['room_results'] = result_list
        if self.config.plane_refine_5:
            dataset_cloud_names = np.array(self.dataset_names[split])
            batch_cloud_names = dataset_cloud_names[end_points['cloud_inds'][:, 0].cpu().numpy()]

            result_list = self.plane_refine(features, f_encoder_list[0], end_points['xyz'][0], batch_cloud_names,
                                            end_points['cloud_centers'])

            # result_list = self.plane_refine(features, f_upsample_list, end_points['xyz'][0], batch_cloud_names,
            #                                 end_points['cloud_centers'])

            # TODO 32+12
            plane_features = torch.ones([features.shape[0], 44, features.shape[2], features.shape[3]]).cuda()
            for k in range(len(result_list)):
                result = result_list[k]
                # every plane
                for l in range(len(result['on_plane_idx_list'])):
                    if len(result['on_plane_idx_list'][l]) != 0:
                        on_plane_idx = result['on_plane_idx_list'][l]
                        on_plane_feature = result['on_plane_feature_list'][l]
                        plane_features[k, :, on_plane_idx, :] = on_plane_feature

                    # FIXME if no off plane points
                    if len(result['off_plane_idx_list'][l]) != 0:
                        off_plane_idx = result['off_plane_idx_list'][l]
                        off_plane_feature = result['off_plane_feature_list'][l]
                        plane_features[k, :, off_plane_idx, :] = off_plane_feature

            features = torch.cat([features, plane_features], dim=1)

            features = self.fc3(features)
            end_points['room_results'] = result_list
        if self.config.plane_refine_6:
            # ###########################Decoder 2############################
            f_decoder_list_2 = []
            for j in range(self.config.num_layers):
                f_interp_i = self.nearest_interpolation(features_to_geo, end_points['interp_idx'][-j - 1])
                f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

                features_to_geo = f_decoder_i
                f_decoder_list_2.append(f_decoder_i)
            # ###########################Decoder 2############################

            dataset_cloud_names = np.array(self.dataset_names[split])
            batch_cloud_names = dataset_cloud_names[end_points['cloud_inds'][:, 0].cpu().numpy()]
            result_list = self.plane_refine(features_to_geo, features, end_points['xyz'][0], batch_cloud_names,
                                            end_points['cloud_centers'])

            plane_features = torch.ones([features.shape[0], 42, features.shape[2], features.shape[3]]).cuda()
            for k in range(len(result_list)):
                result = result_list[k]
                # every plane
                for l in range(len(result['on_plane_idx_list'])):
                    if len(result['on_plane_idx_list'][l]) != 0:
                        on_plane_idx = result['on_plane_idx_list'][l]
                        on_plane_feature = result['on_plane_feature_list'][l]
                        plane_features[k, :, on_plane_idx, :] = on_plane_feature

                    # FIXME if no off plane points
                    if len(result['off_plane_idx_list'][l]) != 0:
                        off_plane_idx = result['off_plane_idx_list'][l]
                        off_plane_feature = result['off_plane_feature_list'][l]
                        plane_features[k, :, off_plane_idx, :] = off_plane_feature

            features = torch.cat([features, plane_features], dim=1)
            features = self.fc3(features)
            end_points['room_results'] = result_list
        if self.config.plane_refine_7:
            # ###########################Decoder 2############################
            f_decoder_list_2 = []
            for j in range(self.config.num_layers):
                f_interp_i = self.nearest_interpolation(features_to_geo, end_points['interp_idx'][-j - 1])
                f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

                features_to_geo = f_decoder_i
                f_decoder_list_2.append(f_decoder_i)
            # ###########################Decoder 2############################

            dataset_cloud_names = np.array(self.dataset_names[split])
            batch_cloud_names = dataset_cloud_names[end_points['cloud_inds'][:, 0].cpu().numpy()]
            result_list = self.plane_refine(features_to_geo, features, end_points['xyz'][0], batch_cloud_names,
                                            end_points['cloud_centers'])

            plane_features = torch.ones([features.shape[0], 44, features.shape[2], features.shape[3]]).cuda()
            for k in range(len(result_list)):
                result = result_list[k]
                # every plane
                for l in range(len(result['on_plane_idx_list'])):
                    if len(result['on_plane_idx_list'][l]) != 0:
                        on_plane_idx = result['on_plane_idx_list'][l]
                        on_plane_feature = result['on_plane_feature_list'][l]
                        plane_features[k, :, on_plane_idx, :] = on_plane_feature

                    # FIXME if no off plane points
                    if len(result['off_plane_idx_list'][l]) != 0:
                        off_plane_idx = result['off_plane_idx_list'][l]
                        off_plane_feature = result['off_plane_feature_list'][l]
                        plane_features[k, :, off_plane_idx, :] = off_plane_feature

            features = torch.cat([features, plane_features], dim=1)
            features = self.fc3(features)
            end_points['room_results'] = result_list
        features = self.dropout(features)
        features = self.fc4(features)
        f_out = features.squeeze(3)

        end_points['logits'] = f_out

        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features


class Plane_refine_block(nn.Module):
    def __init__(self, d_in, config):
        super().__init__()
        self.config = config
        self.idx_list = []
        self.feature_list = []
        self.label_list = []
        # self.res_pointNet = ResnetPointnet(1, d_in, d_in)
        self.fc1 = pt_utils.Conv2d(d_in + d_in, d_in, kernel_size=(1, 1), bn=True)
        # 多来几层？
        self.fc2 = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), bn=True)
        self.fc3 = pt_utils.Conv2d(d_in, 1, kernel_size=(1, 1), bn=False)
        # self.att_pooling = Att_pooling(d_in, d_in)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, feature, feature_geo, xyz, cloud_names, centers):
        room_list = []

        for i in range(len(cloud_names)):
            room_clouds = xyz[i] + centers[i]
            room_feature = torch.cat([feature[i].unsqueeze(0), feature_geo[i].unsqueeze(0)], dim=1)
            room_planes = pd.read_csv(os.path.join(self.config.plane_path, cloud_names[i] + '.csv'))
            # 一个房间
            room_result = self.get_nearby_plane_points(room_clouds, room_feature, room_planes)

            room_list.append(room_result)

        return room_list

    def get_nearby_plane_points(self, clouds, feature, planes):
        planes_length = len(planes)
        room_result = {}
        room_result['label_list'] = []
        room_result['on_plane_idx_list'] = []
        room_result['off_plane_idx_list'] = []
        room_result['on_plane_feature_list'] = []
        room_result['off_plane_feature_list'] = []
        room_result['plane_result_list'] = []
        room_result['mask'] = []
        idx = torch.arange(len(clouds))

        for i in range(planes_length):
            plane_args = planes.iloc[i]
            center = torch.tensor(plane_args[1:4]).cuda()
            normal = torch.tensor(plane_args[4:7]).cuda()
            xyz_min = torch.tensor(plane_args[9:12]).cuda()
            xyz_max = torch.tensor(plane_args[12:15]).cuda()
            label = torch.tensor(plane_args[15]).cuda()

            arange = torch.where(xyz_max != 0)[0]
            range_mask = torch.ge(clouds[:, arange[0]], xyz_min[arange[0]]) * torch.lt(clouds[:, arange[0]],
                                                                                     xyz_max[arange[0]]) \
                         * torch.ge(clouds[:, arange[1]], xyz_min[arange[1]]) * torch.lt(clouds[:, arange[1]],
                                                                                       xyz_max[arange[1]])

            # ori_plane_feature = torch.cat([center, normal, xyz_min, xyz_max]).view(1,12,1,1).float()

            dist = torch.abs(torch.mm((clouds - center), normal.view(3, 1)))
            dist_mask = dist < 0.1
            mask = dist_mask[:, 0] * range_mask
            idx_temp = idx[mask]

            #  FIXME choose maks numbers
            if len(idx_temp) > 1:
                plane_feature = feature[:, :, mask]
                room_result['label_list'].append(label)
                room_result['mask'].append(mask)
                #
                plane_feature = self.fc1(plane_feature)
                plane_feature = self.fc2(plane_feature)
                plane_result = self.fc3(plane_feature)
                # plane_feature, plane_result = self.res_pointNet(plane_feature)

                # with torch.no_grad():
                room_result['plane_result_list'].append(plane_result.squeeze())
                mask_1 = self.sigmoid(plane_result) > 0.5
                mask_2 = self.sigmoid(plane_result) <= 0.5

                if len(torch.nonzero(mask_1)) != 0:
                    idx_on_plane = idx_temp[mask_1.squeeze()]
                    room_result['on_plane_idx_list'].append(idx_on_plane)
                    on_plane_feature = plane_feature[:, :, mask_1.squeeze()]
                    on_plane_feature = torch.nn.functional.max_pool2d(on_plane_feature,
                                                                      kernel_size=[on_plane_feature.shape[2], 1])
                    # on_plane_feature = torch.cat([on_plane_feature, ori_plane_feature], dim=1)
                    room_result['on_plane_feature_list'].append(on_plane_feature)

                else:
                    room_result['on_plane_idx_list'].append([])
                    room_result['on_plane_feature_list'].append([])

                if len(torch.nonzero(mask_2)) != 0:
                    idx_off_plane = idx_temp[mask_2.squeeze()]
                    room_result['off_plane_idx_list'].append(idx_off_plane)
                    off_plane_feature = plane_feature[:, :, mask_2.squeeze()]
                    off_plane_feature = torch.nn.functional.max_pool2d(off_plane_feature,
                                                                       kernel_size=[off_plane_feature.shape[2], 1])
                    # off_plane_feature = torch.cat([off_plane_feature, ori_plane_feature], dim=1)
                    room_result['off_plane_feature_list'].append(off_plane_feature)
                else:
                    room_result['off_plane_idx_list'].append([])
                    room_result['off_plane_feature_list'].append([])

        return room_result

class Plane_refine_block_2(nn.Module):
    def __init__(self, d_in, config):
        super().__init__()
        self.config = config
        self.idx_list = []
        self.feature_list = []
        self.label_list = []

        self.fc1 = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), bn=True)
        # 多来几层？
        self.fc2 = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), bn=True)
        # self.fc3 = pt_utils.Conv2d(d_in, 1, kernel_size=(1, 1), bn=False)

    def forward(self, feature, xyz, cloud_names, centers):
        room_list = []
        for i in range(len(cloud_names)):
            room_clouds = xyz[i] + centers[i]
            room_feature = feature[i].unsqueeze(0)
            room_planes = pd.read_csv(os.path.join(self.config.plane_path, cloud_names[i] + '.csv'))
            # 一个房间
            room_result = self.get_nearby_plane_points(room_clouds, room_feature, room_planes)

            room_list.append(room_result)

        return room_list

    def get_nearby_plane_points(self, clouds, feature, planes):
        planes_length = len(planes)
        room_result = {}
        room_result['label_list'] = []
        room_result['on_plane_idx_list'] = []
        room_result['on_plane_feature_list'] = []
        room_result['mask'] = []
        idx = torch.arange(len(clouds))

        for i in range(planes_length):
            plane_args = planes.iloc[i]
            center = torch.tensor(plane_args[1:4]).cuda()
            normal = torch.tensor(plane_args[4:7]).cuda()
            xyz_min = torch.tensor(plane_args[9:12]).cuda()
            xyz_max = torch.tensor(plane_args[12:15]).cuda()
            # label = torch.tensor(plane_args[15]).cuda()

            arange = torch.where(xyz_max != 0)[0]
            range_mask = torch.ge(clouds[:, arange[0]], xyz_min[arange[0]]) * torch.lt(clouds[:, arange[0]],
                                                                                     xyz_max[arange[0]]) \
                         * torch.ge(clouds[:, arange[1]], xyz_min[arange[1]]) * torch.lt(clouds[:, arange[1]],
                                                                                       xyz_max[arange[1]])

            dist = torch.abs(torch.mm((clouds - center), normal.view(3, 1)))
            dist_mask = dist < 0.05
            mask = dist_mask[:, 0] * range_mask
            idx_temp = idx[mask]

            #  FIXME choose maks numbers
            if len(idx_temp) > 1:
                plane_feature = feature[:, :, mask]
                room_result['mask'].append(mask)
                # FIXME ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 32, 1, 1])
                plane_feature = self.fc1(plane_feature)
                plane_feature = self.fc2(plane_feature)
                room_result['on_plane_idx_list'].append(idx_temp)
                on_plane_feature = plane_feature
                on_plane_feature = torch.nn.functional.max_pool2d(on_plane_feature,
                                                                  kernel_size=[on_plane_feature.shape[2], 1])
                room_result['on_plane_feature_list'].append(on_plane_feature)

        return room_result

class Plane_refine_block_3(nn.Module):
    def __init__(self, d_in, config):
        super().__init__()
        self.config = config
        self.idx_list = []
        self.feature_list = []
        self.label_list = []

        self.fc1 = pt_utils.Conv2d(d_in , d_in, kernel_size=(1, 1), bn=True)
        # 多来几层？
        self.fc2 = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), bn=True)
        # self.fc3 = pt_utils.Conv2d(d_in, 1, kernel_size=(1, 1), bn=False)
        self.att_pool = Att_pooling(d_in, d_in, bn=False)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, feature, feature_geo, xyz, point_labels, cloud_names, centers):
        room_list = []
        for i in range(len(cloud_names)):
            room_clouds = xyz[i] + centers[i]
            # room_feature = torch.cat([feature[i].unsqueeze(0), feature_geo[i].unsqueeze(0)], dim=1)
            room_feature = feature[i].unsqueeze(0)
            point_label = point_labels[i]
            room_planes = pd.read_csv(os.path.join(self.config.plane_path, cloud_names[i] + '.csv'))
            # 一个房间
            room_result = self.get_nearby_plane_points(room_clouds, room_feature, room_planes, point_label)

            room_list.append(room_result)

        return room_list

    def get_nearby_plane_points(self, clouds, feature, planes, point_labels):
        planes_length = len(planes)
        room_result = {}
        room_result['on_plane_idx_list'] = []
        room_result['off_plane_idx_list'] = []
        room_result['on_plane_feature_list'] = []
        room_result['off_plane_feature_list'] = []
        room_result['plane_result_list'] = []
        room_result['mask'] = []
        idx = torch.arange(len(clouds))

        for i in range(planes_length):
            plane_args = planes.iloc[i]
            center = torch.tensor(plane_args[1:4]).cuda()
            normal = torch.tensor(plane_args[4:7]).cuda()
            xyz_min = torch.tensor(plane_args[9:12]).cuda()
            xyz_max = torch.tensor(plane_args[12:15]).cuda()
            plane_label = torch.tensor(plane_args[15]).cuda()

            arange = torch.where(xyz_max != 0)[0]
            range_mask = torch.ge(clouds[:, arange[0]], xyz_min[arange[0]]) * torch.lt(clouds[:, arange[0]],
                                                                                     xyz_max[arange[0]]) \
                         * torch.ge(clouds[:, arange[1]], xyz_min[arange[1]]) * torch.lt(clouds[:, arange[1]],
                                                                                       xyz_max[arange[1]])

            dist = torch.abs(torch.mm((clouds - center), normal.view(3, 1)))
            dist_mask = dist < 0.1
            mask = dist_mask[:, 0] * range_mask
            idx_temp = idx[mask]

            if len(idx_temp) > 1:
                plane_feature = feature[:, :, mask]
                # FIXME 取出平面点对应的label
                point_label = point_labels[mask]
                room_result['mask'].append(mask)
                plane_feature = self.fc1(plane_feature)
                plane_feature = self.fc2(plane_feature)

                mask_1 = point_label == plane_label
                mask_2 = point_label != plane_label

                if len(torch.nonzero(mask_1)) != 0:
                    idx_on_plane = idx_temp[mask_1.squeeze()]
                    room_result['on_plane_idx_list'].append(idx_on_plane)
                    on_plane_feature = plane_feature[:, :, mask_1.squeeze()]
                    # on_plane_feature = torch.nn.functional.max_pool2d(on_plane_feature,
                    #                                                   kernel_size=[on_plane_feature.shape[2], 1])
                    on_plane_feature = self.att_pool(on_plane_feature, dim=2)
                    room_result['on_plane_feature_list'].append(on_plane_feature)

                else:
                    room_result['on_plane_idx_list'].append([])
                    room_result['on_plane_feature_list'].append([])

                if len(torch.nonzero(mask_2)) != 0:
                    idx_off_plane = idx_temp[mask_2.squeeze()]
                    room_result['off_plane_idx_list'].append(idx_off_plane)
                    off_plane_feature = plane_feature[:, :, mask_2.squeeze()]
                    # off_plane_feature = torch.nn.functional.max_pool2d(off_plane_feature,
                    #                                                    kernel_size=[off_plane_feature.shape[2], 1])
                    off_plane_feature = self.att_pool(off_plane_feature, dim=2)
                    room_result['off_plane_feature_list'].append(off_plane_feature)
                else:
                    room_result['off_plane_idx_list'].append([])
                    room_result['off_plane_feature_list'].append([])

        return room_result

class Plane_refine_block_4(nn.Module):
    def __init__(self, d_in, config):
        super().__init__()
        self.config = config
        self.idx_list = []
        self.feature_list = []
        self.label_list = []

        # self.res_pointnet = ResnetPointnet(4, d_in, d_in, True)
        self.fc1 = pt_utils.Conv2d(d_in*2, d_in*2, kernel_size=(1, 1), bn=True)
        # 多来几层？
        self.fc2 = pt_utils.Conv2d(d_in*2, d_in, kernel_size=(1, 1), bn=True)

        self.fc3 = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), bn=False) #
        self.fc4 = pt_utils.Conv2d(d_in, 4, kernel_size=(1, 1), bn=False)  #
        self.att_pool = Att_pooling(d_in, d_in, bn=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, feature, feature_geo, xyz, cloud_names, centers):
        room_list = []
        for i in range(len(cloud_names)):
            room_clouds = xyz[i] + centers[i]
            room_feature = torch.cat([feature[i].unsqueeze(0), feature_geo[i].unsqueeze(0)], dim=1)
            room_planes = pd.read_csv(os.path.join(self.config.plane_path, cloud_names[i] + '.csv'))
            # 一个房间
            room_result = self.get_nearby_plane_points(room_clouds, room_feature, room_planes)

            room_list.append(room_result)

        return room_list

    def get_nearby_plane_points(self, clouds, feature, planes):
        planes_length = len(planes)
        room_result = {}
        room_result['label_list'] = []
        room_result['on_plane_idx_list'] = []
        room_result['off_plane_idx_list'] = []
        room_result['on_plane_feature_list'] = []
        room_result['off_plane_feature_list'] = []
        room_result['plane_result_list'] = []
        room_result['mask'] = []
        idx = torch.arange(len(clouds))

        for i in range(planes_length):
            plane_args = planes.iloc[i]
            center = torch.tensor(plane_args[1:4]).cuda().float()
            normal = torch.tensor(plane_args[4:7]).cuda().float()
            xyz_min = torch.tensor(plane_args[9:12]).cuda().float()
            xyz_max = torch.tensor(plane_args[12:15]).cuda().float()
            label = torch.tensor(plane_args[15]).cuda()

            arange = torch.where(xyz_max != 0)[0]
            range_mask = torch.ge(clouds[:, arange[0]], xyz_min[arange[0]]) * torch.lt(clouds[:, arange[0]],
                                                                                     xyz_max[arange[0]]) \
                         * torch.ge(clouds[:, arange[1]], xyz_min[arange[1]]) * torch.lt(clouds[:, arange[1]],
                                                                                       xyz_max[arange[1]])
            D = torch.mm(-center.view(1, 3), normal.view(3, 1))
            dist = torch.abs(torch.mm(clouds, normal.view(3, 1)) + D)
            # TODO How to combine?

            dist_mask = dist < 0.1
            mask = dist_mask[:, 0] * range_mask
            idx_temp = idx[mask]

            #  FIXME choose maks numbers
            if len(idx_temp) > 1:
                plane_clouds = torch.cat([clouds[mask], torch.ones(len(clouds[mask]),1).cuda()],dim=1)
                plane_feature = feature[:, :, mask]
                room_result['label_list'].append(label)
                room_result['mask'].append(mask)

                plane_feature = self.fc1(plane_feature)
                plane_feature = self.fc2(plane_feature)

                pooled_feature, _ = plane_feature.max(dim=2, keepdim=True)
                pooled_feature = self.fc3(pooled_feature)
                plane_result = self.fc4(pooled_feature) # A,B,C,D

                # plane_feature, plane_result = self.res_pointnet(plane_feature)
                new_plane = plane_result.view(-1, 4) + torch.cat((normal, D.squeeze(0))) # ABCD

                ori_plane_feature = torch.cat([new_plane.squeeze(), xyz_min, xyz_max]).view(1, 10, 1, 1).float()
                # FIXME off plane == 0? on plane == 0
                # with torch.no_grad():
                plane_result = torch.mm(plane_clouds, new_plane.view(4, 1))
                room_result['plane_result_list'].append(plane_result.squeeze())
                mask_1 = plane_result > 0
                mask_2 = plane_result <= 0

                if len(torch.nonzero(mask_1)) != 0:
                    idx_on_plane = idx_temp[mask_1.squeeze()]
                    room_result['on_plane_idx_list'].append(idx_on_plane)
                    on_plane_feature = plane_feature[:, :, mask_1.squeeze()]
                    # on_plane_feature = torch.nn.functional.max_pool2d(on_plane_feature,
                    #                                                   kernel_size=[on_plane_feature.shape[2], 1])
                    on_plane_feature = self.att_pool(on_plane_feature, dim=2)
                    on_plane_feature = torch.cat([on_plane_feature, ori_plane_feature], dim=1)
                    room_result['on_plane_feature_list'].append(on_plane_feature)

                else:
                    room_result['on_plane_idx_list'].append([])
                    room_result['on_plane_feature_list'].append([])

                if len(torch.nonzero(mask_2)) != 0:
                    idx_off_plane = idx_temp[mask_2.squeeze()]
                    room_result['off_plane_idx_list'].append(idx_off_plane)
                    off_plane_feature = plane_feature[:, :, mask_2.squeeze()]
                    # off_plane_feature = torch.nn.functional.max_pool2d(off_plane_feature,
                    #                                                    kernel_size=[off_plane_feature.shape[2], 1])
                    off_plane_feature = self.att_pool(off_plane_feature, dim=2)
                    off_plane_feature = torch.cat([off_plane_feature, ori_plane_feature], dim=1)
                    room_result['off_plane_feature_list'].append(off_plane_feature)
                else:
                    room_result['off_plane_idx_list'].append([])
                    room_result['off_plane_feature_list'].append([])

        return room_result

class Plane_refine_block_5(nn.Module):
    def __init__(self, d_in, config):
        super().__init__()
        self.config = config
        self.idx_list = []
        self.feature_list = []
        self.label_list = []

        # self.fc0 = pt_utils.Conv2d(192 + d_in, d_in*2, kernel_size=(1, 1), bn=True)
        self.fc1 = pt_utils.Conv2d(d_in, d_in*2, kernel_size=(1, 1), bn=True)
        # 多来几层？
        self.fc2 = pt_utils.Conv2d(d_in*2, d_in, kernel_size=(1, 1), bn=True)
        self.fc3 = pt_utils.Conv2d(d_in, 1, kernel_size=(1, 1), bn=False)
        self.att_pooling = Att_pooling(d_in, d_in, bn=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, feature, feature_geo, xyz, cloud_names, centers):
        room_list = []
        # room_feature = torch.cat([feature, torch.cat(feature_geo, dim=1)], dim=1)
        # room_feature = torch.cat([feature, feature_geo], dim=1)
        for i in range(len(cloud_names)):
            room_clouds = xyz[i] + centers[i]
            room_planes = pd.read_csv(os.path.join(self.config.plane_path, cloud_names[i] + '.csv'))
            # 一个房间
            room_result = self.get_nearby_plane_points(room_clouds, feature[i].unsqueeze(0), room_planes)

            room_list.append(room_result)

        return room_list

    def get_nearby_plane_points(self, clouds, feature, planes):
        planes_length = len(planes)
        room_result = {}
        room_result['label_list'] = []
        room_result['on_plane_idx_list'] = []
        room_result['off_plane_idx_list'] = []
        room_result['on_plane_feature_list'] = []
        room_result['off_plane_feature_list'] = []
        room_result['plane_result_list'] = []
        room_result['mask'] = []
        idx = torch.arange(len(clouds))

        for i in range(planes_length):
            plane_args = planes.iloc[i]
            center = torch.tensor(plane_args[1:4]).cuda()
            normal = torch.tensor(plane_args[4:7]).cuda()
            xyz_min = torch.tensor(plane_args[9:12]).cuda()
            xyz_max = torch.tensor(plane_args[12:15]).cuda()
            label = torch.tensor(plane_args[15]).cuda()

            arange = torch.where(xyz_max != 0)[0]
            range_mask = torch.ge(clouds[:, arange[0]], xyz_min[arange[0]]) * torch.lt(clouds[:, arange[0]],
                                                                                     xyz_max[arange[0]]) \
                         * torch.ge(clouds[:, arange[1]], xyz_min[arange[1]]) * torch.lt(clouds[:, arange[1]],
                                                                                       xyz_max[arange[1]])

            ori_plane_feature = torch.cat([center, normal, xyz_min, xyz_max]).view(1,12,1,1).float()
            dist = torch.abs(torch.mm((clouds - center), normal.view(3, 1)))
            dist_mask = dist < 0.1
            mask = dist_mask[:, 0] * range_mask
            idx_temp = idx[mask]

            #  FIXME choose maks numbers
            if len(idx_temp) > 1:
                plane_feature = feature[:, :, mask]
                room_result['label_list'].append(label)
                room_result['mask'].append(mask)
                #
                # plane_feature = self.fc0(plane_feature)
                plane_feature = self.fc1(plane_feature)
                plane_feature = self.fc2(plane_feature)
                plane_result = self.fc3(plane_feature)
                # with torch.no_grad():
                room_result['plane_result_list'].append(plane_result.squeeze())
                mask_1 = self.sigmoid(plane_result) > 0.5
                mask_2 = self.sigmoid(plane_result) <= 0.5

                if len(torch.nonzero(mask_1)) != 0:
                    idx_on_plane = idx_temp[mask_1.squeeze()]
                    room_result['on_plane_idx_list'].append(idx_on_plane)
                    on_plane_feature = plane_feature[:, :, mask_1.squeeze()]
                    # on_plane_feature = torch.nn.functional.max_pool2d(on_plane_feature,
                    #                                                   kernel_size=[on_plane_feature.shape[2], 1])
                    on_plane_feature = self.att_pooling(on_plane_feature, 2)
                    on_plane_feature = torch.cat([on_plane_feature, ori_plane_feature], dim=1)
                    room_result['on_plane_feature_list'].append(on_plane_feature)

                else:
                    room_result['on_plane_idx_list'].append([])
                    room_result['on_plane_feature_list'].append([])

                if len(torch.nonzero(mask_2)) != 0:
                    idx_off_plane = idx_temp[mask_2.squeeze()]
                    room_result['off_plane_idx_list'].append(idx_off_plane)
                    off_plane_feature = plane_feature[:, :, mask_2.squeeze()]
                    # off_plane_feature = torch.nn.functional.max_pool2d(off_plane_feature,
                    #                                                    kernel_size=[off_plane_feature.shape[2], 1])
                    off_plane_feature = self.att_pooling(off_plane_feature, 2)
                    off_plane_feature = torch.cat([off_plane_feature, ori_plane_feature], dim=1)
                    room_result['off_plane_feature_list'].append(off_plane_feature)
                else:
                    room_result['off_plane_idx_list'].append([])
                    room_result['off_plane_feature_list'].append([])

        return room_result

class Plane_refine_block_6(nn.Module):
    def __init__(self, d_in, config):
        super().__init__()
        self.config = config
        self.idx_list = []
        self.feature_list = []
        self.label_list = []

        # self.res_pointnet = ResnetPointnet(4, d_in, d_in, True)
        self.fc1 = pt_utils.Conv2d(d_in*2 , d_in*2, kernel_size=(1, 1), bn=True)
        # 多来几层？
        self.fc2 = pt_utils.Conv2d(d_in*2, d_in, kernel_size=(1, 1), bn=True)

        self.fc3 = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), bn=False) #
        # TODO: droup out here?
        self.fc4 = pt_utils.Conv2d(d_in, 4, kernel_size=(1, 1), bn=False)  #

        self.att_pool = Att_pooling(d_in, d_in, bn=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, feature_geo, feature_sem, xyz, cloud_names, centers):
        room_list = []
        room_feature = torch.cat([feature_geo, feature_sem], dim=1)
        for i in range(len(cloud_names)):
            room_clouds = xyz[i] + centers[i]
            room_planes = pd.read_csv(os.path.join(self.config.plane_path, cloud_names[i] + '.csv'))
            # 一个房间
            room_result = self.get_nearby_plane_points(room_clouds, room_feature[i].unsqueeze(0), room_planes)

            room_list.append(room_result)

        return room_list

    def get_nearby_plane_points(self, clouds, feature, planes):
        planes_length = len(planes)
        room_result = {}
        room_result['label_list'] = []
        room_result['on_plane_idx_list'] = []
        room_result['off_plane_idx_list'] = []
        room_result['on_plane_feature_list'] = []
        room_result['off_plane_feature_list'] = []
        room_result['plane_result_list'] = []
        room_result['mask'] = []
        idx = torch.arange(len(clouds))

        for i in range(planes_length):
            plane_args = planes.iloc[i]
            center = torch.tensor(plane_args[1:4]).cuda().float()
            normal = torch.tensor(plane_args[4:7]).cuda().float()
            xyz_min = torch.tensor(plane_args[9:12]).cuda().float()
            xyz_max = torch.tensor(plane_args[12:15]).cuda().float()
            label = torch.tensor(plane_args[15]).cuda()

            arange = torch.where(xyz_max != 0)[0]
            range_mask = torch.ge(clouds[:, arange[0]], xyz_min[arange[0]]) * torch.lt(clouds[:, arange[0]],
                                                                                     xyz_max[arange[0]]) \
                         * torch.ge(clouds[:, arange[1]], xyz_min[arange[1]]) * torch.lt(clouds[:, arange[1]],
                                                                                       xyz_max[arange[1]])
            D = torch.mm(-center.view(1, 3), normal.view(3, 1))
            dist = torch.abs(torch.mm(clouds, normal.view(3, 1)) + D)
            # TODO How to combine?

            dist_mask = dist < 0.1
            mask = dist_mask[:, 0] * range_mask
            idx_temp = idx[mask]

            #  FIXME choose maks numbers
            if len(idx_temp) > 1:
                plane_clouds = torch.cat([clouds[mask], torch.ones(len(clouds[mask]),1).cuda()],dim=1)
                plane_feature = feature[:, :, mask]
                room_result['label_list'].append(label)
                room_result['mask'].append(mask)

                plane_feature = self.fc1(plane_feature)
                plane_feature = self.fc2(plane_feature)
                #TODO: Att_pooling here?
                pooled_feature, _ = plane_feature.max(dim=2, keepdim=True)
                pooled_feature = self.fc3(pooled_feature)
                plane_result = self.fc4(pooled_feature) # A,B,C,D

                # plane_feature, plane_result = self.res_pointnet(plane_feature)
                new_plane = plane_result.view(-1, 4) + torch.cat((normal, D.squeeze(0))) # ABCD

                # FIXME: should update this to new plane??
                ori_plane_feature = torch.cat([new_plane.squeeze(), xyz_min, xyz_max]).view(1, 10, 1, 1).float()

                # with torch.no_grad():
                plane_result = torch.mm(plane_clouds, new_plane.view(4, 1))
                room_result['plane_result_list'].append(plane_result.squeeze())
                mask_1 = plane_result > 0
                mask_2 = plane_result <= 0
                # FIXME: what happen if there is no off plane points?
                if len(torch.nonzero(mask_1)) != 0:
                    idx_on_plane = idx_temp[mask_1.squeeze()]
                    room_result['on_plane_idx_list'].append(idx_on_plane)
                    on_plane_feature = plane_feature[:, :, mask_1.squeeze()]
                    # on_plane_feature = torch.nn.functional.max_pool2d(on_plane_feature,
                    #                                                   kernel_size=[on_plane_feature.shape[2], 1])
                    on_plane_feature = self.att_pool(on_plane_feature, dim=2)
                    on_plane_feature = torch.cat([on_plane_feature, ori_plane_feature], dim=1)
                    room_result['on_plane_feature_list'].append(on_plane_feature)

                else:
                    room_result['on_plane_idx_list'].append([])
                    room_result['on_plane_feature_list'].append([])

                if len(torch.nonzero(mask_2)) != 0:
                    idx_off_plane = idx_temp[mask_2.squeeze()]
                    room_result['off_plane_idx_list'].append(idx_off_plane)
                    off_plane_feature = plane_feature[:, :, mask_2.squeeze()]
                    # off_plane_feature = torch.nn.functional.max_pool2d(off_plane_feature,
                    #                                                    kernel_size=[off_plane_feature.shape[2], 1])
                    off_plane_feature = self.att_pool(off_plane_feature, dim=2)
                    off_plane_feature = torch.cat([off_plane_feature, ori_plane_feature], dim=1)
                    room_result['off_plane_feature_list'].append(off_plane_feature)
                else:
                    room_result['off_plane_idx_list'].append([])
                    room_result['off_plane_feature_list'].append([])

        return room_result

class Plane_refine_block_7(nn.Module):
    def __init__(self, d_in, config):
        super().__init__()
        self.config = config
        self.idx_list = []
        self.feature_list = []
        self.label_list = []

        self.fc0 = pt_utils.Conv2d(d_in*2, d_in*2, kernel_size=(1, 1), bn=True)
        self.fc1 = pt_utils.Conv2d(d_in*2, d_in, kernel_size=(1, 1), bn=True)
        # 多来几层？
        self.fc2 = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), bn=True)
        self.fc3 = pt_utils.Conv2d(d_in, 1, kernel_size=(1, 1), bn=False)
        self.att_pooling = Att_pooling(d_in, d_in, bn=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, feature_geo, feature_sem, xyz, cloud_names, centers):
        room_list = []
        room_feature = torch.cat([feature_geo, feature_sem], dim=1)
        # room_feature = torch.cat([feature, feature_geo], dim=1)
        for i in range(len(cloud_names)):
            room_clouds = xyz[i] + centers[i]

            room_planes = pd.read_csv(os.path.join(self.config.plane_path, cloud_names[i] + '.csv'))
            room_result = self.get_nearby_plane_points(room_clouds, room_feature[i].unsqueeze(0), room_planes)
            room_list.append(room_result)

        return room_list

    def get_nearby_plane_points(self, clouds, feature, planes):
        planes_length = len(planes)
        room_result = {}
        room_result['label_list'] = []
        room_result['on_plane_idx_list'] = []
        room_result['off_plane_idx_list'] = []
        room_result['on_plane_feature_list'] = []
        room_result['off_plane_feature_list'] = []
        room_result['plane_result_list'] = []
        room_result['mask'] = []
        idx = torch.arange(len(clouds))

        for i in range(planes_length):
            plane_args = planes.iloc[i]
            center = torch.tensor(plane_args[1:4]).cuda()
            normal = torch.tensor(plane_args[4:7]).cuda()
            xyz_min = torch.tensor(plane_args[9:12]).cuda()
            xyz_max = torch.tensor(plane_args[12:15]).cuda()
            label = torch.tensor(plane_args[15]).cuda()

            arange = torch.where(xyz_max != 0)[0]
            range_mask = torch.ge(clouds[:, arange[0]], xyz_min[arange[0]]) * torch.lt(clouds[:, arange[0]],
                                                                                     xyz_max[arange[0]]) \
                         * torch.ge(clouds[:, arange[1]], xyz_min[arange[1]]) * torch.lt(clouds[:, arange[1]],
                                                                                       xyz_max[arange[1]])

            ori_plane_feature = torch.cat([center, normal, xyz_min, xyz_max]).view(1,12,1,1).float()

            dist = torch.abs(torch.mm((clouds - center), normal.view(3, 1)))
            dist_mask = dist < 0.1
            mask = dist_mask[:, 0] * range_mask
            idx_temp = idx[mask]

            #  FIXME choose maks numbers
            if len(idx_temp) > 1:
                plane_feature = feature[:, :, mask]
                room_result['label_list'].append(label)
                room_result['mask'].append(mask)
                #
                plane_feature = self.fc0(plane_feature)
                plane_feature = self.fc1(plane_feature)
                plane_feature = self.fc2(plane_feature)
                plane_result = self.fc3(plane_feature)
                # plane_feature, plane_result = self.res_pointNet(plane_feature)

                # with torch.no_grad():
                room_result['plane_result_list'].append(plane_result.squeeze())
                mask_1 = self.sigmoid(plane_result) > 0.5
                mask_2 = self.sigmoid(plane_result) <= 0.5

                if len(torch.nonzero(mask_1)) != 0:
                    idx_on_plane = idx_temp[mask_1.squeeze()]
                    room_result['on_plane_idx_list'].append(idx_on_plane)
                    on_plane_feature = plane_feature[:, :, mask_1.squeeze()]
                    # on_plane_feature = torch.nn.functional.max_pool2d(on_plane_feature,
                    #                                                   kernel_size=[on_plane_feature.shape[2], 1])
                    on_plane_feature = self.att_pooling(on_plane_feature, 2)
                    on_plane_feature = torch.cat([on_plane_feature, ori_plane_feature], dim=1)
                    room_result['on_plane_feature_list'].append(on_plane_feature)

                else:
                    room_result['on_plane_idx_list'].append([])
                    room_result['on_plane_feature_list'].append([])

                if len(torch.nonzero(mask_2)) != 0:
                    idx_off_plane = idx_temp[mask_2.squeeze()]
                    room_result['off_plane_idx_list'].append(idx_off_plane)
                    off_plane_feature = plane_feature[:, :, mask_2.squeeze()]
                    # off_plane_feature = torch.nn.functional.max_pool2d(off_plane_feature,
                    #                                                    kernel_size=[off_plane_feature.shape[2], 1])
                    off_plane_feature = self.att_pooling(off_plane_feature, 2)
                    off_plane_feature = torch.cat([off_plane_feature, ori_plane_feature], dim=1)
                    room_result['off_plane_feature_list'].append(off_plane_feature)
                else:
                    room_result['off_plane_idx_list'].append([])
                    room_result['off_plane_feature_list'].append([])

        return room_result


def compute_acc(end_points):
    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])
    end_points['acc'] = acc
    return acc, end_points


class IoUCalculator:
    def __init__(self, cfg):
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, end_points):
        logits = end_points['valid_logits']
        labels = end_points['valid_labels']
        pred = logits.max(dim=1)[1]
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        val_total_correct = 0
        val_total_seen = 0

        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.cfg.num_classes, 1))
        self.gt_classes += np.sum(conf_matrix, axis=1)  # TP + FN
        self.positive_classes += np.sum(conf_matrix, axis=0)  # TP + FP
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            print(self.gt_classes[n], self.positive_classes[n], self.true_positive_classes[n])
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / float(
                    self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc + shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  # d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out // 2, kernel_size=(1, 1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out // 2)

        self.mlp2 = pt_utils.Conv2d(d_out // 2, d_out // 2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)),
                                             neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)),
                                             neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        relative_dis = torch.sqrt(
            torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz],
                                     dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out, bn=True):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=bn)

    def forward(self, feature_set, dim=3):
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=dim)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=dim, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


def compute_loss(end_points, cfg):
    logits = end_points['logits']
    labels = end_points['labels']

    logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
    labels = labels.reshape(-1)

    # Boolean mask of points that should be ignored
    ignored_bool = torch.zeros_like(labels)
    for ign_label in cfg.ignored_label_inds:
        ignored_bool = ignored_bool | (labels == ign_label).long()

    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    valid_logits = logits[valid_idx, :]
    valid_labels_init = labels[valid_idx]

    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, cfg.num_classes).long().cuda()
    inserted_value = torch.zeros((1,)).long().cuda()
    for ign_label in cfg.ignored_label_inds:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
    loss = get_loss(valid_logits, valid_labels, cfg.class_weights)
    # loss = get_loss(valid_logits, valid_labels)
    end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
    end_points['loss'] = loss
    return loss, end_points

def get_loss(logits, labels, pre_cal_weights=None):
    # calculate the weighted cross entropy according to the inverse frequency
    if pre_cal_weights is not None:
        class_weights = torch.from_numpy(pre_cal_weights).float().cuda()
    # one_hot_labels = F.one_hot(labels, self.config.num_classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')
    output_loss = criterion(logits, labels)
    output_loss = output_loss.mean()
    return output_loss

def compute_plane_loss(end_points):
    room_results = end_points['room_results']
    labels = end_points['labels']
    # labels = labels.reshape(-1)
    # FIXME data unbalance
    bce_loss = torch.nn.BCEWithLogitsLoss()
    sigmoid = torch.nn.Sigmoid()
    # focal_loss = pt_utils.FocalLoss(alpha=0.75, gamma=2)
    result_list = []
    label_list = []

    for i in range(len(room_results)):
        # every room
        room_result = room_results[i]
        for j in range(len(room_result['on_plane_idx_list'])):
            # every plane
            plane_label = room_result['label_list'][j]
            plane_mask = room_result['mask'][j]
            result_list.append(room_result['plane_result_list'][j])

            label = labels[i, plane_mask]
            temp = torch.ones_like(label).float()
            temp[label != plane_label] = 0.
            label_list.append(temp)

    # plane_result = torch.cat(result_list, dim=2)
    label_result = torch.cat(label_list, dim=0)
    loss = bce_loss(torch.cat(result_list, dim=0).squeeze(), label_result) * 30
    end_points['plane_loss'] = loss
    with torch.no_grad():
        plane_result = sigmoid(torch.cat(result_list, dim=0))
        # print('sigmoid')
        plane_result[plane_result > 0.5] = 1.
        plane_result[plane_result <= 0.5] = 0.

        acc = (plane_result.squeeze() == label_result).sum().float() / float(label_result.shape[0])
        end_points['plane_result'] = plane_result
        end_points['label_result'] = label_result
        end_points['plane_acc'] = acc
    return loss, end_points

def visual_plane_error(end_points):
    room_results = end_points['room_results']
    labels = end_points['labels']
    preds = torch.ones_like(labels)
   # FIXME data unbalance

    sigmoid = torch.nn.Sigmoid()

    for i in range(len(room_results)):
        # every room
        room_result = room_results[i]
        result = torch.ones_like(labels[i]) * -1
        pred = torch.ones_like(labels[i]) * -1
        for j in range(len(room_result['on_plane_idx_list'])):
            # every plane
            plane_label = room_result['label_list'][j]
            plane_mask = room_result['mask'][j]
            plane_result = sigmoid(room_result['plane_result_list'][j])

            plane_result[plane_result > 0.5] = 1.
            plane_result[plane_result <= 0.5] = 0.

            label = labels[i, plane_mask]
            temp = torch.ones_like(label).float()
            temp[label != plane_label] = 0.

            idx1 = torch.where(plane_result == temp)
            idx2 = torch.where(plane_result != temp)
            temp[idx1] = 1
            temp[idx2] = 0
            result[plane_mask] = temp.long()
            pred[plane_mask] = plane_result.long()
        labels[i] = result
        preds[i] = pred

    end_points['plane_visual'] = labels
    end_points['plane_visual_pred'] = preds
    # end_points['label_result'] = label_result
    # end_points['plane_acc'] = acc
    return end_points
