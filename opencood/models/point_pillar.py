# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.simple_planning_head import SimplePlanningHead


class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()

        self.dual_frame = args.get('dual_frame', False)
        self.freeze_backbone = args.get('freeze_backbone', False)
        bev_channels = 128 * 3
        planning_args = args.get('planning_head', {})
        self.use_planning_head = planning_args.get('enabled', False)

        # PIllar VFE
        self.pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=args['pillar_vfe'].get('num_point_features', 4),
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range']
        )
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        if self.dual_frame:
            self.temporal_fusion = nn.Sequential(
                nn.Conv2d(bev_channels * 2, bev_channels, kernel_size=3,
                          padding=1, bias=False),
                nn.BatchNorm2d(bev_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            )

        self.cls_head = nn.Conv2d(bev_channels, args['anchor_number'],
                                  kernel_size=1)
        box_code_size = args.get('box_code_size', 8)

        self.reg_head = nn.Conv2d(
            bev_channels,
            box_code_size * args['anchor_number'],
            kernel_size=1
        )

        if self.use_planning_head:
            self.planning_head = SimplePlanningHead(
                in_channels=bev_channels,
                hidden_channels=planning_args.get('hidden_channels', 128),
                mlp_hidden_dim=planning_args.get('mlp_hidden_dim', 256),
                num_waypoints=planning_args.get('num_waypoints', 6),
            )

    def encode_frame(self, processed_lidar):
        batch_dict = {'voxel_features': processed_lidar['voxel_features'],
                      'voxel_coords': processed_lidar['voxel_coords'],
                      'voxel_num_points': processed_lidar['voxel_num_points']}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        return batch_dict['spatial_features_2d']

    def forward(self, data_dict):
        if self.dual_frame:
            feat_cur = self.encode_frame(data_dict['processed_lidar'])
            feat_prev = self.encode_frame(data_dict['processed_lidar_prev'])
            spatial_features_2d = self.temporal_fusion(
                torch.cat([feat_cur, feat_prev], dim=1))
        else:
            spatial_features_2d = self.encode_frame(
                data_dict['processed_lidar'])

        if self.freeze_backbone and self.training:
            spatial_features_2d = spatial_features_2d.detach()

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {'psm': psm,
                       'rm': rm}

        if self.use_planning_head:
            output_dict['future_waypoints'] = self.planning_head(
                spatial_features_2d)

        return output_dict