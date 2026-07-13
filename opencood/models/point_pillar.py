# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.v2xverse_planning_head import V2XVersePlanningHead


class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()

        self.dual_frame = args.get('dual_frame', False)
        self.freeze_backbone = args.get('freeze_backbone', False)
        self.lidar_range = args['lidar_range']
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
            self.planning_head = V2XVersePlanningHead(
                feature_dir=planning_args.get('feature_dir', bev_channels),
                input_frame=planning_args.get('input_frame', 5),
                output_points=planning_args.get('num_waypoints', 10),
            )

    def encode_frame(self, processed_lidar):
        batch_dict = {'voxel_features': processed_lidar['voxel_features'],
                      'voxel_coords': processed_lidar['voxel_coords'],
                      'voxel_num_points': processed_lidar['voxel_num_points']}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        return batch_dict['spatial_features_2d']

    def build_v2xverse_occupancy(self, data_dict, h, w, device):
        """
        Build V2Xverse-style occupancy [B, 5, 6, H, W] and target [B, 2]
        from EarlyFusionDataset tensors.
        """
        object_bbx_center = data_dict['object_bbx_center']
        object_bbx_mask = data_dict['object_bbx_mask']
        future_waypoints = data_dict['future_waypoints']

        batch_size = object_bbx_center.shape[0]
        x_min, y_min, _, x_max, y_max, _ = self.lidar_range

        occupancy_single = torch.zeros(
            batch_size, 6, h, w, device=device, dtype=torch.float32)

        # channel 0: actor occupancy from object centers
        centers = object_bbx_center[..., :2]  # [B, max_num, 2]
        mask = object_bbx_mask.bool()
        for b in range(batch_size):
            valid = centers[b][mask[b]]
            if valid.numel() == 0:
                continue
            x_idx = ((valid[:, 0] - x_min) / (x_max - x_min) * (w - 1)).long()
            y_idx = ((valid[:, 1] - y_min) / (y_max - y_min) * (h - 1)).long()
            x_idx = torch.clamp(x_idx, 0, w - 1)
            y_idx = torch.clamp(y_idx, 0, h - 1)
            occupancy_single[b, 0, y_idx, x_idx] = 1.0

        # channel 1: ego occupancy at ego origin (0, 0)
        ego_x = int(round((0.0 - x_min) / (x_max - x_min) * (w - 1)))
        ego_y = int(round((0.0 - y_min) / (y_max - y_min) * (h - 1)))
        ego_x = max(0, min(w - 1, ego_x))
        ego_y = max(0, min(h - 1, ego_y))
        occupancy_single[:, 1, ego_y, ego_x] = 1.0

        # target: final GT waypoint
        target = future_waypoints[:, -1, :]  # [B, 2]

        # channel 2: local command / target point occupancy
        tx = ((target[:, 0] - x_min) / (x_max - x_min) * (w - 1)).long()
        ty = ((target[:, 1] - y_min) / (y_max - y_min) * (h - 1)).long()
        tx = torch.clamp(tx, 0, w - 1)
        ty = torch.clamp(ty, 0, h - 1)
        for b in range(batch_size):
            occupancy_single[b, 2, ty[b], tx[b]] = 1.0

        # channel 3/4: normalized x/y coordinate maps
        xs = torch.linspace(0.0, 1.0, w, device=device).view(1, 1, 1, w)
        ys = torch.linspace(0.0, 1.0, h, device=device).view(1, 1, h, 1)
        occupancy_single[:, 3:4] = xs.expand(batch_size, 1, h, w)
        occupancy_single[:, 4:5] = ys.expand(batch_size, 1, h, w)

        # channel 5: zero road-map placeholder (already zeros)

        occupancy = occupancy_single.unsqueeze(1).repeat(1, 5, 1, 1, 1)
        return occupancy, target

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
            if self.dual_frame:
                if self.freeze_backbone:
                    feat_cur = feat_cur.detach()
                    feat_prev = feat_prev.detach()
                feature_seq = torch.stack(
                    [feat_prev, feat_prev, feat_prev, feat_cur, feat_cur],
                    dim=1)
            else:
                feature_seq = spatial_features_2d.unsqueeze(1).repeat(
                    1, 5, 1, 1, 1)

            if self.freeze_backbone:
                feature_seq = feature_seq.detach()

            occupancy, target = self.build_v2xverse_occupancy(
                data_dict,
                spatial_features_2d.shape[-2],
                spatial_features_2d.shape[-1],
                spatial_features_2d.device,
            )

            planner_out = self.planning_head({
                "occupancy": occupancy,
                "feature_warpped_list": [feature_seq],
                "target": target,
            })
            output_dict["future_waypoints"] = planner_out["future_waypoints"]

        return output_dict
