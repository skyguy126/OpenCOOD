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
        self.input_frame = int(planning_args.get(
            'input_frame', args.get('history_frames', 5)))

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

        self.planning_feature_adapter = None
        if self.use_planning_head:
            planner_in = planning_args.get('feature_dir', 128)
            bev_in = planning_args.get('bev_channels', bev_channels)
            # Planner-owned adapter: OpenCOOD BEV (384) -> V2Xverse feature_dir (128).
            if bev_in != planner_in:
                self.planning_feature_adapter = nn.Sequential(
                    nn.Conv2d(bev_in, planner_in, kernel_size=1, bias=False),
                    nn.BatchNorm2d(planner_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                )
            self.planning_head = V2XVersePlanningHead(
                feature_dir=planner_in,
                input_frame=self.input_frame,
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
        Build V2Xverse-style occupancy [B, T, 6, H, W] and target [B, 2].

        Channel meanings (V2Xverse pnp_dataset / generate_planning_input):
          0: other-actor occupancy (per history frame)
          1: ego occupancy (past ego pose in current ego frame)
          2: local navigation-command / target-point occupancy
          3: metric x coordinate map
          4: metric y coordinate map
          5: drivable-area / road map (zeros on OPV2V; no birdview)

        Target is `planning_target` (route command), never a future waypoint.
        """
        assert 'planning_target' in data_dict, (
            "planning_target missing — refusing to derive target from "
            "future_waypoints (endpoint leakage).")
        target = data_dict['planning_target'].to(device=device, dtype=torch.float32)
        if target.ndim == 1:
            target = target.unsqueeze(0)

        batch_size = target.shape[0]
        t_len = self.input_frame
        x_min, y_min, _, x_max, y_max, _ = self.lidar_range

        occupancy = torch.zeros(
            batch_size, t_len, 6, h, w, device=device, dtype=torch.float32)

        history_actor_xy = data_dict.get('history_actor_xy', None)
        history_actor_mask = data_dict.get('history_actor_mask', None)
        history_ego_xy = data_dict.get('history_ego_xy', None)

        def xy_to_idx(xy):
            # xy: [..., 2] in ego meters -> (x_idx along W, y_idx along H)
            x_idx = ((xy[..., 0] - x_min) / (x_max - x_min) * (w - 1)).long()
            y_idx = ((xy[..., 1] - y_min) / (y_max - y_min) * (h - 1)).long()
            x_idx = torch.clamp(x_idx, 0, w - 1)
            y_idx = torch.clamp(y_idx, 0, h - 1)
            return x_idx, y_idx

        for t in range(t_len):
            # channel 0: actors at history timestamp t
            if history_actor_xy is not None and history_actor_mask is not None:
                centers = history_actor_xy[:, t]  # [B, max_num, 2]
                mask = history_actor_mask[:, t].bool()
                for b in range(batch_size):
                    valid = centers[b][mask[b]]
                    if valid.numel() == 0:
                        continue
                    x_idx, y_idx = xy_to_idx(valid)
                    occupancy[b, t, 0, y_idx, x_idx] = 1.0
            else:
                # Fallback: current-frame boxes only (still no future leakage).
                centers = data_dict['object_bbx_center'][..., :2]
                mask = data_dict['object_bbx_mask'].bool()
                for b in range(batch_size):
                    valid = centers[b][mask[b]]
                    if valid.numel() == 0:
                        continue
                    x_idx, y_idx = xy_to_idx(valid)
                    occupancy[b, t, 0, y_idx, x_idx] = 1.0

            # channel 1: ego occupancy at past ego position in current frame
            if history_ego_xy is not None:
                ego_xy = history_ego_xy[:, t]  # [B, 2]
            else:
                ego_xy = torch.zeros(batch_size, 2, device=device)
            ex, ey = xy_to_idx(ego_xy)
            for b in range(batch_size):
                occupancy[b, t, 1, ey[b], ex[b]] = 1.0

            # channel 2: local command / target point (same across time)
            tx, ty = xy_to_idx(target)
            for b in range(batch_size):
                occupancy[b, t, 2, ty[b], tx[b]] = 1.0

        # channels 3/4: metric coordinate maps (V2Xverse-style, not 0-1)
        xs = torch.linspace(x_min, x_max, w, device=device).view(1, 1, 1, w)
        ys = torch.linspace(y_min, y_max, h, device=device).view(1, 1, h, 1)
        occupancy[:, :, 3:4] = xs.expand(batch_size, t_len, 1, h, w)
        occupancy[:, :, 4:5] = ys.expand(batch_size, t_len, 1, h, w)
        # channel 5: road/drivable map unavailable on OPV2V -> zeros

        return occupancy, target

    def forward(self, data_dict):
        # Detection path: keep existing early-fusion dual-frame temporal fusion.
        if self.dual_frame:
            feat_cur = self.encode_frame(data_dict['processed_lidar'])
            feat_prev = self.encode_frame(data_dict['processed_lidar_prev'])
            spatial_features_2d = self.temporal_fusion(
                torch.cat([feat_cur, feat_prev], dim=1))
        else:
            spatial_features_2d = self.encode_frame(
                data_dict['processed_lidar'])
            feat_cur = spatial_features_2d

        if self.freeze_backbone and self.training:
            spatial_features_2d = spatial_features_2d.detach()

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {'psm': psm,
                       'rm': rm}

        if self.use_planning_head:
            # Encode five real chronological history frames (oldest -> current).
            if 'processed_lidar_history' in data_dict:
                hist_feats = []
                for hist_lidar in data_dict['processed_lidar_history']:
                    feat_t = self.encode_frame(hist_lidar)
                    if self.freeze_backbone:
                        feat_t = feat_t.detach()
                    hist_feats.append(feat_t)
                feature_seq = torch.stack(hist_feats, dim=1)  # [B, T, C, H, W]
            elif self.dual_frame:
                # Should not happen for the planning baseline; fail loudly.
                raise RuntimeError(
                    "processed_lidar_history missing while planning head is "
                    "enabled. Refusing to pad/repeat dual-frame features into "
                    "a fake 5-frame sequence.")
            else:
                raise RuntimeError(
                    "processed_lidar_history required for V2Xverse planning.")

            assert feature_seq.shape[1] == self.input_frame, (
                "Expected %d history BEV frames, got %d"
                % (self.input_frame, feature_seq.shape[1]))

            # Planner-owned adapter (trainable) after frozen BEV features.
            b, t, c, h, w = feature_seq.shape
            feat_flat = feature_seq.view(b * t, c, h, w)
            if self.planning_feature_adapter is not None:
                feat_flat = self.planning_feature_adapter(feat_flat)
            feature_seq = feat_flat.view(
                b, t, feat_flat.shape[1], h, w)

            occupancy, target = self.build_v2xverse_occupancy(
                data_dict, h, w, feature_seq.device)

            # Leakage guards (training + eval): target must not be GT endpoint.
            if 'future_waypoints' in data_dict:
                gt_end = data_dict['future_waypoints'][:, -1, :].to(target.device)
                if torch.allclose(target, gt_end, atol=1e-4, rtol=0):
                    raise RuntimeError(
                        "Endpoint leakage detected: planning_target equals "
                        "future_waypoints[:, -1].")

            planner_out = self.planning_head({
                "occupancy": occupancy,
                "feature_warpped_list": [feature_seq],
                "target": target,
            })
            output_dict["future_waypoints"] = planner_out["future_waypoints"]
            output_dict["planning_target"] = target
            output_dict["planning_occupancy"] = occupancy
            output_dict["planning_feature_seq"] = feature_seq

        return output_dict
