# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.anchor_number = args['anchor_number']
        self.box_code_size = args.get('box_code_size', 8)
        self.speed_norm = float(args.get('speed_norm', 30.0))
        bev_channels = 128 * 3
        planning_args = args.get('planning_head', {})
        self.use_planning_head = planning_args.get('enabled', False)
        self.use_velocity_in_planning = bool(
            planning_args.get('use_velocity_in_planning', False))
        self.input_frame = int(planning_args.get(
            'input_frame', args.get('history_frames', 5)))
        self.occupancy_channels = 7 if self.use_velocity_in_planning else 6

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
        box_code_size = self.box_code_size

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
                occupancy_channels=self.occupancy_channels,
            )

    def encode_frame(self, processed_lidar):
        batch_dict = {'voxel_features': processed_lidar['voxel_features'],
                      'voxel_coords': processed_lidar['voxel_coords'],
                      'voxel_num_points': processed_lidar['voxel_num_points']}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        return batch_dict['spatial_features_2d']

    def encode_history(self, hist_lidars):
        """
        Encode T history voxel batches in one backbone pass.

        Each item is a collated batch with the same batch size. Voxel batch
        indices are offset so time is folded into the batch dimension, then
        split back to [B, T, C, H, W].
        """
        voxel_features = []
        voxel_coords = []
        voxel_num_points = []
        batch_size = None
        for t, lidar in enumerate(hist_lidars):
            coords = lidar['voxel_coords']
            if coords.numel() == 0:
                b = 1 if batch_size is None else batch_size
            else:
                b = int(coords[:, 0].max().item()) + 1
            if batch_size is None:
                batch_size = b
            coords = coords.clone()
            if coords.numel() > 0:
                coords[:, 0] = coords[:, 0] + t * batch_size
            voxel_features.append(lidar['voxel_features'])
            voxel_coords.append(coords)
            voxel_num_points.append(lidar['voxel_num_points'])

        merged = {
            'voxel_features': torch.cat(voxel_features, 0),
            'voxel_coords': torch.cat(voxel_coords, 0),
            'voxel_num_points': torch.cat(voxel_num_points, 0),
        }
        t_len = len(hist_lidars)
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.encode_frame(merged)
        else:
            feat = self.encode_frame(merged)
        if feat.shape[0] != t_len * batch_size:
            # A history frame with no voxels drops a batch index. Fall back
            # so the time layout stays [B, T, C, H, W].
            frames = []
            for lidar in hist_lidars:
                if self.freeze_backbone:
                    with torch.no_grad():
                        frames.append(self.encode_frame(lidar))
                else:
                    frames.append(self.encode_frame(lidar))
            return torch.stack(frames, dim=1)
        c, h, w = feat.shape[1:]
        return feat.view(t_len, batch_size, c, h, w).transpose(0, 1).contiguous()

    def predicted_speed_bev(self, psm, rm):
        """
        Soft BEV speed map from frozen detection heads.

        psm: [B, A, H, W], rm: [B, A * box_code_size, H, W]
        Returns normalized speed in roughly [0, 1+] as [B, 1, H, W].
        """
        assert self.box_code_size >= 8, (
            "predicted speed requires box_code_size >= 8")
        b, a, h, w = psm.shape
        rm = rm.view(b, a, self.box_code_size, h, w)
        # Network regresses speed / speed_norm (see VoxelPostprocessor).
        speed_normed = rm[:, :, 7, :, :]
        scores = torch.sigmoid(psm)
        denom = scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
        vel = (scores * speed_normed).sum(dim=1, keepdim=True) / denom
        return vel.clamp(0.0, 2.0)

    def history_predicted_velocity_maps(self, hist_feats):
        """
        Per-history-frame predicted velocity BEV maps [B, T, 1, H, W].

        Uses the same dual-frame temporal fusion + cls/reg heads as detection
        so the planner consumes the course-project velocity prediction, not GT.
        Batched over time to avoid serial GPU launches.
        """
        b, t_len, c, h, w = hist_feats.shape
        feat_t = hist_feats
        if self.dual_frame:
            feat_prev = torch.cat(
                [hist_feats[:, :1], hist_feats[:, :-1]], dim=1)
            fused = self.temporal_fusion(
                torch.cat([feat_t, feat_prev], dim=2).reshape(
                    b * t_len, c * 2, h, w))
        else:
            fused = feat_t.reshape(b * t_len, c, h, w)
        if self.freeze_backbone:
            fused = fused.detach()
        psm_t = self.cls_head(fused)
        rm_t = self.reg_head(fused)
        vel = self.predicted_speed_bev(psm_t, rm_t)
        return vel.view(b, t_len, 1, h, w)

    def build_v2xverse_occupancy(self, data_dict, h, w, device,
                                 velocity_maps=None):
        """
        Build V2Xverse-style occupancy [B, T, C, H, W] and target [B, 2].

        Channel meanings (V2Xverse pnp_dataset / generate_planning_input):
          0: other-actor occupancy (per history frame)
          1: ego occupancy (past ego pose in current ego frame)
          2: local navigation-command / target-point occupancy
          3: metric x coordinate map
          4: metric y coordinate map
          5: drivable-area / road map (zeros on OPV2V; no birdview)
          6: (optional) predicted actor speed BEV map from detection head

        Target is `planning_target` (route command), never a future waypoint.
        """
        assert 'planning_target' in data_dict, (
            "planning_target missing — refusing to derive target from "
            "future_waypoints (endpoint leakage).")
        target = data_dict['planning_target'].to(
            device=device, dtype=torch.float32, non_blocking=True)
        if target.ndim == 1:
            target = target.unsqueeze(0)

        batch_size = target.shape[0]
        t_len = self.input_frame
        x_min, y_min, _, x_max, y_max, _ = self.lidar_range
        n_ch = self.occupancy_channels

        occupancy = torch.zeros(
            batch_size, t_len, n_ch, h, w, device=device, dtype=torch.float32)

        history_actor_xy = data_dict.get('history_actor_xy', None)
        history_actor_mask = data_dict.get('history_actor_mask', None)
        history_ego_xy = data_dict.get('history_ego_xy', None)

        def xy_to_idx(xy):
            x_idx = ((xy[..., 0] - x_min) / (x_max - x_min) * (w - 1)).long()
            y_idx = ((xy[..., 1] - y_min) / (y_max - y_min) * (h - 1)).long()
            x_idx = torch.clamp(x_idx, 0, w - 1)
            y_idx = torch.clamp(y_idx, 0, h - 1)
            return x_idx, y_idx

        if history_actor_xy is not None and history_actor_mask is not None:
            for t in range(t_len):
                centers = history_actor_xy[:, t].to(
                    device=device, non_blocking=True)
                mask = history_actor_mask[:, t].to(
                    device=device, non_blocking=True)
                x_idx, y_idx = xy_to_idx(centers)
                b_idx = torch.arange(
                    batch_size, device=device).view(
                        batch_size, 1).expand_as(x_idx)
                valid = mask.bool()
                if valid.any():
                    occupancy[b_idx[valid], t, 0,
                              y_idx[valid], x_idx[valid]] = 1.0
        else:
            centers = data_dict['object_bbx_center'][..., :2].to(
                device=device, non_blocking=True)
            mask = data_dict['object_bbx_mask'].to(
                device=device, non_blocking=True)
            x_idx, y_idx = xy_to_idx(centers)
            b_idx = torch.arange(
                batch_size, device=device).view(
                    batch_size, 1).expand_as(x_idx)
            valid = mask.bool()
            if valid.any():
                vb = b_idx[valid]
                vy = y_idx[valid]
                vx = x_idx[valid]
                for t in range(t_len):
                    occupancy[vb, t, 0, vy, vx] = 1.0

        if history_ego_xy is not None:
            ego_xy = history_ego_xy.to(device=device, non_blocking=True)
        else:
            ego_xy = torch.zeros(
                batch_size, t_len, 2, device=device, dtype=torch.float32)
        if ego_xy.ndim == 2:
            ego_xy = ego_xy.unsqueeze(1).expand(-1, t_len, -1)
        ex, ey = xy_to_idx(ego_xy)
        b_idx = torch.arange(batch_size, device=device).view(
            batch_size, 1).expand_as(ex)
        t_idx = torch.arange(t_len, device=device).view(
            1, t_len).expand_as(ex)
        occupancy[b_idx.reshape(-1), t_idx.reshape(-1), 1,
                  ey.reshape(-1), ex.reshape(-1)] = 1.0

        tx, ty = xy_to_idx(target)
        b_idx = torch.arange(batch_size, device=device).view(batch_size, 1).expand(
            batch_size, t_len)
        t_idx = torch.arange(t_len, device=device).view(1, t_len).expand(
            batch_size, t_len)
        ty_exp = ty.view(batch_size, 1).expand(batch_size, t_len)
        tx_exp = tx.view(batch_size, 1).expand(batch_size, t_len)
        occupancy[b_idx.reshape(-1), t_idx.reshape(-1), 2,
                  ty_exp.reshape(-1), tx_exp.reshape(-1)] = 1.0

        xs = torch.linspace(x_min, x_max, w, device=device).view(1, 1, 1, w)
        ys = torch.linspace(y_min, y_max, h, device=device).view(1, 1, h, 1)
        occupancy[:, :, 3:4] = xs.expand(batch_size, t_len, 1, h, w)
        occupancy[:, :, 4:5] = ys.expand(batch_size, t_len, 1, h, w)

        if self.use_velocity_in_planning:
            assert velocity_maps is not None, (
                "use_velocity_in_planning requires velocity_maps")
            assert velocity_maps.shape[:2] == (batch_size, t_len), (
                "velocity_maps shape %s does not match occupancy batch/time"
                % (tuple(velocity_maps.shape),))
            if velocity_maps.shape[-2:] != (h, w):
                flat = velocity_maps.view(
                    batch_size * t_len, 1,
                    velocity_maps.shape[-2], velocity_maps.shape[-1])
                flat = F.interpolate(
                    flat, size=(h, w), mode='bilinear', align_corners=False)
                velocity_maps = flat.view(batch_size, t_len, 1, h, w)
            occupancy[:, :, 6:7] = velocity_maps

        return occupancy, target

    def forward(self, data_dict):
        planning_only = bool(getattr(self, 'planning_only', False))
        # Planner-only training does not use detection heads. Encoding the
        # current/prev pair on top of the history sequence just doubles work.
        if not (planning_only and self.use_planning_head):
            if self.dual_frame:
                if self.freeze_backbone:
                    with torch.no_grad():
                        feat_cur = self.encode_frame(
                            data_dict['processed_lidar'])
                        feat_prev = self.encode_frame(
                            data_dict['processed_lidar_prev'])
                        spatial_features_2d = self.temporal_fusion(
                            torch.cat([feat_cur, feat_prev], dim=1))
                else:
                    feat_cur = self.encode_frame(data_dict['processed_lidar'])
                    feat_prev = self.encode_frame(
                        data_dict['processed_lidar_prev'])
                    spatial_features_2d = self.temporal_fusion(
                        torch.cat([feat_cur, feat_prev], dim=1))
            else:
                if self.freeze_backbone:
                    with torch.no_grad():
                        spatial_features_2d = self.encode_frame(
                            data_dict['processed_lidar'])
                else:
                    spatial_features_2d = self.encode_frame(
                        data_dict['processed_lidar'])

            if self.freeze_backbone and self.training:
                spatial_features_2d = spatial_features_2d.detach()

            psm = self.cls_head(spatial_features_2d)
            rm = self.reg_head(spatial_features_2d)
            output_dict = {'psm': psm, 'rm': rm}
        else:
            output_dict = {}

        if self.use_planning_head:
            # Encode five real chronological history frames (oldest -> current).
            if 'processed_lidar_history' in data_dict:
                feature_seq = self.encode_history(
                    data_dict['processed_lidar_history'])
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

            velocity_maps = None
            if self.use_velocity_in_planning:
                # Predicted speed from frozen det heads (before planner adapter).
                with torch.no_grad():
                    velocity_maps = self.history_predicted_velocity_maps(
                        feature_seq)

            # Planner-owned adapter (trainable) after frozen BEV features.
            b, t, c, h, w = feature_seq.shape
            feat_flat = feature_seq.view(b * t, c, h, w)
            if self.planning_feature_adapter is not None:
                feat_flat = self.planning_feature_adapter(feat_flat)
            feature_seq = feat_flat.view(
                b, t, feat_flat.shape[1], h, w)

            occupancy, target = self.build_v2xverse_occupancy(
                data_dict, h, w, feature_seq.device,
                velocity_maps=velocity_maps)

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
