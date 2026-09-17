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


# Planner occupancy channel counts are derived from motion_mode only.
MOTION_MODE_CHANNELS = {
    'none': 6,
    'gated_speed': 7,
    'confidence_gated_speed': 8,
}


def resolve_motion_mode(planning_args):
    """
    Resolve planner motion_mode from YAML.

    Prefer explicit motion_mode. Legacy use_velocity_in_planning maps to
    gated_speed (confidence * best-anchor speed, no confidence normalization).
    """
    if planning_args is None:
        planning_args = {}
    motion_mode = planning_args.get('motion_mode', None)
    if motion_mode is None:
        if bool(planning_args.get('use_velocity_in_planning', False)):
            motion_mode = 'gated_speed'
        else:
            motion_mode = 'none'
    if motion_mode not in MOTION_MODE_CHANNELS:
        raise ValueError(
            "planning_head.motion_mode must be one of %s, got %r"
            % (sorted(MOTION_MODE_CHANNELS), motion_mode))
    return motion_mode


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
        self.motion_mode = resolve_motion_mode(planning_args)
        self.occupancy_channels = MOTION_MODE_CHANNELS[self.motion_mode]
        # Legacy alias: any non-none motion mode injects motion into occupancy.
        self.use_velocity_in_planning = self.motion_mode != 'none'
        self.input_frame = int(planning_args.get(
            'input_frame', args.get('history_frames', 5)))
        # Default pooling for this class historically replaced mean with attention.
        self.planner_pooling = planning_args.get('pooling', 'attention')

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
                pooling=self.planner_pooling,
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

    def predicted_motion_bev(self, psm, rm):
        """
        Confidence-aware motion maps from frozen detection heads.

        For each BEV cell, take the highest-confidence anchor's objectness and
        its predicted speed, then form gated_speed = confidence * speed.
        Background cells stay near zero because confidence is low; there is
        NO division by summed confidence (that would amplify unsupervised
        background regression values).

        psm: [B, A, H, W], rm: [B, A * box_code_size, H, W]
        Returns:
            confidence: [B, 1, H, W] in approximately [0, 1]
            gated_speed: [B, 1, H, W] (confidence * clamped best-anchor speed)
        """
        assert self.box_code_size >= 8, (
            "predicted motion requires box_code_size >= 8")
        b, a, h, w = psm.shape
        rm = rm.view(b, a, self.box_code_size, h, w)
        # Network regresses speed / speed_norm (see VoxelPostprocessor).
        speed = rm[:, :, 7]
        scores = torch.sigmoid(psm)
        confidence, best_anchor = scores.max(dim=1, keepdim=True)
        best_speed = speed.gather(1, best_anchor)
        best_speed = best_speed.clamp(0.0, 2.0)
        gated_speed = confidence * best_speed

        assert torch.isfinite(confidence).all(), "confidence has non-finite values"
        assert torch.isfinite(gated_speed).all(), "gated_speed has non-finite values"
        return confidence, gated_speed

    def history_predicted_motion_maps(self, hist_feats):
        """
        Per-history-frame confidence and gated-speed BEV maps.

        Returns:
            confidence: [B, T, 1, H, W]
            gated_speed: [B, T, 1, H, W]

        Uses the same dual-frame temporal fusion + cls/reg heads as detection
        so the planner consumes frozen detector predictions, not GT.
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
        confidence, gated_speed = self.predicted_motion_bev(psm_t, rm_t)
        return (
            confidence.view(b, t_len, 1, h, w),
            gated_speed.view(b, t_len, 1, h, w),
        )

    @staticmethod
    def _resize_motion_map(motion_map, batch_size, t_len, h, w):
        """Bilinear-resize a [B,T,1,H',W'] motion map to planner (H, W)."""
        if motion_map.shape[-2:] == (h, w):
            return motion_map
        flat = motion_map.view(
            batch_size * t_len, 1,
            motion_map.shape[-2], motion_map.shape[-1])
        flat = F.interpolate(
            flat, size=(h, w), mode='bilinear', align_corners=False)
        return flat.view(batch_size, t_len, 1, h, w)

    def build_v2xverse_occupancy(self, data_dict, h, w, device,
                                 confidence_maps=None, gated_speed_maps=None):
        """
        Build V2Xverse-style occupancy [B, T, C, H, W] and target [B, 2].

        Channel meanings (V2Xverse pnp_dataset / generate_planning_input):
          0: other-actor occupancy (per history frame)
          1: ego occupancy (past ego pose in current ego frame)
          2: local navigation-command / target-point occupancy
          3: metric x coordinate map
          4: metric y coordinate map
          5: drivable-area / road map (zeros on OPV2V; no birdview)

        Extra channels are controlled by motion_mode (not inferred from C):
          gated_speed (C=7):
            6: confidence * predicted_speed
          confidence_gated_speed (C=8):
            6: detector confidence / objectness
            7: confidence * predicted_speed

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
        n_ch = MOTION_MODE_CHANNELS[self.motion_mode]
        assert n_ch == self.occupancy_channels, (
            "occupancy_channels (%d) disagrees with motion_mode %r (-> %d)"
            % (self.occupancy_channels, self.motion_mode, n_ch))

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

        if self.motion_mode == 'none':
            assert occupancy.shape[2] == 6, (
                "motion_mode=none requires 6 occupancy channels, got %d"
                % occupancy.shape[2])
        elif self.motion_mode == 'gated_speed':
            assert gated_speed_maps is not None, (
                "motion_mode=gated_speed requires gated_speed_maps")
            assert gated_speed_maps.shape[:2] == (batch_size, t_len), (
                "gated_speed_maps shape %s does not match occupancy batch/time"
                % (tuple(gated_speed_maps.shape),))
            gated_speed_maps = self._resize_motion_map(
                gated_speed_maps, batch_size, t_len, h, w)
            occupancy[:, :, 6:7] = gated_speed_maps
            assert occupancy.shape[2] == 7, (
                "motion_mode=gated_speed requires 7 occupancy channels, got %d"
                % occupancy.shape[2])
        elif self.motion_mode == 'confidence_gated_speed':
            assert confidence_maps is not None and gated_speed_maps is not None, (
                "motion_mode=confidence_gated_speed requires confidence_maps "
                "and gated_speed_maps")
            assert confidence_maps.shape[:2] == (batch_size, t_len), (
                "confidence_maps shape %s does not match occupancy batch/time"
                % (tuple(confidence_maps.shape),))
            assert gated_speed_maps.shape[:2] == (batch_size, t_len), (
                "gated_speed_maps shape %s does not match occupancy batch/time"
                % (tuple(gated_speed_maps.shape),))
            confidence_maps = self._resize_motion_map(
                confidence_maps, batch_size, t_len, h, w)
            gated_speed_maps = self._resize_motion_map(
                gated_speed_maps, batch_size, t_len, h, w)
            occupancy[:, :, 6:7] = confidence_maps
            occupancy[:, :, 7:8] = gated_speed_maps
            assert occupancy.shape[2] == 8, (
                "motion_mode=confidence_gated_speed requires 8 occupancy "
                "channels, got %d" % occupancy.shape[2])
        else:
            raise ValueError("Unhandled motion_mode %r" % self.motion_mode)

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

            confidence_maps = None
            gated_speed_maps = None
            if self.motion_mode != 'none':
                # Motion from frozen det heads (before planner adapter).
                with torch.no_grad():
                    confidence_maps, gated_speed_maps = (
                        self.history_predicted_motion_maps(feature_seq))

            # Planner-owned adapter (trainable) after frozen BEV features.
            b, t, c, h, w = feature_seq.shape
            feat_flat = feature_seq.view(b * t, c, h, w)
            if self.planning_feature_adapter is not None:
                feat_flat = self.planning_feature_adapter(feat_flat)
            feature_seq = feat_flat.view(
                b, t, feat_flat.shape[1], h, w)

            occupancy, target = self.build_v2xverse_occupancy(
                data_dict, h, w, feature_seq.device,
                confidence_maps=confidence_maps,
                gated_speed_maps=gated_speed_maps)

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
