# -*- coding: utf-8 -*-
"""Sanity checks for confidence-aware planner motion conditioning."""

import inspect
import math
import os
import sys
import unittest

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from opencood.models.point_pillar import (  # noqa: E402
    MOTION_MODE_CHANNELS,
    PointPillar,
    resolve_motion_mode,
)
from opencood.models.sub_modules.v2xverse_planning_head import (  # noqa: E402
    V2XVersePlanningHead,
)
from opencood.models.sub_modules.v2xverse_planning_head_mean import (  # noqa: E402
    V2XVerseMeanPlanningHead,
)


def _minimal_planning_args(motion_mode='none', pooling='mean'):
    return {
        'dual_frame': False,
        'freeze_backbone': True,
        'lidar_range': [-140.8, -40, -3, 140.8, 40, 1],
        'anchor_number': 2,
        'box_code_size': 8,
        'voxel_size': [0.4, 0.4, 4],
        'pillar_vfe': {
            'use_norm': True,
            'with_distance': False,
            'use_absolute_xyz': True,
            'num_point_features': 4,
            'num_filters': [64],
        },
        'point_pillar_scatter': {
            'num_features': 64,
            'grid_size': [704, 200, 1],
        },
        'base_bev_backbone': {
            'layer_nums': [3, 5, 8],
            'layer_strides': [2, 2, 2],
            'num_filters': [64, 128, 256],
            'upsample_strides': [1, 2, 4],
            'num_upsample_filter': [128, 128, 128],
        },
        'planning_head': {
            'enabled': True,
            'type': 'v2xverse',
            'motion_mode': motion_mode,
            'pooling': pooling,
            'bev_channels': 384,
            'feature_dir': 128,
            'input_frame': 5,
            'num_waypoints': 10,
        },
    }


class ConfidenceAwareMotionTests(unittest.TestCase):
    def test_gated_speed_is_product_not_normalized(self):
        """confidence=0.001, speed=1.0 -> gated ~= 0.001, not ~1.0."""
        confidence = torch.tensor([[[[0.001]]]])
        predicted_speed = torch.tensor([[[[1.0]]]])
        gated = confidence * predicted_speed
        self.assertAlmostEqual(float(gated), 0.001, places=6)
        self.assertNotAlmostEqual(float(gated), 1.0, places=3)

    def test_predicted_motion_bev_no_confidence_division(self):
        model = PointPillar(_minimal_planning_args('gated_speed'))
        model.eval()
        b, a, h, w = 1, 2, 4, 5
        # Anchor 0: very low score, speed 1.0; anchor 1 even lower.
        psm = torch.full((b, a, h, w), -10.0)
        psm[:, 0] = -6.90675478  # sigmoid ≈ 0.001
        rm = torch.zeros(b, a * 8, h, w)
        # speed channel at index 7 within each anchor block
        rm = rm.view(b, a, 8, h, w)
        rm[:, 0, 7] = 1.0
        rm[:, 1, 7] = 2.0
        rm = rm.view(b, a * 8, h, w)

        confidence, gated_speed = model.predicted_motion_bev(psm, rm)
        self.assertEqual(tuple(confidence.shape), (1, 1, h, w))
        self.assertEqual(tuple(gated_speed.shape), (1, 1, h, w))
        self.assertTrue(torch.isfinite(confidence).all())
        self.assertTrue(torch.isfinite(gated_speed).all())
        # Best anchor is 0 with conf≈0.001 and speed 1.0 -> gated≈0.001
        self.assertTrue(torch.allclose(
            gated_speed, confidence * 1.0, atol=1e-5))
        self.assertLess(float(gated_speed.mean()), 0.01)
        # Explicitly reject confidence-normalized behavior (~1.0).
        self.assertGreater(abs(float(gated_speed.mean()) - 1.0), 0.5)

    def test_motion_mode_channel_counts(self):
        self.assertEqual(MOTION_MODE_CHANNELS['none'], 6)
        self.assertEqual(MOTION_MODE_CHANNELS['gated_speed'], 7)
        self.assertEqual(MOTION_MODE_CHANNELS['confidence_gated_speed'], 8)

        for mode, n_ch in MOTION_MODE_CHANNELS.items():
            model = PointPillar(_minimal_planning_args(mode))
            self.assertEqual(model.motion_mode, mode)
            self.assertEqual(model.occupancy_channels, n_ch)
            self.assertEqual(
                model.planning_head.occupancy_channels, n_ch)

    def test_legacy_use_velocity_maps_to_gated_speed(self):
        self.assertEqual(
            resolve_motion_mode({'use_velocity_in_planning': True}),
            'gated_speed')
        self.assertEqual(
            resolve_motion_mode({'use_velocity_in_planning': False}),
            'none')
        self.assertEqual(
            resolve_motion_mode({'motion_mode': 'confidence_gated_speed'}),
            'confidence_gated_speed')

    def test_historical_six_channel_mean_head(self):
        head = V2XVerseMeanPlanningHead(
            feature_dir=128, occupancy_channels=6)
        self.assertEqual(head.occupancy_channels, 6)
        self.assertEqual(head.pooling, 'mean')
        self.assertFalse(hasattr(head, 'spatial_attn') and
                         head.__dict__.get('spatial_attn') is not None)

    def test_residual_attention_alpha_init_and_shape(self):
        head = V2XVersePlanningHead(
            feature_dir=128,
            occupancy_channels=8,
            pooling='residual_attention',
        )
        alpha = float(torch.sigmoid(head.attn_mix_logit))
        self.assertAlmostEqual(alpha, 0.10, places=4)

        b, t, h, w = 2, 5, 8, 10
        occupancy = torch.zeros(b, t, 8, h, w)
        feature_seq = torch.zeros(b, t, 128, h, w)
        target = torch.zeros(b, 2)
        out = head({
            'occupancy': occupancy,
            'feature_warpped_list': [feature_seq],
            'target': target,
        })
        self.assertEqual(tuple(out['future_waypoints'].shape), (b, 10, 2))

        # Directly check pooled feature shape via pool_spatial.
        x_3 = torch.randn(b, 256, 4, 5)
        pooled = head.pool_spatial(x_3)
        self.assertEqual(tuple(pooled.shape), (b, 256))

    def test_motion_raster_source_has_no_confidence_division(self):
        src = inspect.getsource(PointPillar.predicted_motion_bev)
        self.assertNotIn('/ denom', src)
        self.assertNotIn('scores.sum', src)
        # Must multiply confidence * speed, not normalize.
        self.assertIn('gated_speed = confidence * best_speed', src)
        self.assertNotIn('clamp_min(1e-6)', src)


if __name__ == '__main__':
    unittest.main()
