# -*- coding: utf-8 -*-
# PointPillar with the original V2Xverse mean-pool planner by default.
# Optionally constructs the attention / residual-attention head when
# planning_head.pooling is set accordingly (Experiment 3).

from opencood.models.point_pillar import PointPillar
from opencood.models.sub_modules.v2xverse_planning_head import (
    V2XVersePlanningHead,
)
from opencood.models.sub_modules.v2xverse_planning_head_mean import (
    V2XVerseMeanPlanningHead,
)


class PointPillarV2XverseMean(PointPillar):
    """Frozen-backbone planner graph; default pooling is uniform spatial mean."""

    def __init__(self, args):
        super(PointPillarV2XverseMean, self).__init__(args)
        if not self.use_planning_head:
            return
        planning_args = args.get('planning_head', {})
        planner_in = planning_args.get('feature_dir', 128)
        pooling = planning_args.get('pooling', 'mean')
        self.planner_pooling = pooling

        if pooling == 'mean':
            # Drop the attention head constructed by PointPillar.__init__.
            # Mean head has no spatial_attn params (historical checkpoint compat).
            self.planning_head = V2XVerseMeanPlanningHead(
                feature_dir=planner_in,
                input_frame=self.input_frame,
                output_points=planning_args.get('num_waypoints', 10),
                occupancy_channels=self.occupancy_channels,
            )
        elif pooling in ('attention', 'residual_attention'):
            self.planning_head = V2XVersePlanningHead(
                feature_dir=planner_in,
                input_frame=self.input_frame,
                output_points=planning_args.get('num_waypoints', 10),
                occupancy_channels=self.occupancy_channels,
                pooling=pooling,
            )
        else:
            raise ValueError(
                "planning_head.pooling must be mean, attention, or "
                "residual_attention, got %r" % pooling)
