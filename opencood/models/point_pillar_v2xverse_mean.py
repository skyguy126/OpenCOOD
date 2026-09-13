# -*- coding: utf-8 -*-
# PointPillar with the original V2Xverse mean-pool planner.
# Does not modify the attention planning head used by the other experiments.

from opencood.models.point_pillar import PointPillar
from opencood.models.sub_modules.v2xverse_planning_head_mean import (
    V2XVerseMeanPlanningHead,
)


class PointPillarV2XverseMean(PointPillar):
    """Same frozen-backbone planner graph, uniform spatial mean pooling."""

    def __init__(self, args):
        super(PointPillarV2XverseMean, self).__init__(args)
        if not self.use_planning_head:
            return
        planning_args = args.get('planning_head', {})
        planner_in = planning_args.get('feature_dir', 128)
        # Drop the attention head constructed by PointPillar.__init__.
        self.planning_head = V2XVerseMeanPlanningHead(
            feature_dir=planner_in,
            input_frame=self.input_frame,
            output_points=planning_args.get('num_waypoints', 10),
            occupancy_channels=self.occupancy_channels,
        )
