# Experiment 1 — With Velocity Prediction (Ablation)

Setting: dual-frame PointPillar backbone trained with speed regression, then V2XVerse planner trained with **frozen** backbone.

**Note:** Predicted velocity is **not** used by the planner (BEV features + occupancy + planning target only).

## Checkpoints

| Component | Path |
|-----------|------|
| Backbone (det + velocity) | `/home/project/x2_multiframe/net_epoch15.pth` |
| Planner | `/home/project/path_v2xverse/net_epoch50.pth` |

## Metrics (OPV2V test)

| Task | Metric | Value |
|------|--------|-------|
| Detection | AP@0.5 | 0.91 |
| Detection | AP@0.7 | 0.85 |
| Velocity | Speed MAE | 0.549 m/s |
| Velocity | Speed RMSE | 0.878 m/s |
| Planning | ADE (mean) | 0.640 |
| Planning | FDE (mean) | 1.390 |

Eval sources: detection + ADE/FDE from `path_v2xverse_vamsi_eval_08_23`; velocity from `x2_multiframe_path/velocity_eval_iou03_15.txt` (epoch-15 speed head). Scripts: OpenCOOD AP inference, `opencood/eval/velocity_eval.py`, `opencood/eval/planning_eval.py`.
