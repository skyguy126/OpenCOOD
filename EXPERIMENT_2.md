# Experiment 2 — Velocity Wired Into Planner (Ablation)

Same setup as Experiment 1, except predicted speed from the frozen detection head is fed into the V2XVerse planner as occupancy channel 6.

## Checkpoints (train into a new dir; does not overwrite Exp 1)

| Component | Path |
|-----------|------|
| Backbone (det + velocity) | `/home/project/x2_multiframe/net_epoch15.pth` |
| Planner (this run) | `/home/project/path_v2xverse_vel/` |

## Config

`opencood/hypes_yaml/point_pillar_early_fusion_baseline_vel.yaml`  
(`use_velocity_in_planning: true`; all other train hyperparams match Exp 1)

## Metrics to report (same eval scripts as Exp 1)

Detection AP@0.5 / AP@0.7 · Velocity MAE/RMSE · Planning ADE / FDE
