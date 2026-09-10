# Ablation

- **Experiment 1:** frozen detection backbone (detection + velocity) → V2XVerse planning
- **Experiment 2:** frozen detection backbone (detection only) → V2XVerse planning

Velocity is **not** an input to the planner; it only shapes backbone weights during backbone training.

---

# Experiment 1

## Velocity prediction

Dual-frame (`dual_frame: true`): current and previous LiDAR are encoded **separately**, then fused with a temporal conv. Frames are **not** stacked into one point cloud. Speed is the 8th box regression dim (`box_code_size: 8`).

## Metrics (OPV2V test)

| Task | Metric | Value |
|------|--------|-------|
| Detection | AP@0.5 | 0.91 |
| Detection | AP@0.7 | 0.85 |
| Velocity | Speed MAE | 0.549 m/s |
| Velocity | Speed RMSE | 0.878 m/s |
| Planning | ADE (mean) | 0.640 |
| Planning | FDE (mean) | 1.390 |

## Hyperparameters

| | Backbone | Planner |
|--|----------|---------|
| Config | `point_pillar_early_fusion_x2.yaml` | `point_pillar_early_fusion_baseline.yaml` |
| Epochs | 15 (best) / 30 scheduled | 50 |
| Batch size | 2 | 2 |
| Optimizer | Adam, lr 0.002 | AdamW, lr 1e-4 |
| LR schedule | MultiStep [10, 15], γ=0.1 | CosineAnnealingLR, T_max=50 |
| Fusion | Early, `dual_frame: true` | Early, frozen backbone, `planning_only` |
| Loss | cls 1.0, reg 2.0, speed 4.0 | planning L1, weight 1.0 |
| Planner I/O | — | 5 history frames → 10 waypoints |
| Grad clip | — | 10 |

## Paths

| | Path |
|--|------|
| Backbone (best) | `/home/project/x2_multiframe/net_epoch15.pth` |
| Planner (best) | `/home/project/path_v2xverse/net_epoch50.pth` |

## Train commands

```bash
# Backbone (det + velocity)
cd /media/Disk2/OpenCOOD_vamsi_2
python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_x2.yaml \
  --model_dir /home/project/x2_multiframe

# Planner (frozen backbone)
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_baseline.yaml \
  --pretrained_dir /home/project/x2_multiframe \
  --model_dir /home/project/path_v2xverse
```
