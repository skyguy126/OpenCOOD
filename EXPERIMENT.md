# Experiments

Planner change: V2XVerse uniform spatial mean pooling → learned spatial attention pooling (`256 → 64 → 1`, softmax over H×W). Velocity is **not** fed into the planner in either experiment.

Env: `conda activate v2xreal`

---

# Experiment 1 — Backbone without velocity

Detection-only dual-frame backbone (done), then frozen-backbone attention planner (to retrain).

**Backbone**
```bash
cd /media/Disk2/OpenCOOD_vamsi_2
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_x2_det_only.yaml --model_dir /home/project/x2_detection_only
```

**Planner**
```bash
test ! -e /home/project/path_v2xverse_det_only_attn && CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_baseline_det_only_attn.yaml --pretrained_dir /home/project/x2_detection_only --model_dir /home/project/path_v2xverse_det_only_attn
```

| | Backbone | Planner |
|--|----------|---------|
| Config | `point_pillar_early_fusion_x2_det_only.yaml` | `point_pillar_early_fusion_baseline_det_only_attn.yaml` |
| Output | `x2_detection_only` | `path_v2xverse_det_only_attn` |

| Task | Metric | Value |
|------|--------|-------|
| Detection | AP@0.3 | 0.90 |
| Detection | AP@0.5 | 0.89 |
| Detection | AP@0.7 | 0.85 |
| Planning | ADE / FDE | TBD |

---

# Experiment 2 — Backbone with velocity

Det+velocity dual-frame backbone (done), then frozen-backbone attention planner (to retrain).

**Backbone**
```bash
cd /media/Disk2/OpenCOOD_vamsi_2
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_x2.yaml --model_dir /home/project/x2_multiframe
```

**Planner**
```bash
test ! -e /home/project/path_v2xverse_attn && CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_baseline_attn.yaml --pretrained_dir /home/project/x2_multiframe --model_dir /home/project/path_v2xverse_attn
```

| | Backbone | Planner |
|--|----------|---------|
| Config | `point_pillar_early_fusion_x2.yaml` | `point_pillar_early_fusion_baseline_attn.yaml` |
| Output | `x2_multiframe` | `path_v2xverse_attn` |

| Task | Metric | Value |
|------|--------|-------|
| Detection | AP@0.3 | 0.92 |
| Detection | AP@0.5 | 0.91 |
| Detection | AP@0.7 | 0.85 |
| Velocity | Speed MAE | 0.549 m/s |
| Velocity | Speed RMSE | 0.878 m/s |
| Planning | ADE / FDE | TBD (attention planner retrain) |
