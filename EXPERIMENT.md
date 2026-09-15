# Experiments

Planner change: V2XVerse uniform spatial mean pooling → learned spatial attention pooling (`256 → 64 → 1`, softmax over H×W). Velocity is **not** fed into the planner in any experiment. Experiments 3–4 are the original mean-pool baselines (det-only vs velocity backbone).

Env: `conda activate v2xreal`

---

# Experiment 1 — Backbone without velocity

Detection-only dual-frame backbone (done), then frozen-backbone attention planner (done, epoch 50).

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
| Planning | ADE / FDE | 0.7655 / 1.6115 |

---

# Experiment 2 — Backbone with velocity

Det+velocity dual-frame backbone (done), then frozen-backbone attention planner (done, epoch 50).

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
| Planning | ADE / FDE | 0.6522 / 1.4261 |

---

# Experiment 3 — Original V2Xverse planner (no-velocity backbone)

Frozen-backbone mean-pool planner (pre-attention V2Xverse head) on the detection-only backbone. Single process on GPU 0; batch 2. CPU affinity `0-11,40-51` (GPU 0 NUMA, clear of the other jobs).

**Backbone**
```bash
cd /media/Disk2/OpenCOOD_vamsi_2
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_x2_det_only.yaml --model_dir /home/project/x2_detection_only
```

**Planner**
```bash
test ! -e /home/project/path_v2xverse_det_only && CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_v2xverse_mean_planner.py
```

| | Backbone | Planner |
|--|----------|---------|
| Config | `point_pillar_early_fusion_x2_det_only.yaml` | `point_pillar_early_fusion_baseline_det_only_mean.yaml` |
| Output | `x2_detection_only` | `path_v2xverse_det_only` |

| Task | Metric | Value |
|------|--------|-------|
| Detection | AP@0.3 | 0.90 |
| Detection | AP@0.5 | 0.89 |
| Detection | AP@0.7 | 0.85 |
| Planning | ADE / FDE | 0.5919 / 1.2634 |

---

# Experiment 4 — Original V2Xverse planner (velocity backbone)

Frozen-backbone mean-pool planner (pre-attention V2Xverse head) on the det+velocity backbone. Same recipe as Experiment 3, but pretrained from `x2_multiframe`. Checkpoint: `/home/project/path_v2xverse` (epoch 50); ADE/FDE from eval copy `path_v2xverse_vamsi_eval_08_23` (`n=2170`).

**Backbone**
```bash
cd /media/Disk2/OpenCOOD_vamsi_2
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_x2.yaml --model_dir /home/project/x2_multiframe
```

**Planner** (historical single-GPU run; head was mean-pool in `point_pillar` before the attention change)
```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_baseline.yaml --model_dir /home/project/path_v2xverse --pretrained_dir /home/project/x2_multiframe
```

| | Backbone | Planner |
|--|----------|---------|
| Config | `point_pillar_early_fusion_x2.yaml` | `point_pillar_early_fusion_baseline.yaml` |
| Output | `x2_multiframe` | `path_v2xverse` |

| Task | Metric | Value |
|------|--------|-------|
| Detection | AP@0.3 | 0.92 |
| Detection | AP@0.5 | 0.91 |
| Detection | AP@0.7 | 0.85 |
| Velocity | Speed MAE | 0.549 m/s |
| Velocity | Speed RMSE | 0.878 m/s |
| Planning | ADE / FDE | 0.6403 / 1.3899 |
