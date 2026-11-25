# Experiments

We study confidence-aware motion conditioning for cooperative BEV planning. Rather than injecting detector-regressed speed as a confidence-normalized dense field, the revised planner preserves detector objectness when constructing motion features. This enables a controlled ablation of gated motion, explicit uncertainty, and residual spatial attention.

**Proposed contribution:** confidence-aware motion conditioning of a V2XVerse-style cooperative planner, where detector objectness is retained as part of the motion representation rather than normalized away before planning.

All three planner experiments reuse the frozen velocity backbone checkpoint `/home/project/x2_multiframe/net_epoch15.pth`. They share the same training recipe, optimizer, loss, dataset, epochs, batch size, frozen backbone, and evaluation procedure. The **only** intentional differences are planner motion representation and, in Experiment 3, spatial pooling.

Env: `conda activate v2xreal`

## Main ablation

| ID        | Motion representation    | Pooling            | ADE / FDE       | Status                                      |
| --------- | ------------------------ | ------------------ | --------------- | ------------------------------------------- |
| Reference | none                     | mean               | 0.6403 / 1.3899 | completed historical baseline               |
| 1         | confidence-gated speed   | mean               | 1.0599 / 2.0063 | completed (`n=2170`, epoch 50)              |
| 2         | confidence + gated speed | mean               | 0.7706 / 1.6203 | completed (`n=2170`, epoch 50)              |
| 3         | confidence + gated speed | residual attention | 0.7191 / 1.5043 | completed (`n=2170`, epoch 50)              |

### Experiment 1

Tests whether velocity becomes useful when unreliable/background regression is attenuated by detector confidence (`motion_mode: gated_speed`, 7 occupancy channels).

### Experiment 2

Tests whether exposing confidence separately helps the planner distinguish low speed from low certainty (`motion_mode: confidence_gated_speed`, 8 occupancy channels).

### Experiment 3

Tests whether spatial attention adds value after the motion representation has been corrected, while retaining mean pooling through a residual/convex mixture (`pooling: residual_attention`, same 8-channel motion as Experiment 2).

Controlled differences:

* **Reference vs 1:** same mean pooling; add confidence-gated speed channel
* **1 vs 2:** same mean pooling; Exp 2 also supplies explicit detector confidence
* **2 vs 3:** same 8-channel motion; mean vs residual attention pooling

---

## Previous diagnostic results

These runs motivated the revised study. They remain documented as diagnostic evidence and are **not** retrained.

| Diagnostic | Description | ADE / FDE |
|------------|-------------|-----------|
| Velocity backbone + mean pool + no motion | Original 6-channel V2XVerse occupancy | 0.6403 / 1.3899 |
| Naive confidence-normalized speed channel | `sum(score * speed) / sum(score)` as occupancy channel 6 | 0.7395 / 1.5968 |
| Naive speed + replacement attention | Same normalized speed + hard attention pooling | 1.3394 / 2.4652 |

The naive speed raster divided by summed detector confidence and could therefore expose unsupervised background regression values to the planner. That representation is **not** used by Experiments 1–3.

### Frozen-backbone no-motion baselines

Same V2XVerse mean planner and original 6-channel occupancy (no speed fed to planning). Differences are backbone frame count and whether the backbone was trained with velocity.

| Backbone | Frames | Velocity head | Detection AP @ 0.3 / 0.5 / 0.7 | ADE / FDE | Paths |
|----------|--------|---------------|-------------------------------|-----------|-------|
| Det-only single-frame | single (`dual_frame: false`) | no | 0.90 / 0.89 / 0.83 | 0.7089 / 1.5281 | `x2_detection_single` + `path_v2xverse_det_only_single` |
| Det-only multi-frame | dual | no | 0.90 / 0.89 / 0.85 | 0.5919 / 1.2634 | `x2_detection_only` + `path_v2xverse_det_only` |
| Velocity multi-frame | dual | yes | 0.92 / 0.91 / 0.85 | 0.6403 / 1.3899 | `x2_multiframe` + `path_v2xverse` / `path_v2xverse_vamsi_eval_08_23` |

Single-frame det-only planning eval: epoch 50, `n=2170` (`planning_only_eval_ep50.csv`). Detection AP from `inference.py` on `x2_detection_single` epoch 30.

---

## Shared training recipe (Experiments 1–3)

```yaml
train_params:
  batch_size: 2
  epoches: 50
  freeze_backbone: true
  pretrained_epoch: 15
  gradient_clip_norm: 10

optimizer:
  core_method: AdamW
  lr: 0.0001
```

* Backbone checkpoint: `--pretrained_dir /home/project/x2_multiframe` (`net_epoch15.pth`)
* Loss: planning-only L1 waypoint loss (unchanged)
* Single-process / single-GPU training (no DDP); the three jobs may run concurrently on separate GPUs
* Motion maps are built from frozen detection heads under `torch.no_grad()`; no gradients into the perception backbone

Motion construction (all new experiments):

```python
confidence, best_anchor = scores.max(dim=1, keepdim=True)
best_speed = speed.gather(1, best_anchor).clamp(0.0, 2.0)
gated_speed = confidence * best_speed  # no division by confidence
```

---

# Experiment 1 — Gated speed + mean pooling

Planner occupancy = original 6 channels + `confidence * predicted_speed` (7 total).

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/point_pillar_planner_gated_speed.yaml \
  --pretrained_dir /home/project/x2_multiframe \
  --model_dir /home/project/path_v2xverse_gated_speed
```

| | Value |
|--|-------|
| Config | `point_pillar_planner_gated_speed.yaml` |
| Output | `/home/project/path_v2xverse_gated_speed` |
| Motion | `gated_speed` (7 ch) |
| Pooling | `mean` |
| ADE / FDE | 1.0599 / 2.0063 |

---

# Experiment 2 — Explicit confidence + gated speed + mean pooling

Planner occupancy = original 6 + detector confidence + `confidence * predicted_speed` (8 total).

```bash
CUDA_VISIBLE_DEVICES=1 python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/point_pillar_planner_conf_motion.yaml \
  --pretrained_dir /home/project/x2_multiframe \
  --model_dir /home/project/path_v2xverse_conf_motion
```

| | Value |
|--|-------|
| Config | `point_pillar_planner_conf_motion.yaml` |
| Output | `/home/project/path_v2xverse_conf_motion` |
| Motion | `confidence_gated_speed` (8 ch) |
| Pooling | `mean` |
| ADE / FDE | 0.7706 / 1.6203 |

---

# Experiment 3 — Confidence-aware motion + residual spatial attention

Same 8-channel motion as Experiment 2. Pooling is a learnable convex blend of mean and attention features with `alpha ≈ 0.10` at initialization (starts near the known-good mean-pool representation).

```bash
CUDA_VISIBLE_DEVICES=2 python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/point_pillar_planner_conf_motion_resattn.yaml \
  --pretrained_dir /home/project/x2_multiframe \
  --model_dir /home/project/path_v2xverse_conf_motion_resattn
```

| | Value |
|--|-------|
| Config | `point_pillar_planner_conf_motion_resattn.yaml` |
| Output | `/home/project/path_v2xverse_conf_motion_resattn` |
| Motion | `confidence_gated_speed` (8 ch) |
| Pooling | `residual_attention` |
| ADE / FDE | 0.7191 / 1.5043 |

---

# Reference — Velocity backbone + mean planner (no motion)

Historical baseline: frozen `x2_multiframe`, original 6-channel V2XVerse occupancy, mean pooling, no velocity input to the planner. ADE/FDE from `path_v2xverse_vamsi_eval_08_23` (`n=2170`).

**Backbone**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_x2.yaml --model_dir /home/project/x2_multiframe
```

**Planner (do not retrain for this ablation)**
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

---

# Baseline — Single-frame detection-only + mean planner (no motion)

Single-frame PointPillar without velocity (`dual_frame: false`, 7D boxes). Frozen backbone + V2XVerse mean planner with original 6-channel occupancy. Does not overwrite `x2_detection_only` / `path_v2xverse_det_only`.

**Backbone**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_det_only_single.yaml --model_dir /home/project/x2_detection_single
```

**Planner**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_planner_det_only_single_mean.yaml --pretrained_dir /home/project/x2_detection_single --model_dir /home/project/path_v2xverse_det_only_single
```

**Eval**
```bash
CUDA_VISIBLE_DEVICES=0 python -u opencood/eval/planning_eval.py --model_dir /home/project/path_v2xverse_det_only_single --save_csv --csv_name planning_only_eval_ep50.csv
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir /home/project/x2_detection_single --fusion_method early
```

| | Backbone | Planner |
|--|----------|---------|
| Config | `point_pillar_early_fusion_det_only_single.yaml` | `point_pillar_planner_det_only_single_mean.yaml` |
| Output | `x2_detection_single` | `path_v2xverse_det_only_single` |

| Task | Metric | Value |
|------|--------|-------|
| Detection | AP@0.3 | 0.90 |
| Detection | AP@0.5 | 0.89 |
| Detection | AP@0.7 | 0.83 |
| Velocity | Speed MAE / RMSE | — |
| Planning | ADE / FDE | 0.7089 / 1.5281 (`n=2170`, epoch 50) |
