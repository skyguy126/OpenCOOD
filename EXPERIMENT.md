# Experiments

Ablation for velocity-conditioned planning: **baseline planner** = V2XVerse uniform spatial mean pooling; **attention planner** = learned spatial attention pooling (`256 → 64 → 1`, softmax over H×W, zero-init scorer). **Velocity in occupancy** = frozen-backbone predicted speed as planner channel 6.

Env: `conda activate v2xreal`

| ID | Backbone | Planner | ADE / FDE |
|----|----------|---------|-----------|
| 1 | detection only backbone | baseline planner (mean pool; no speed channel) | 0.5919 / 1.2634 |
| 2 | velocity backbone | baseline planner (mean pool; **no** speed channel) | 0.6403 / 1.3899 |
| 3 | velocity backbone | baseline planner (mean pool; **with** speed channel) | TODO |
| 4 | velocity backbone | attention planner (spatial attn; **with** speed channel) | 1.3394 / 2.4652 |

Rows 2–4 share the same frozen velocity backbone (`x2_multiframe`). They differ only in the planner head: mean vs attention, and whether predicted speed is occupancy channel 6.


---

# Experiment 1 — Detection only backbone + baseline planner

Frozen-backbone mean-pool planner on the detection-only backbone. No velocity in planner occupancy.

**Backbone**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_x2_det_only.yaml --model_dir /home/project/x2_detection_only
```

**Planner**
```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_v2xverse_mean_planner.py
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

# Experiment 2 — Velocity backbone + baseline planner (no velocity occupancy)

Frozen-backbone mean-pool planner on the det+velocity backbone. Velocity is **not** fed into planner occupancy (6 channels). Checkpoint `/home/project/path_v2xverse`; ADE/FDE from `path_v2xverse_vamsi_eval_08_23` (`n=2170`).

**Backbone**
```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_x2.yaml --model_dir /home/project/x2_multiframe
```

**Planner**
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

# Experiment 3 — Velocity backbone + baseline planner + velocity occupancy

Same as Experiment 2, but predicted speed is wired into planner occupancy (channel 6). Control for Experiment 4.

**Backbone:** reuse `x2_multiframe` (Experiment 2).

**Planner** (running)
```bash
CUDA_VISIBLE_DEVICES=2 python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_baseline_mean_vel.yaml --pretrained_dir /home/project/x2_multiframe --model_dir /home/project/path_v2xverse_mean_vel --cpu_affinity 20-39
```

| | Backbone | Planner |
|--|----------|---------|
| Config | `point_pillar_early_fusion_x2.yaml` | `point_pillar_early_fusion_baseline_mean_vel.yaml` |
| Output | `x2_multiframe` | `path_v2xverse_mean_vel` |

| Task | Metric | Value |
|------|--------|-------|
| Detection | AP@0.3 | 0.92 |
| Detection | AP@0.5 | 0.91 |
| Detection | AP@0.7 | 0.85 |
| Velocity | Speed MAE | 0.549 m/s |
| Velocity | Speed RMSE | 0.878 m/s |
| Planning | ADE / FDE | TODO |

---

# Experiment 4 — Velocity backbone + attention planner + velocity occupancy

Method: attention planner with predicted speed in occupancy on the velocity backbone. Eval: epoch 50, held-out test (`n=2170`), `planning_only_eval_ep50.csv`.

**Backbone:** reuse `x2_multiframe` (Experiment 2).

**Planner**
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 --master_port=29511 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion_baseline_attn_vel.yaml --pretrained_dir /home/project/x2_multiframe --model_dir /home/project/path_v2xverse_attn_vel --cpu_affinity 0-19
```

**Eval**
```bash
CUDA_VISIBLE_DEVICES=1 python -u opencood/eval/planning_eval.py --model_dir /home/project/path_v2xverse_attn_vel --save_csv --csv_name planning_only_eval_ep50.csv 2>&1 | tee /home/project/path_v2xverse_attn_vel/planning_only_eval_ep50_output.txt
```

| | Backbone | Planner |
|--|----------|---------|
| Config | `point_pillar_early_fusion_x2.yaml` | `point_pillar_early_fusion_baseline_attn_vel.yaml` |
| Output | `x2_multiframe` | `path_v2xverse_attn_vel` |

| Task | Metric | Value |
|------|--------|-------|
| Detection | AP@0.3 | 0.92 |
| Detection | AP@0.5 | 0.91 |
| Detection | AP@0.7 | 0.85 |
| Velocity | Speed MAE | 0.549 m/s |
| Velocity | Speed RMSE | 0.878 m/s |
| Planning | ADE / FDE | 1.3394 / 2.4652 |
