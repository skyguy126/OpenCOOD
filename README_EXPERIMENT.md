# Planning Results: Frozen Backbone + V2XVerse Planner

**Status: TODO** — detection AP must be re-evaluated on a shared validate split before the table below is final (Exp 2 AP was previously measured on the OPV2V test set; Exp 0–1 used validate).

**Dataset:** OPV2V. Backbones and planners are trained on the **train** split. Detection AP is reported on the **validate** split (not held out from training hyperparameter choices). Planning ADE / FDE is reported on the **test** split (`n=2170`), which is held out and never used for training.

This note compares three setups that share the same V2XVerse-style planning module. In each case the perception backbone is frozen, and only the planner is trained. The planner uses the original 6-channel occupancy representation (no speed channel is passed into planning).

The series isolates backbone frame count and velocity training. Experiments 1–2 are the dual-frame pair (same architecture; only velocity training differs). Experiment 0 is the single-frame control:

0. **Single-frame detection-only** → baseline without velocity and without dual-frame input
1. **Dual-frame detection-only** → add dual-frame temporal fusion (still no velocity head)
2. **Dual-frame with velocity** → same dual-frame architecture, backbone also trained with velocity estimation (speed still not fed to the planner)

**Metric:** ADE / FDE (meters). Lower is better. Planning eval `n=2170` (held-out test).

| # | Backbone | Frames | Velocity head | AP @ 0.3 / 0.5 / 0.7 | Speed MAE / RMSE | ADE / FDE |
|---|----------|--------|---------------|----------------------|------------------|-----------|
| 0 | Detection-only | single | no | 0.90 / 0.89 / 0.83 | — | 0.7089 / 1.5281 |
| 1 | Detection-only | dual | no | 0.90 / 0.89 / 0.85 | — | 0.5919 / 1.2634 |
| 2 | Detection + velocity | dual | yes (unused by planner) | 0.92 / 0.91 / 0.85 | 0.549 / 0.878 m/s | 0.6403 / 1.3899 |

---

## 0. Single-frame detection-only backbone + V2XVerse planner

Baseline without velocity and without dual-frame input. The backbone is a single-frame detection-only PointPillar (`dual_frame: false`, no velocity head). Its weights are frozen. The V2XVerse planning module is trained on top.

![Exp 0 flow: single-frame detection-only backbone + V2XVerse planner](docs/figures/planning_exp0_single_frame_det_only.jpg)

Frozen backbone itself (no planner):

| Task | Metric | Value |
|------|--------|-------|
| Detection | AP @ 0.3 / 0.5 / 0.7 | 0.90 / 0.89 / 0.83 |
| Velocity | Speed MAE / RMSE | — |

Planner (frozen backbone + V2XVerse planner):

| Metric | Value |
|--------|-------|
| ADE    | 0.7089 |
| FDE    | 1.5281 |

---

## 1. Dual-frame detection-only backbone + V2XVerse planner

Same as Experiment 0, except the frozen backbone uses dual-frame input and temporal fusion (still no velocity head). Dual-frame is kept so Experiments 1 and 2 share the same architecture.

![Exp 1 flow: dual-frame detection-only backbone + V2XVerse planner](docs/figures/planning_exp1_det_only_flow.jpg)

Frozen backbone itself (no planner):

| Task | Metric | Value |
|------|--------|-------|
| Detection | AP @ 0.3 / 0.5 / 0.7 | 0.90 / 0.89 / 0.85 |
| Velocity | Speed MAE / RMSE | — |

Planner (frozen backbone + V2XVerse planner):

| Metric | Value |
|--------|-------|
| ADE    | 0.5919 |
| FDE    | 1.2634 |

---

## 2. Dual-frame velocity backbone + V2XVerse planner

Same dual-frame architecture as Experiment 1, but the backbone was also trained with velocity estimation. Its weights are frozen. The planner still does not receive predicted speed.

![Exp 2 flow: dual-frame velocity backbone + V2XVerse planner (speed unused)](docs/figures/planning_exp2_velocity_backbone_flow.jpg)

Frozen backbone itself (no planner):

| Task | Metric | Value |
|------|--------|-------|
| Detection | AP @ 0.3 / 0.5 / 0.7 | 0.92 / 0.91 / 0.85 |
| Velocity | Speed MAE / RMSE | 0.549 / 0.878 m/s |

Planner (frozen backbone + V2XVerse planner):

| Metric | Value |
|--------|-------|
| ADE    | 0.6403 |
| FDE    | 1.3899 |

---

**Takeaway:** Dual-frame temporal fusion is the largest win for planning (Exp 0 → 1: ADE 0.709 → 0.592) at nearly matched detection AP. Adding velocity training on that dual-frame backbone without feeding speed to the planner hurts planning (Exp 1 → 2: ADE 0.592 → 0.640) even though detection AP rises slightly. Best no-motion planner here remains the dual-frame detection-only backbone.
