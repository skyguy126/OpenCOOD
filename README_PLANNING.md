# Planning Results: Frozen Backbone + V2XVerse Planner

This note compares two setups that share the same V2XVerse-style planning module. In both cases the perception backbone is frozen, and only the planner is trained. The planner uses the original occupancy representation (no speed channel is passed into planning). The only difference is whether the frozen backbone was trained with velocity estimation.

**Metric:** ADE / FDE (meters). Lower is better.

---

## 1. Our backbone without velocity + V2XVerse planner

The backbone is our detection-only model (no velocity head). Its weights are frozen. The V2XVerse planning module is trained on top.

| Metric | Value |
|--------|-------|
| ADE    | 0.5919 |
| FDE    | 1.2634 |

---

## 2. Our backbone with velocity + V2XVerse planner

The backbone is our model trained with velocity estimation. Its weights are frozen. The same V2XVerse planning module is trained on top, still without feeding predicted speed into the planner.

| Metric | Value |
|--------|-------|
| ADE    | 0.6403 |
| FDE    | 1.3899 |

For reference, the velocity backbone itself reports:

| Task | Metric | Value |
|------|--------|-------|
| Detection | AP @ 0.3 / 0.5 / 0.7 | 0.92 / 0.91 / 0.85 |
| Velocity | Speed MAE / RMSE | 0.549 / 0.878 m/s |

---

## Side-by-side

| Setup | Backbone | Velocity in backbone | Velocity into planner | ADE | FDE |
|-------|----------|----------------------|-----------------------|-----|-----|
| 1 | Ours (detection-only) | No | No | 0.5919 | 1.2634 |
| 2 | Ours (with velocity) | Yes | No | 0.6403 | 1.3899 |

**Takeaway:** With the same V2XVerse planner and no motion input to planning, the detection-only frozen backbone outperforms the velocity-trained frozen backbone on ADE and FDE.
