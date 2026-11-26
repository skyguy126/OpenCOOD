"""Generate the planning-ablation diagrams used in README_EXPERIMENT.md.

Usage: python docs/figures/make_planning_figures.py
Writes PNGs next to this script.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- palette -------------------------------------------------------------
INK = "#1f2a44"
MUTED = "#6b7280"
BOX_FACE = "#eef2f8"
BOX_EDGE = "#9aa7bd"
TRAIN_FACE = "#fdebd3"
TRAIN_EDGE = "#d9822b"
DELTA_FACE = "#dff3e4"
DELTA_EDGE = "#2e8b57"
SPEED = "#c0392b"

# ---- experiment definitions ---------------------------------------------
# Columns of the pipeline (early fusion and outputs are identical in every
# experiment, so they are folded into the column headers / end cap).
COLS = ["Input LiDAR", "Frozen PointPillar", "Planner input", "V2XVerse planner"]

EXPS = [
    dict(
        id=0,
        title="Single-frame det-only",
        cells=["single frame", "det only", "6-ch, no speed", "trainable\n5 hist frames"],
        delta=None,
        delta_label="baseline",
        speed_used=False,
        ade="0.709",
        fde="1.528",
    ),
    dict(
        id=1,
        title="Dual-frame det-only",
        cells=["dual frame", "det only", "6-ch, no speed", "trainable\n5 hist frames"],
        delta={0},
        delta_label="+ dual-frame input",
        speed_used=False,
        ade="0.592",
        fde="1.263",
    ),
    dict(
        id=2,
        title="Dual-frame + velocity head (speed unused)",
        cells=["dual frame", "det + velocity", "6-ch, speed unused", "trainable\n5 hist frames"],
        delta={1},
        delta_label="+ velocity head",
        speed_used=False,
        ade="0.640",
        fde="1.390",
    ),
    dict(
        id=3,
        title="Dual-frame + velocity head (speed used)",
        cells=["dual frame", "det + velocity", "8-ch, conf × speed", "trainable\n5 hist frames"],
        delta={2},
        delta_label="+ speed into planner",
        speed_used=True,
        ade="0.771",
        fde="1.620",
    ),
]

# ---- layout constants (data units) --------------------------------------
BOX_W = 2.25
BOX_H = 0.78
GAP = 0.42
LABEL_W = 3.3   # left column with experiment id / delta
RES_W = 1.7     # right column with ADE / FDE
ROW_H = 1.12


def col_x(i):
    return LABEL_W + i * (BOX_W + GAP)


def box(ax, x, y, w, h, text, face, edge, bold=False, lw=1.2, fs=9.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.08",
            fc=face, ec=edge, lw=lw,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, color=INK,
        fontweight="bold" if bold else "normal", linespacing=1.15,
    )


def arrow(ax, x0, y0, x1, y1, color=INK, ls="-", lw=1.1, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=9, color=color, lw=lw,
            linestyle=ls, connectionstyle=f"arc3,rad={rad}",
            shrinkA=0, shrinkB=0,
        )
    )


def draw_row(ax, exp, y, best_ade=None, show_label=True):
    """Draw one pipeline row with its bottom edge at y."""
    yc = y + BOX_H / 2
    if show_label:
        ax.text(0.32, yc + 0.14, f"Exp {exp['id']}", ha="left", va="center",
                fontsize=11, fontweight="bold", color=INK)
        ax.text(0.32, yc - 0.18, exp["delta_label"], ha="left", va="center",
                fontsize=9, color=DELTA_EDGE if exp["delta"] else MUTED,
                fontweight="bold" if exp["delta"] else "normal",
                style="normal" if exp["delta"] else "italic")

    for i, txt in enumerate(exp["cells"]):
        x = col_x(i)
        changed = exp["delta"] is not None and i in exp["delta"]
        trainable = i == len(COLS) - 1
        if changed:
            face, edge, bold, lw = DELTA_FACE, DELTA_EDGE, True, 2.0
        elif trainable:
            face, edge, bold, lw = TRAIN_FACE, TRAIN_EDGE, False, 1.2
        else:
            face, edge, bold, lw = BOX_FACE, BOX_EDGE, False, 1.2
        box(ax, x, y, BOX_W, BOX_H, txt, face, edge, bold=bold, lw=lw)
        if i > 0:
            arrow(ax, x - GAP + 0.04, yc, x - 0.04, yc)

    # speed path: backbone -> planner (dashed, below the row)
    if exp["speed_used"]:
        x0 = col_x(1) + BOX_W * 0.5
        x1 = col_x(3) + BOX_W * 0.5
        arrow(ax, x0, y - 0.02, x1, y - 0.02,
              color=SPEED, ls=(0, (3, 2)), lw=1.3, rad=0.12)
        ax.text((x0 + x1) / 2, y - 0.44, "predicted speed (confidence-gated)",
                ha="center", va="top", fontsize=8.5, color=SPEED, style="italic")

    # results
    xr = col_x(len(COLS)) + 0.05
    arrow(ax, xr - GAP + 0.04, yc, xr - 0.04, yc)
    is_best = best_ade is not None and exp["ade"] == best_ade
    ax.text(xr + 0.05, yc + 0.14, f"ADE {exp['ade']}", ha="left", va="center",
            fontsize=10, color=INK, fontweight="bold" if is_best else "normal")
    ax.text(xr + 0.05, yc - 0.18, f"FDE {exp['fde']}", ha="left", va="center",
            fontsize=10, color=INK, fontweight="bold" if is_best else "normal")


def headers(ax, y, legend=True):
    # legend in the otherwise empty top-left corner
    for k, (face, edge, label) in enumerate(
        [(DELTA_FACE, DELTA_EDGE, "changed vs. prev. exp"), (TRAIN_FACE, TRAIN_EDGE, "trained")]
    ):
        if not legend:
            break
        lx, ly = 0.32, y + 0.12 - k * 0.24
        ax.add_patch(FancyBboxPatch((lx, ly - 0.08), 0.22, 0.16,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=face, ec=edge, lw=1.4))
        ax.text(lx + 0.3, ly, label, ha="left", va="center", fontsize=8, color=MUTED)
    for i, name in enumerate(COLS):
        ax.text(col_x(i) + BOX_W / 2, y, name, ha="center", va="bottom",
                fontsize=9, color=MUTED, fontweight="bold")
    ax.text(col_x(len(COLS)) + 0.1, y, "Test (m) ↓", ha="left", va="bottom",
            fontsize=9, color=MUTED, fontweight="bold")


def total_width():
    return col_x(len(COLS)) + RES_W


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def overview():
    n = len(EXPS)
    W = total_width()
    top_pad = 0.65  # room for column headers
    bottom_pad = 0.62  # room for the speed arc under the last row
    H = n * ROW_H + top_pad
    scale = 0.85
    fig, ax = plt.subplots(figsize=(W * scale, (H + bottom_pad) * scale))
    ax.set_xlim(0, W)
    ax.set_ylim(-bottom_pad, H)
    ax.axis("off")

    best = min(e["ade"] for e in EXPS)
    # rows top -> bottom in chronological order
    ys = [H - top_pad - (k + 1) * ROW_H + (ROW_H - BOX_H) for k in range(n)]
    headers(ax, H - top_pad + 0.12 + (ROW_H - BOX_H))
    for exp, y in zip(EXPS, ys):
        draw_row(ax, exp, y, best_ade=best)

    # chronological spine on the left
    for k in range(n - 1):
        y_from = ys[k] + 0.02
        y_to = ys[k + 1] + BOX_H - 0.02
        arrow(ax, 0.1, y_from, 0.1, y_to, color=DELTA_EDGE, lw=1.4)

    save(fig, "planning_ablation_overview.png")


def single(exp, best):
    W = total_width()
    bottom = 0.62 if exp["speed_used"] else 0.05
    H = BOX_H + 0.45
    scale = 0.85
    fig, ax = plt.subplots(figsize=(W * scale, (H + bottom) * scale))
    ax.set_xlim(0, W)
    ax.set_ylim(-bottom, H)
    ax.axis("off")
    y = 0.0
    headers(ax, BOX_H + 0.1, legend=False)
    draw_row(ax, exp, y, best_ade=best)
    save(fig, f"planning_exp{exp['id']}.png")


if __name__ == "__main__":
    overview()
    best = min(e["ade"] for e in EXPS)
    for e in EXPS:
        single(e, best)
