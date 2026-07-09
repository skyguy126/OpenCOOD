"""
Examples:
python opencood/tools/plot_planning_trajectories.py \
  --csv_path /home/project/path/planning_eval_full.csv \
  --output_path /home/project/path/planning_trajectory_samples.png

python opencood/tools/plot_planning_trajectories.py \
  --csv_path /home/project/path/planning_eval_full.csv \
  --output_path /home/project/path/planning_trajectory_samples_fde.png \
  --metric fde
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) == 0:
        raise RuntimeError(f"CSV has no rows: {csv_path}")

    return rows


def row_metric(row, metric_name):
    return float(row[metric_name])


def select_rows(rows, metric_name):
    sorted_rows = sorted(rows, key=lambda row: row_metric(row, metric_name))
    best_row = sorted_rows[0]
    worst_row = sorted_rows[-1]
    median_row = sorted_rows[len(sorted_rows) // 2]
    return [
        ("best", best_row),
        ("median", median_row),
        ("worst", worst_row),
    ]


def extract_waypoints(row, prefix):
    return np.array(
        [[float(row[f"{prefix}_wp{i}_x"]), float(row[f"{prefix}_wp{i}_y"])]
         for i in range(6)],
        dtype=np.float32,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--metric", choices=["ade", "fde"], default="ade")
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    selected_rows = select_rows(rows, args.metric)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (label, row) in zip(axes, selected_rows):
        gt = extract_waypoints(row, "gt")
        pred = extract_waypoints(row, "pred")

        ax.plot(gt[:, 0], gt[:, 1], "-o", label="GT")
        ax.plot(pred[:, 0], pred[:, 1], "-o", label="Pred")
        ax.scatter([0.0], [0.0], marker="x", color="black", label="Ego now")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(
            f"{label.capitalize()} frame\n"
            f"dataset_idx={row['dataset_idx']}, "
            f"ADE={float(row['ade']):.3f}, FDE={float(row['fde']):.3f}"
        )

    axes[0].legend()
    fig.suptitle(f"Planning trajectories by {args.metric.upper()}")
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=200)
    plt.close(fig)

    print("========== PLANNING TRAJECTORY SAMPLES ==========")
    for label, row in selected_rows:
        print(
            f"{label}: dataset_idx={row['dataset_idx']}, "
            f"ADE={float(row['ade']):.6f}, "
            f"FDE={float(row['fde']):.6f}"
        )
    print(f"Saved plot to: {args.output_path}")


if __name__ == "__main__":
    main()
