"""
Example:
python opencood/tools/plot_planning_curve.py \
  --csv_path /home/project/path/planning_eval_full.csv \
  --output_path /home/project/path/planning_waypoint_curve.png
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--title", default="Per-waypoint Mean Planning Error")
    args = parser.parse_args()

    waypoint_errors = [[] for _ in range(6)]

    with open(args.csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise RuntimeError(f"CSV has no rows: {args.csv_path}")

    for row in rows:
        for i in range(6):
            waypoint_errors[i].append(float(row[f"wp{i}_error"]))

    mean_errors = np.array([np.mean(v) for v in waypoint_errors], dtype=np.float32)
    waypoint_idx = np.arange(6)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(waypoint_idx, mean_errors, marker="o", linewidth=2)
    plt.xticks(waypoint_idx)
    plt.xlabel("Waypoint index")
    plt.ylabel("Mean L2 error (m)")
    plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=200)
    plt.close()

    print("========== PER-WAYPOINT MEAN ERROR ==========")
    print(f"Rows used: {len(rows)}")
    print(f"Mean errors: {mean_errors.tolist()}")
    print(f"Saved plot to: {args.output_path}")


if __name__ == "__main__":
    main()
