import argparse
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_num_waypoints(df):
    k = 1
    while (
        f"pred_wp{k}_x" in df.columns and
        f"pred_wp{k}_y" in df.columns and
        f"gt_wp{k}_x" in df.columns and
        f"gt_wp{k}_y" in df.columns
    ):
        k += 1
    return k - 1


def get_xy(row, prefix, num_wp):
    xs = []
    ys = []
    for k in range(1, num_wp + 1):
        xs.append(float(row[f"{prefix}_wp{k}_x"]))
        ys.append(float(row[f"{prefix}_wp{k}_y"]))
    return np.array(xs), np.array(ys)


def get_index_text(row):
    if "dataset_idx" in row.index:
        return "dataset_idx", int(row["dataset_idx"])
    if "sample_idx" in row.index:
        return "dataset_idx", int(row["sample_idx"])
    return "row", int(row.name)


def plot_one(ax, row, label, num_wp):
    gt_x, gt_y = get_xy(row, "gt", num_wp)
    pred_x, pred_y = get_xy(row, "pred", num_wp)

    ade = float(row["ade"])
    fde = float(row["fde"])

    idx_name, idx_val = get_index_text(row)

    ax.plot(gt_x, gt_y, marker="o", label="GT")
    ax.plot(pred_x, pred_y, marker="o", label="Pred")
    ax.scatter([0], [0], marker="x", s=60, color="black", label="Ego now")

    ax.set_title(f"{label}\n{idx_name}={idx_val}, ADE={ade:.3f}, FDE={fde:.3f}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axis("equal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument(
        "--rank_by",
        choices=["ade", "fde"],
        default="ade"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    if len(df) == 0:
        raise ValueError("CSV is empty.")

    num_wp = get_num_waypoints(df)
    if num_wp == 0:
        raise ValueError("No waypoint columns found. Expected pred_wp1_x, gt_wp1_x, etc.")

    df_sorted = df.sort_values(args.rank_by, ascending=True).reset_index(drop=True)

    best = df_sorted.iloc[0]
    median = df_sorted.iloc[len(df_sorted) // 2]
    worst = df_sorted.iloc[-1]

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Planning trajectories by {args.rank_by.upper()}")

    plot_one(axes[0], best, "Best frame", num_wp)
    plot_one(axes[1], median, "Median frame", num_wp)
    plot_one(axes[2], worst, "Worst frame", num_wp)

    axes[0].legend(loc="best")

    plt.tight_layout()
    plt.savefig(args.out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", args.out_path)
    print("Rows:", len(df))
    print("Waypoints:", num_wp)
    print("Ranked by:", args.rank_by)


if __name__ == "__main__":
    main()