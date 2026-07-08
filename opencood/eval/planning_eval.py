# -*- coding: utf-8 -*-
"""
Planning-only evaluation for OpenCOOD + planning head.

Computes:
  - ADE: Average Displacement Error
  - FDE: Final Displacement Error
  - waypoint-wise L2 error

Expected:
  model output:
      output_dict['future_waypoints']          # [B, 6, 2]

  batch GT:
      batch_data['ego']['future_waypoints']    # [B, 6, 2]

Example:
python opencood/tools/eval_planning_only.py \
  --model_dir /home/project/x2_multiframe/ \
  --debug_shapes \
  --max_samples 5

Full:
python opencood/tools/eval_planning_only.py \
  --model_dir /home/project/x2_multiframe/ \
  --save_csv
"""

import argparse
import csv
import os
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def test_parser():
    parser = argparse.ArgumentParser(description="Planning-only ADE/FDE evaluation")

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to trained model directory.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="-1 means evaluate the full dataset.",
    )

    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Save per-sample planning metrics.",
    )

    parser.add_argument(
        "--csv_name",
        type=str,
        default="planning_only_eval.csv",
    )

    parser.add_argument(
        "--debug_shapes",
        action="store_true",
        help="Print batch/model keys and waypoint shapes.",
    )

    parser.add_argument(
        "--allow_truncate",
        action="store_true",
        help=(
            "If prediction and GT have different waypoint counts, "
            "evaluate only the shared first min(N_pred, N_gt) waypoints."
        ),
    )

    return parser.parse_args()


def list_nested_keys(obj: Any, prefix: str = "", max_depth: int = 3) -> List[str]:
    keys = []

    if max_depth < 0:
        return keys

    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            keys.append(name)
            keys.extend(list_nested_keys(v, name, max_depth - 1))

    return keys


def normalize_waypoints(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Normalize waypoint tensor to [B, N, 2].

    Supports:
      [B, N, 2]
      [B, N, >=2]
      [N, 2]
      [B, 2N]
    """
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")

    if x.ndim == 3:
        if x.shape[-1] < 2:
            raise ValueError(f"{name} last dim must be >= 2, got {tuple(x.shape)}")
        return x[..., :2].float()

    if x.ndim == 2:
        if x.shape[-1] == 2:
            return x.unsqueeze(0).float()

        if x.shape[-1] % 2 == 0:
            return x.view(x.shape[0], -1, 2).float()

    raise ValueError(f"Unsupported {name} shape: {tuple(x.shape)}")


@torch.no_grad()
def compute_planning_metrics(
    pred_wp: torch.Tensor,
    gt_wp: torch.Tensor,
    allow_truncate: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    pred_wp: [B, N, 2]
    gt_wp:   [B, N, 2]
    """
    pred_wp = normalize_waypoints(pred_wp, "pred_wp")
    gt_wp = normalize_waypoints(gt_wp, "gt_wp")

    if pred_wp.shape[0] != gt_wp.shape[0]:
        raise ValueError(
            f"Batch size mismatch: pred={tuple(pred_wp.shape)}, gt={tuple(gt_wp.shape)}"
        )

    if pred_wp.shape[1] != gt_wp.shape[1]:
        if allow_truncate:
            n = min(pred_wp.shape[1], gt_wp.shape[1])
            pred_wp = pred_wp[:, :n, :]
            gt_wp = gt_wp[:, :n, :]
        else:
            raise ValueError(
                f"Waypoint count mismatch: pred={tuple(pred_wp.shape)}, "
                f"gt={tuple(gt_wp.shape)}. "
                f"If this is intentional, rerun with --allow_truncate."
            )

    if pred_wp.shape != gt_wp.shape:
        raise ValueError(
            f"Waypoint shape mismatch: pred={tuple(pred_wp.shape)}, "
            f"gt={tuple(gt_wp.shape)}"
        )

    # L2 distance per waypoint.
    # Unit is meters if your GT future_waypoints are xy displacement in meters.
    dist = torch.linalg.norm(pred_wp - gt_wp, dim=-1)  # [B, N]

    ade = dist.mean(dim=1)  # [B]
    fde = dist[:, -1]       # [B]

    return {
        "ade": ade,
        "fde": fde,
        "per_wp_l2": dist,
        "pred_wp": pred_wp,
        "gt_wp": gt_wp,
    }


def summarize(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "rmse": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    arr = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    opt = test_parser()

    hypes = yaml_utils.load_yaml(None, opt)

    print("Building dataset...")
    dataset = build_dataset(hypes, visualize=True, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=opt.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    print(f"Dataset size: {len(dataset)}")

    print("Creating model...")
    model = train_utils.create_model(hypes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Loading checkpoint from: {opt.model_dir}")
    _, model = train_utils.load_saved_model(opt.model_dir, model)

    model.to(device)
    model.eval()

    ade_all = []
    fde_all = []
    per_wp_all = []
    csv_rows = []

    printed_debug = False
    skipped = 0

    max_iter = len(data_loader)
    if opt.max_samples > 0:
        max_iter = min(max_iter, opt.max_samples)

    for sample_idx, batch_data in tqdm(enumerate(data_loader), total=max_iter):
        if sample_idx >= max_iter:
            break

        if batch_data is None:
            skipped += 1
            continue

        batch_data = train_utils.to_device(batch_data, device)

        if "ego" not in batch_data:
            raise KeyError("Expected batch_data['ego'], but it was not found.")

        batch_ego = batch_data["ego"]

        with torch.no_grad():
            output_dict = model(batch_ego)

        if opt.debug_shapes and not printed_debug:
            print("\n========== DEBUG ==========")

            print("\nBatch ego keys:")
            for key in list_nested_keys(batch_ego, max_depth=3):
                print("  ", key)

            print("\nModel output keys:")
            for key in list_nested_keys(output_dict, max_depth=3):
                print("  ", key)

            if "future_waypoints" in batch_ego:
                print(
                    "\nGT future_waypoints shape:",
                    tuple(batch_ego["future_waypoints"].shape),
                )
                print(
                    "GT future_waypoints[0]:",
                    batch_ego["future_waypoints"][0].detach().cpu(),
                )
            else:
                print("\nGT future_waypoints missing from batch_ego")

            if "future_waypoints" in output_dict:
                print(
                    "\nPred future_waypoints shape:",
                    tuple(output_dict["future_waypoints"].shape),
                )
                print(
                    "Pred future_waypoints[0]:",
                    output_dict["future_waypoints"][0].detach().cpu(),
                )
            else:
                print("\nPred future_waypoints missing from output_dict")

            printed_debug = True

        if "future_waypoints" not in batch_ego:
            skipped += 1
            raise KeyError(
                "batch_data['ego']['future_waypoints'] is missing. "
                "You probably added future_waypoints to collate_batch_train "
                "but not collate_batch_test."
            )

        if "future_waypoints" not in output_dict:
            skipped += 1
            raise KeyError(
                "output_dict['future_waypoints'] is missing. "
                "Check that your PointPillar forward returns planner output."
            )

        gt_wp = batch_ego["future_waypoints"]
        pred_wp = output_dict["future_waypoints"]

        metrics = compute_planning_metrics(
            pred_wp,
            gt_wp,
            allow_truncate=opt.allow_truncate,
        )

        ade = metrics["ade"].detach().cpu()              # [B]
        fde = metrics["fde"].detach().cpu()              # [B]
        per_wp_l2 = metrics["per_wp_l2"].detach().cpu()  # [B, N]
        pred_wp_cpu = metrics["pred_wp"].detach().cpu()  # [B, N, 2]
        gt_wp_cpu = metrics["gt_wp"].detach().cpu()      # [B, N, 2]

        ade_all.extend(ade.tolist())
        fde_all.extend(fde.tolist())
        per_wp_all.append(per_wp_l2)

        batch_size = ade.shape[0]
        num_wp = per_wp_l2.shape[1]

        for b in range(batch_size):
            row = {
                "sample_idx": sample_idx,
                "batch_idx": b,
                "ade": float(ade[b].item()),
                "fde": float(fde[b].item()),
            }

            for k in range(num_wp):
                row[f"wp{k + 1}_l2"] = float(per_wp_l2[b, k].item())

                row[f"pred_wp{k + 1}_x"] = float(pred_wp_cpu[b, k, 0].item())
                row[f"pred_wp{k + 1}_y"] = float(pred_wp_cpu[b, k, 1].item())

                row[f"gt_wp{k + 1}_x"] = float(gt_wp_cpu[b, k, 0].item())
                row[f"gt_wp{k + 1}_y"] = float(gt_wp_cpu[b, k, 1].item())

            csv_rows.append(row)

    print("\n========== Planning Evaluation ==========")
    print(f"Evaluated waypoint samples: {len(ade_all)}")
    print(f"Skipped samples:            {skipped}")

    ade_summary = summarize(ade_all)
    fde_summary = summarize(fde_all)

    print("\nADE:")
    print(f"  mean:   {ade_summary['mean']:.4f}")
    print(f"  median: {ade_summary['median']:.4f}")
    print(f"  std:    {ade_summary['std']:.4f}")
    print(f"  rmse:   {ade_summary['rmse']:.4f}")
    print(f"  min:    {ade_summary['min']:.4f}")
    print(f"  max:    {ade_summary['max']:.4f}")

    print("\nFDE:")
    print(f"  mean:   {fde_summary['mean']:.4f}")
    print(f"  median: {fde_summary['median']:.4f}")
    print(f"  std:    {fde_summary['std']:.4f}")
    print(f"  rmse:   {fde_summary['rmse']:.4f}")
    print(f"  min:    {fde_summary['min']:.4f}")
    print(f"  max:    {fde_summary['max']:.4f}")

    if len(per_wp_all) > 0:
        per_wp_cat = torch.cat(per_wp_all, dim=0)  # [num_samples, N]
        per_wp_mean = per_wp_cat.mean(dim=0)
        per_wp_median = per_wp_cat.median(dim=0).values

        print("\nWaypoint-wise L2 error:")
        for k in range(per_wp_cat.shape[1]):
            print(
                f"  wp{k + 1}: "
                f"mean={per_wp_mean[k].item():.4f}, "
                f"median={per_wp_median[k].item():.4f}"
            )

    if opt.save_csv:
        if len(csv_rows) == 0:
            print("\nNo CSV saved because no planning samples were evaluated.")
        else:
            csv_path = os.path.join(opt.model_dir, opt.csv_name)

            fieldnames = list(csv_rows[0].keys())

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)

            print(f"\nSaved CSV to: {csv_path}")


if __name__ == "__main__":
    main()