import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def tensor_shape_summary(x):
    if torch.is_tensor(x):
        return tuple(x.shape)
    # numpy arrays (and other array-likes) used by dataset __getitem__.
    if hasattr(x, "shape"):
        try:
            return tuple(x.shape)
        except Exception:
            pass
    if isinstance(x, dict):
        return {k: tensor_shape_summary(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tensor_shape_summary(v) for v in x]
    return type(x).__name__


def save_csv(rows, csv_path, num_waypoints):
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    fieldnames = ["dataset_idx", "ade", "fde"]
    for i in range(num_waypoints):
        fieldnames.append(f"wp{i}_error")
    for i in range(num_waypoints):
        fieldnames.extend(
            [
                f"gt_wp{i}_x",
                f"gt_wp{i}_y",
                f"pred_wp{i}_x",
                f"pred_wp{i}_y",
            ]
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved CSV to: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--hypes_yaml", default=None)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--debug_batch", action="store_true")
    args = parser.parse_args()

    hypes_yaml = args.hypes_yaml
    if hypes_yaml is None:
        hypes_yaml = os.path.join(args.model_dir, "config.yaml")

    print(f"Loading config: {hypes_yaml}")
    hypes = yaml_utils.load_yaml(hypes_yaml, None)
    print(f"Fusion dataset: {hypes['fusion']['core_method']}")
    dataset = build_dataset(hypes, visualize=False, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    if args.debug_batch:
        print("========== DEBUG BATCH (collated batch_data['ego']) ==========")
        first_batch = next(iter(data_loader))
        if first_batch is None:
            raise RuntimeError("First batch was None.")
        ego = first_batch["ego"]
        for key, value in ego.items():
            vtype = type(value).__name__
            vshape = tensor_shape_summary(value)
            print(f"{key}: type={vtype}, shape={vshape}")
        return

    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.cuda()
    model.eval()

    evaluated_batches = 0
    ade_values = []
    fde_values = []
    per_timestep_errors = []
    csv_rows = []
    gt_shape = None
    pred_shape = None

    with torch.no_grad():
        for dataset_idx, batch_data in enumerate(tqdm(data_loader)):
            if args.max_batches > 0 and evaluated_batches >= args.max_batches:
                break
            if batch_data is None:
                continue

            batch_data = train_utils.to_device(batch_data, torch.device("cuda"))
            ego = batch_data["ego"]

            if "future_waypoints" not in ego:
                raise KeyError(
                    "batch_data['ego']['future_waypoints'] is missing. "
                    f"Available ego keys: {list(ego.keys())}. "
                    f"Fusion dataset: {hypes['fusion']['core_method']}. "
                    "Only the early-fusion dataset path in this repo currently "
                    "adds GT future_waypoints. Rerun with --debug_batch to inspect "
                    "the first batch."
                )

            # Leakage checks: target must exist and must not be the GT endpoint.
            if "planning_target" not in ego:
                raise KeyError(
                    "planning_target missing from batch. Endpoint-derived "
                    "targets are no longer allowed."
                )
            gt_end = ego["future_waypoints"][:, -1, :]
            tgt = ego["planning_target"]
            if torch.allclose(tgt, gt_end, atol=1e-4, rtol=0):
                raise RuntimeError(
                    "Endpoint leakage: planning_target equals "
                    "future_waypoints[:, -1]."
                )

            output_dict = model(batch_data["ego"])
            if "future_waypoints" not in output_dict:
                raise KeyError(
                    "output_dict['future_waypoints'] is missing. "
                    f"Available output keys: {list(output_dict.keys())}."
                )

            # Model must not receive future waypoints as an input feature.
            # They are labels only (present in batch for metric computation).
            if "planning_target" in output_dict:
                out_tgt = output_dict["planning_target"]
                if torch.allclose(out_tgt, gt_end, atol=1e-4, rtol=0):
                    raise RuntimeError(
                        "Endpoint leakage in model output planning_target."
                    )

            pred_waypoints = output_dict["future_waypoints"]
            gt_waypoints = batch_data["ego"]["future_waypoints"]

            if gt_shape is None:
                gt_shape = tuple(gt_waypoints.shape)
                pred_shape = tuple(pred_waypoints.shape)
                print(
                    "Leakage check passed: planning_target != GT endpoint. "
                    f"target[0]={tgt[0].detach().cpu().tolist()}, "
                    f"gt_end[0]={gt_end[0].detach().cpu().tolist()}"
                )

            # V2Xverse ADE_FDE: per-sample mean over waypoints, then mean FDE
            # on the last waypoint. Aggregate across samples with np.mean.
            errors = torch.linalg.norm(pred_waypoints - gt_waypoints, dim=-1)
            ade_batch = errors.mean(dim=1)  # [B]
            fde_batch = errors[:, -1]       # [B]
            per_timestep = errors.mean(dim=0).detach().cpu().numpy()

            ade_values.extend(ade_batch.detach().cpu().tolist())
            fde_values.extend(fde_batch.detach().cpu().tolist())
            per_timestep_errors.append(per_timestep)
            evaluated_batches += 1

            if args.csv_path is not None:
                pred_np = pred_waypoints[0].detach().cpu().numpy()
                gt_np = gt_waypoints[0].detach().cpu().numpy()
                err_np = errors[0].detach().cpu().numpy()
                num_waypoints = gt_np.shape[0]

                row = {
                    "dataset_idx": int(dataset_idx),
                    "ade": float(err_np.mean()),
                    "fde": float(err_np[-1]),
                }
                for i in range(num_waypoints):
                    row[f"wp{i}_error"] = float(err_np[i])
                for i in range(num_waypoints):
                    row[f"gt_wp{i}_x"] = float(gt_np[i, 0])
                    row[f"gt_wp{i}_y"] = float(gt_np[i, 1])
                    row[f"pred_wp{i}_x"] = float(pred_np[i, 0])
                    row[f"pred_wp{i}_y"] = float(pred_np[i, 1])
                csv_rows.append(row)

    if evaluated_batches == 0:
        print("No batches were evaluated.")
        return

    ade_mean = float(np.mean(ade_values))
    fde_mean = float(np.mean(fde_values))
    per_wp_mean = np.mean(np.stack(per_timestep_errors, axis=0), axis=0)
    num_waypoints = per_wp_mean.shape[0]

    print("========== PLANNING / WAYPOINT EVALUATION ==========")
    print(f"Evaluated batches: {evaluated_batches}")
    print(f"GT waypoint shape: {gt_shape}")
    print(f"Pred waypoint shape: {pred_shape}")
    print(f"ADE: {ade_mean:.6f}")
    print(f"FDE: {fde_mean:.6f}")
    print(f"Per-waypoint mean L2: {per_wp_mean.tolist()}")
    print(
        "Frame ADE min/median/max: "
        f"{float(np.min(ade_values)):.6f} / "
        f"{float(np.median(ade_values)):.6f} / "
        f"{float(np.max(ade_values)):.6f}"
    )

    if args.csv_path is not None:
        save_csv(csv_rows, args.csv_path, num_waypoints)


if __name__ == "__main__":
    main()
