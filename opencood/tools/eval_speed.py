import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset


def get_dataset_sample_info(dataset, idx):
    """
    Recover dataset/scenario/timestamp info for a given dataloader index.

    This assumes OpenCOOD BaseDataset-style attributes:
        dataset.len_record
        dataset.scenario_database
        dataset.return_timestamp_key(...)
    """
    scenario_index = 0
    for i, ele in enumerate(dataset.len_record):
        if idx < ele:
            scenario_index = i
            break

    timestamp_index = idx if scenario_index == 0 else \
        idx - dataset.len_record[scenario_index - 1]

    scenario_database = dataset.scenario_database[scenario_index]
    timestamp_key = dataset.return_timestamp_key(
        scenario_database,
        timestamp_index
    )

    cav_ids = list(scenario_database.keys())

    scenario_path = "UNKNOWN"
    first_yaml = "UNKNOWN"
    first_lidar = "UNKNOWN"

    try:
        first_cav = cav_ids[0]
        first_yaml = scenario_database[first_cav][timestamp_key].get("yaml", "UNKNOWN")
        first_lidar = scenario_database[first_cav][timestamp_key].get("lidar", "UNKNOWN")

        # expected: scenario_folder / cav_id / timestamp.yaml
        if first_yaml != "UNKNOWN":
            scenario_path = os.path.dirname(os.path.dirname(first_yaml))
    except Exception as e:
        scenario_path = f"FAILED_TO_PARSE_PATH: {e}"

    return {
        "dataset_idx": idx,
        "scenario_index": scenario_index,
        "timestamp_index": timestamp_index,
        "timestamp_key": timestamp_key,
        "scenario_path": scenario_path,
        "first_yaml": first_yaml,
        "first_lidar": first_lidar,
        "cav_ids": cav_ids,
    }


def extract_pred_boxes_8d(post_processor, ego_data, output_dict):
    """
    Decode early-fusion model output into 8D predicted boxes.

    For early fusion:
        ego_data contains anchor_box
        output_dict directly contains psm and rm

    Returns:
        pred_boxes_8d: torch.Tensor, shape (N, 8)
        pred_scores: torch.Tensor, shape (N,)
    """
    box_code_size = post_processor.params.get("box_code_size", 8)

    anchor_box = ego_data["anchor_box"]
    prob = output_dict["psm"]
    reg = output_dict["rm"]

    prob = torch.sigmoid(prob.permute(0, 2, 3, 1))
    prob = prob.reshape(1, -1)

    # Requires your modified delta_to_boxes3d returning 8D boxes.
    batch_box3d = post_processor.delta_to_boxes3d(reg, anchor_box)

    score_threshold = post_processor.params["target_args"]["score_threshold"]

    mask = torch.gt(prob, score_threshold)
    mask = mask.view(1, -1)

    mask_reg = mask.unsqueeze(2).repeat(1, 1, box_code_size)

    assert batch_box3d.shape[0] == 1

    boxes3d_all = torch.masked_select(
        batch_box3d[0],
        mask_reg[0]
    ).view(-1, box_code_size)

    scores = torch.masked_select(prob[0], mask[0])

    if boxes3d_all.shape[0] == 0:
        return None, None

    return boxes3d_all.detach().cpu(), scores.detach().cpu()


def nearest_gt_speed_match(pred_boxes_8d, gt_boxes_8d, max_dist=3.0, frame_idx=None):
    """
    Match each predicted box to nearest GT center and compare speed.

    Returns:
        records: list of dicts containing speed error and debug info.

    Note:
        This is a rough sanity metric. It uses nearest-center matching
        before NMS, so outliers may be matching artifacts.
    """
    records = []

    if pred_boxes_8d is None or gt_boxes_8d is None:
        return records

    if len(pred_boxes_8d) == 0 or len(gt_boxes_8d) == 0:
        return records

    pred_centers = pred_boxes_8d[:, :2]
    gt_centers = gt_boxes_8d[:, :2]

    used_gt = set()

    for i in range(len(pred_boxes_8d)):
        dists = np.linalg.norm(gt_centers - pred_centers[i], axis=1)
        nearest = int(np.argmin(dists))

        if dists[nearest] <= max_dist and nearest not in used_gt:
            pred_speed = float(pred_boxes_8d[i, 7])
            gt_speed = float(gt_boxes_8d[nearest, 7])
            abs_error = abs(pred_speed - gt_speed)

            records.append({
                "frame_idx": frame_idx,
                "pred_idx": int(i),
                "gt_idx": int(nearest),
                "pred_speed": pred_speed,
                "gt_speed": gt_speed,
                "abs_error": abs_error,
                "center_dist": float(dists[nearest]),
                "pred_center_x": float(pred_boxes_8d[i, 0]),
                "pred_center_y": float(pred_boxes_8d[i, 1]),
                "gt_center_x": float(gt_boxes_8d[nearest, 0]),
                "gt_center_y": float(gt_boxes_8d[nearest, 1]),
                "pred_box": pred_boxes_8d[i].tolist(),
                "gt_box": gt_boxes_8d[nearest].tolist(),
            })

            used_gt.add(nearest)

    return records


def print_speed_distribution(name, values):
    if len(values) == 0:
        print(f"{name}: no values")
        return

    values = np.array(values, dtype=np.float32)
    percentiles = np.percentile(values, [0, 25, 50, 75, 90, 95, 99, 100])

    print(f"\n========== {name} DISTRIBUTION ==========")
    print(f"count: {len(values)}")
    print(f"mean: {float(np.mean(values)):.4f}")
    print(f"std: {float(np.std(values)):.4f}")
    print(f"min/p25/p50/p75/p90/p95/p99/max:")
    print(" ".join([f"{p:.4f}" for p in percentiles]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to OpenCOOD log folder containing checkpoint and config.yaml"
    )
    parser.add_argument(
        "--hypes_yaml",
        default=None,
        help="Optional yaml path. If omitted, uses config.yaml in model_dir."
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=-1,
        help="Limit number of validation batches. Use -1 for all."
    )
    parser.add_argument(
        "--max_match_dist",
        type=float,
        default=3.0,
        help="Max XY center distance for pred-GT speed matching."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of worst speed-error cases to print."
    )
    args = parser.parse_args()

    if args.hypes_yaml is None:
        hypes_yaml = os.path.join(args.model_dir, "config.yaml")
    else:
        hypes_yaml = args.hypes_yaml

    print("Loading config:", hypes_yaml)
    hypes = yaml_utils.load_yaml(hypes_yaml, None)

    print("Building validation/test dataset...")
    dataset = build_dataset(hypes, visualize=False, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )

    print("Dataset length:", len(dataset))
    print("Dataloader length:", len(data_loader))

    print("Creating model...")
    model = train_utils.create_model(hypes)
    model.cuda()
    model.eval()

    print("Loading checkpoint from:", args.model_dir)
    _, model = train_utils.load_saved_model(args.model_dir, model)

    post_processor = dataset.post_processor

    all_abs_errors = []
    all_records = []

    all_gt_speeds = []
    all_pred_speeds = []

    num_frames_with_matches = 0
    total_preds = 0
    total_gt = 0

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(data_loader)):
            if args.max_batches > 0 and i >= args.max_batches:
                break

            if batch_data is None:
                continue

            batch_data = train_utils.to_device(
                batch_data,
                torch.device("cuda")
            )

            output_dict = model(batch_data["ego"])

            pred_boxes_8d, pred_scores = extract_pred_boxes_8d(
                post_processor,
                batch_data["ego"],
                output_dict
            )

            gt_boxes = batch_data["ego"]["object_bbx_center"]
            gt_mask = batch_data["ego"]["object_bbx_mask"]

            # batch size should be 1 for eval
            gt_boxes = gt_boxes[0].detach().cpu().numpy()
            gt_mask = gt_mask[0].detach().cpu().numpy().astype(bool)
            gt_boxes_valid = gt_boxes[gt_mask]

            total_gt += len(gt_boxes_valid)

            if len(gt_boxes_valid) > 0:
                all_gt_speeds.extend(gt_boxes_valid[:, 7].tolist())

            if pred_boxes_8d is None:
                continue

            pred_boxes_8d_np = pred_boxes_8d.numpy()
            total_preds += len(pred_boxes_8d_np)

            if len(pred_boxes_8d_np) > 0:
                all_pred_speeds.extend(pred_boxes_8d_np[:, 7].tolist())

            records = nearest_gt_speed_match(
                pred_boxes_8d_np,
                gt_boxes_valid,
                max_dist=args.max_match_dist,
                frame_idx=i
            )

            if len(records) > 0:
                num_frames_with_matches += 1
                all_records.extend(records)
                all_abs_errors.extend([r["abs_error"] for r in records])

    print("\n========== SPEED EVALUATION ==========")
    print("Total GT boxes:", total_gt)
    print("Total predicted boxes before NMS:", total_preds)
    print("Frames with at least one matched pred-GT pair:", num_frames_with_matches)
    print("Total matched pairs:", len(all_abs_errors))
    print("Max match distance:", args.max_match_dist)

    print_speed_distribution("GT SPEED", all_gt_speeds)
    print_speed_distribution("PREDICTED SPEED BEFORE NMS", all_pred_speeds)

    if len(all_abs_errors) == 0:
        print("\nNo speed matches found.")
        print("Try increasing --max_match_dist or lowering score_threshold in YAML.")
        return

    errors = np.array(all_abs_errors)

    print("\n========== SPEED ERROR METRICS ==========")
    print("Speed MAE:", float(np.mean(errors)))
    print("Speed RMSE:", float(np.sqrt(np.mean(errors ** 2))))
    print("Speed median AE:", float(np.median(errors)))
    print("Speed max AE:", float(np.max(errors)))
    print("Speed std AE:", float(np.std(errors)))

    print("\n========== WORST SPEED ERRORS WITH DATASET INFO ==========")
    all_records_sorted = sorted(
        all_records,
        key=lambda r: r["abs_error"],
        reverse=True
    )

    for r in all_records_sorted[:args.top_k]:
        info = get_dataset_sample_info(dataset, r["frame_idx"])

        print(
            f"dataset_idx={info['dataset_idx']} "
            f"scenario_index={info['scenario_index']} "
            f"timestamp_index={info['timestamp_index']} "
            f"timestamp={info['timestamp_key']} "
            f"err={r['abs_error']:.2f} m/s "
            f"pred_speed={r['pred_speed']:.2f} "
            f"gt_speed={r['gt_speed']:.2f} "
            f"center_dist={r['center_dist']:.2f} "
            f"pred_idx={r['pred_idx']} "
            f"gt_idx={r['gt_idx']} "
            f"pred_center=({r['pred_center_x']:.2f}, {r['pred_center_y']:.2f}) "
            f"gt_center=({r['gt_center_x']:.2f}, {r['gt_center_y']:.2f}) "
            f"scenario_path={info['scenario_path']} "
            f"cavs={info['cav_ids']}"
        )

        print(f"    first_yaml={info['first_yaml']}")
        print(f"    first_lidar={info['first_lidar']}")
        print(f"    pred_box={np.array(r['pred_box'])}")
        print(f"    gt_box={np.array(r['gt_box'])}")


if __name__ == "__main__":
    main()