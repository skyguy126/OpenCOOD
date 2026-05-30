import argparse
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils


KMH_TO_MPS = 1.0 / 3.6


def decode_pred_boxes_8d(post_processor, ego_data, output_dict, score_threshold=None):
    """
    Decode early-fusion output to 8D boxes:
    [x, y, z, h, w, l, yaw, speed]

    NOTE:
    In this dataset, the raw speed label appears to be km/h.
    This function keeps the raw value as-is. Conversion happens later.
    """
    box_code_size = post_processor.params.get("box_code_size", 8)

    anchor_box = ego_data["anchor_box"]
    prob = output_dict["psm"]
    reg = output_dict["rm"]

    prob = torch.sigmoid(prob.permute(0, 2, 3, 1))
    prob = prob.reshape(1, -1)

    batch_box3d = post_processor.delta_to_boxes3d(reg, anchor_box)

    if score_threshold is None:
        score_threshold = post_processor.params["target_args"]["score_threshold"]

    mask = torch.gt(prob, score_threshold)
    mask = mask.view(1, -1)

    mask_reg = mask.unsqueeze(2).repeat(1, 1, box_code_size)

    boxes8d = torch.masked_select(
        batch_box3d[0],
        mask_reg[0]
    ).view(-1, box_code_size)

    scores = torch.masked_select(prob[0], mask[0])

    if boxes8d.shape[0] == 0:
        return None, None

    return boxes8d, scores


def apply_nms_to_8d_boxes(post_processor, boxes8d, scores):
    """
    Apply NMS using 7D box geometry while preserving the speed column.
    """
    if boxes8d is None or boxes8d.shape[0] == 0:
        return None, None

    boxes7d = boxes8d[:, :7]

    pred_corners = box_utils.boxes_to_corners_3d(
        boxes7d,
        order=post_processor.params["order"]
    )

    keep_idx = box_utils.nms_rotated(
        pred_corners,
        scores,
        post_processor.params["nms_thresh"]
    )

    return boxes8d[keep_idx], scores[keep_idx]


def standup_iou_matrix(pred_boxes8d, gt_boxes8d, post_processor):
    """
    Compute BEV standup-box IoU between predicted and GT boxes.

    Returns:
        iou: np.ndarray, shape (num_pred, num_gt)
    """
    pred_boxes7d = pred_boxes8d[:, :7]
    gt_boxes7d = gt_boxes8d[:, :7]

    pred_corners = box_utils.boxes_to_corners_3d(
        pred_boxes7d,
        order=post_processor.params["order"]
    )
    gt_corners = box_utils.boxes_to_corners_3d(
        gt_boxes7d,
        order=post_processor.params["order"]
    )

    pred_standup = box_utils.corner_to_standup_box_torch(pred_corners)
    gt_standup = box_utils.corner_to_standup_box_torch(gt_corners)

    pred = pred_standup.detach().cpu().numpy()
    gt = gt_standup.detach().cpu().numpy()

    iou = np.zeros((pred.shape[0], gt.shape[0]), dtype=np.float32)

    for i in range(pred.shape[0]):
        px1, py1, px2, py2 = pred[i]
        p_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)

        for j in range(gt.shape[0]):
            gx1, gy1, gx2, gy2 = gt[j]
            g_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)

            ix1 = max(px1, gx1)
            iy1 = max(py1, gy1)
            ix2 = min(px2, gx2)
            iy2 = min(py2, gy2)

            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = p_area + g_area - inter

            if union > 0:
                iou[i, j] = inter / union

    return iou


def match_each_gt_to_prediction(pred_boxes8d, scores, gt_boxes8d, post_processor, iou_thresh=0.3):
    """
    GT-centric matching.

    For each GT box:
        find the unmatched predicted box with highest IoU.
        if IoU >= threshold, compare speed.

    Speed convention:
        raw speed is assumed to be km/h based on YAML inspection.
        m/s values are computed by raw / 3.6.

    Returns:
        records: matched GT-pred speed comparison records
        missed_gt_indices: GT objects with no matching prediction
    """
    records = []
    missed_gt_indices = []

    if pred_boxes8d is None or pred_boxes8d.shape[0] == 0:
        return records, list(range(gt_boxes8d.shape[0]))

    if gt_boxes8d is None or gt_boxes8d.shape[0] == 0:
        return records, missed_gt_indices

    iou = standup_iou_matrix(pred_boxes8d, gt_boxes8d, post_processor)

    used_pred = set()

    for gt_idx in range(gt_boxes8d.shape[0]):
        pred_order = np.argsort(-iou[:, gt_idx])

        matched = False

        for pred_idx in pred_order:
            pred_idx = int(pred_idx)

            if pred_idx in used_pred:
                continue

            best_iou = float(iou[pred_idx, gt_idx])

            if best_iou < iou_thresh:
                break

            pred_speed_raw_kmh = float(pred_boxes8d[pred_idx, 7].detach().cpu())
            gt_speed_raw_kmh = float(gt_boxes8d[gt_idx, 7].detach().cpu())

            pred_speed_mps = pred_speed_raw_kmh * KMH_TO_MPS
            gt_speed_mps = gt_speed_raw_kmh * KMH_TO_MPS

            signed_error_raw_kmh = pred_speed_raw_kmh - gt_speed_raw_kmh
            abs_error_raw_kmh = abs(signed_error_raw_kmh)

            signed_error_mps = pred_speed_mps - gt_speed_mps
            abs_error_mps = abs(signed_error_mps)

            records.append({
                "gt_idx": int(gt_idx),
                "pred_idx": int(pred_idx),
                "iou": best_iou,
                "score": float(scores[pred_idx].detach().cpu()),

                "pred_speed_raw_kmh": pred_speed_raw_kmh,
                "gt_speed_raw_kmh": gt_speed_raw_kmh,
                "signed_error_raw_kmh": signed_error_raw_kmh,
                "abs_error_raw_kmh": abs_error_raw_kmh,

                "pred_speed_mps": pred_speed_mps,
                "gt_speed_mps": gt_speed_mps,
                "signed_error_mps": signed_error_mps,
                "abs_error_mps": abs_error_mps,

                "pred_box": pred_boxes8d[pred_idx].detach().cpu().numpy(),
                "gt_box": gt_boxes8d[gt_idx].detach().cpu().numpy(),
            })

            used_pred.add(pred_idx)
            matched = True
            break

        if not matched:
            missed_gt_indices.append(int(gt_idx))

    return records, missed_gt_indices


def get_scenario_ranges(dataset):
    """
    Return list of (scenario_idx, start_idx, end_idx_exclusive).

    OpenCOOD dataset.len_record is usually cumulative:
        [112, 240, 351, ...]

    But this also handles the case where len_record is per-scenario length:
        [112, 128, 111, ...]

    The detection is based on total dataset length.
    """
    ranges = []
    len_record = list(dataset.len_record)

    if len(len_record) == 0:
        return ranges

    dataset_length = len(dataset)

    # Most OpenCOOD datasets use cumulative len_record.
    if int(len_record[-1]) == dataset_length:
        prev_end = 0
        for scenario_idx, end_idx in enumerate(len_record):
            end_idx = int(end_idx)
            start_idx = int(prev_end)
            ranges.append((scenario_idx, start_idx, end_idx))
            prev_end = end_idx

    # Some datasets may store per-scenario lengths.
    elif sum([int(x) for x in len_record]) == dataset_length:
        start_idx = 0
        for scenario_idx, length in enumerate(len_record):
            length = int(length)
            end_idx = start_idx + length
            ranges.append((scenario_idx, start_idx, end_idx))
            start_idx = end_idx

    else:
        raise ValueError(
            "Could not interpret dataset.len_record. "
            f"len_record={len_record}, len(dataset)={dataset_length}"
        )

    return ranges


def get_dataset_sample_info(dataset, idx):
    """
    Recover scenario/timestamp info for debugging.
    idx must be the original dataset index, not subset index.
    """
    scenario_ranges = get_scenario_ranges(dataset)

    scenario_index = None
    timestamp_index = None

    for s_idx, start_idx, end_idx in scenario_ranges:
        if start_idx <= idx < end_idx:
            scenario_index = s_idx
            timestamp_index = idx - start_idx
            break

    if scenario_index is None:
        raise IndexError(f"Dataset index {idx} is outside scenario ranges.")

    scenario_database = dataset.scenario_database[scenario_index]
    timestamp_key = dataset.return_timestamp_key(
        scenario_database,
        timestamp_index
    )

    cav_ids = list(scenario_database.keys())

    scenario_path = "UNKNOWN"
    try:
        first_cav = cav_ids[0]
        yaml_path = scenario_database[first_cav][timestamp_key].get("yaml", "UNKNOWN")
        if yaml_path != "UNKNOWN":
            scenario_path = os.path.dirname(os.path.dirname(yaml_path))
    except Exception:
        pass

    return {
        "dataset_idx": int(idx),
        "scenario_index": int(scenario_index),
        "timestamp_index": int(timestamp_index),
        "timestamp": timestamp_key,
        "scenario_path": scenario_path,
        "cav_ids": cav_ids,
    }


def save_records_to_csv(records, csv_path):
    """
    Save matched GT-pred speed comparison records to CSV.
    One row = one matched GT box and predicted box pair.
    """
    if len(records) == 0:
        print("No matched records to save to CSV.")
        return

    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    fieldnames = [
        "dataset_idx",
        "scenario_index",
        "timestamp_index",
        "timestamp",
        "scenario_path",
        "gt_idx",
        "pred_idx",
        "iou",
        "score",

        "gt_speed_raw_kmh",
        "pred_speed_raw_kmh",
        "signed_error_raw_kmh",
        "abs_error_raw_kmh",

        "gt_speed_mps",
        "pred_speed_mps",
        "signed_error_mps",
        "abs_error_mps",

        "gt_x", "gt_y", "gt_z", "gt_h", "gt_w", "gt_l", "gt_yaw",
        "pred_x", "pred_y", "pred_z", "pred_h", "pred_w", "pred_l", "pred_yaw",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            gt_box = r["gt_box"]
            pred_box = r["pred_box"]

            row = {
                "dataset_idx": r["dataset_idx"],
                "scenario_index": r["scenario_index"],
                "timestamp_index": r["timestamp_index"],
                "timestamp": r["timestamp"],
                "scenario_path": r["scenario_path"],
                "gt_idx": r["gt_idx"],
                "pred_idx": r["pred_idx"],
                "iou": r["iou"],
                "score": r["score"],

                "gt_speed_raw_kmh": r["gt_speed_raw_kmh"],
                "pred_speed_raw_kmh": r["pred_speed_raw_kmh"],
                "signed_error_raw_kmh": r["signed_error_raw_kmh"],
                "abs_error_raw_kmh": r["abs_error_raw_kmh"],

                "gt_speed_mps": r["gt_speed_mps"],
                "pred_speed_mps": r["pred_speed_mps"],
                "signed_error_mps": r["signed_error_mps"],
                "abs_error_mps": r["abs_error_mps"],

                "gt_x": gt_box[0],
                "gt_y": gt_box[1],
                "gt_z": gt_box[2],
                "gt_h": gt_box[3],
                "gt_w": gt_box[4],
                "gt_l": gt_box[5],
                "gt_yaw": gt_box[6],

                "pred_x": pred_box[0],
                "pred_y": pred_box[1],
                "pred_z": pred_box[2],
                "pred_h": pred_box[3],
                "pred_w": pred_box[4],
                "pred_l": pred_box[5],
                "pred_yaw": pred_box[6],
            }

            writer.writerow(row)

    print(f"\nSaved CSV to: {csv_path}")


def build_subset_for_scenarios(dataset, scenario_indices):
    """
    Build a torch Subset that contains only frames from requested scenario indices.

    Returns:
        dataset_for_loader
        selected_indices
    """
    scenario_ranges = get_scenario_ranges(dataset)

    print("\n========== AVAILABLE SCENARIO RANGES ==========")
    for scenario_idx, start_idx, end_idx in scenario_ranges:
        print(
            f"scenario {scenario_idx}: "
            f"dataset_idx {start_idx} to {end_idx - 1} "
            f"({end_idx - start_idx} frames)"
        )

    if scenario_indices is None:
        return dataset, None

    selected_indices = []
    available = {s for s, _, _ in scenario_ranges}

    for requested in scenario_indices:
        if requested not in available:
            print(f"WARNING: scenario {requested} is not available in this dataset.")

    print("\n========== SELECTED SCENARIO RANGES ==========")
    for scenario_idx, start_idx, end_idx in scenario_ranges:
        if scenario_idx in scenario_indices:
            selected_indices.extend(list(range(start_idx, end_idx)))
            print(
                f"selected scenario {scenario_idx}: "
                f"dataset_idx {start_idx} to {end_idx - 1} "
                f"({end_idx - start_idx} frames)"
            )

    if len(selected_indices) == 0:
        valid = [s for s, _, _ in scenario_ranges]
        raise ValueError(
            f"No frames found for scenario_indices={scenario_indices}. "
            f"Valid scenario indices are: {valid}"
        )

    dataset_for_loader = Subset(dataset, selected_indices)
    return dataset_for_loader, selected_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--hypes_yaml", default=None)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--score_thresh", type=float, default=None)
    parser.add_argument("--iou_thresh", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=20)

    parser.add_argument(
        "--csv_path",
        default=None,
        help="Optional path to save matched GT-pred speed comparison CSV."
    )

    parser.add_argument(
        "--scenario_indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of scenario indices to evaluate, e.g. --scenario_indices 0 3 5"
    )

    parser.add_argument(
        "--list_scenarios",
        action="store_true",
        help="List available scenario indices and exit."
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers. Use 0 for debugging/stability."
    )

    args = parser.parse_args()

    hypes_yaml = args.hypes_yaml
    if hypes_yaml is None:
        hypes_yaml = os.path.join(args.model_dir, "config.yaml")

    print("Loading config:", hypes_yaml)
    hypes = yaml_utils.load_yaml(hypes_yaml, None)

    print("Building validation/test dataset...")
    dataset = build_dataset(hypes, visualize=False, train=False)

    print("Dataset length:", len(dataset))
    print("len_record:", list(dataset.len_record))
    print("Number of scenario_database entries:", len(dataset.scenario_database))

    dataset_for_loader, selected_indices = build_subset_for_scenarios(
        dataset,
        args.scenario_indices
    )

    if args.list_scenarios:
        print("\nExiting after listing scenarios.")
        return

    data_loader = DataLoader(
        dataset_for_loader,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )

    print("Dataloader length:", len(data_loader))

    if args.scenario_indices is not None:
        print("Filtering to scenario indices:", args.scenario_indices)
        print("max_batches means evaluated batches after scenario filtering.")

    print("Creating model...")
    model = train_utils.create_model(hypes)
    model.cuda()
    model.eval()

    print("Loading checkpoint from:", args.model_dir)
    _, model = train_utils.load_saved_model(args.model_dir, model)

    post_processor = dataset.post_processor

    all_records = []
    total_gt = 0
    total_matched_gt = 0
    total_missed_gt = 0
    total_pred_after_nms = 0
    frames_with_match = 0
    evaluated_batches = 0

    with torch.no_grad():
        for loader_idx, batch_data in enumerate(tqdm(data_loader)):

            if args.max_batches > 0 and evaluated_batches >= args.max_batches:
                break

            if batch_data is None:
                continue

            if selected_indices is not None:
                original_idx = selected_indices[loader_idx]
            else:
                original_idx = loader_idx

            info = get_dataset_sample_info(dataset, original_idx)

            evaluated_batches += 1

            batch_data = train_utils.to_device(
                batch_data,
                torch.device("cuda")
            )

            output_dict = model(batch_data["ego"])

            pred_boxes8d, scores = decode_pred_boxes_8d(
                post_processor,
                batch_data["ego"],
                output_dict,
                score_threshold=args.score_thresh
            )

            if pred_boxes8d is not None:
                pred_boxes8d, scores = apply_nms_to_8d_boxes(
                    post_processor,
                    pred_boxes8d,
                    scores
                )

            gt_boxes = batch_data["ego"]["object_bbx_center"][0]
            gt_mask = batch_data["ego"]["object_bbx_mask"][0].bool()
            gt_boxes8d = gt_boxes[gt_mask]

            total_gt += int(gt_boxes8d.shape[0])

            if pred_boxes8d is not None:
                total_pred_after_nms += int(pred_boxes8d.shape[0])

            records, missed = match_each_gt_to_prediction(
                pred_boxes8d,
                scores,
                gt_boxes8d,
                post_processor,
                iou_thresh=args.iou_thresh
            )

            for r in records:
                r["dataset_idx"] = info["dataset_idx"]
                r["scenario_index"] = info["scenario_index"]
                r["timestamp_index"] = info["timestamp_index"]
                r["timestamp"] = info["timestamp"]
                r["scenario_path"] = info["scenario_path"]

            all_records.extend(records)

            total_matched_gt += len(records)
            total_missed_gt += len(missed)

            if len(records) > 0:
                frames_with_match += 1

    print("\n========== GT-CENTRIC SPEED EVALUATION ==========")
    print("IoU threshold:", args.iou_thresh)
    print("Score threshold:", args.score_thresh)
    print("Scenario filter:", args.scenario_indices)
    print("Evaluated batches:", evaluated_batches)
    print("Total GT boxes:", total_gt)
    print("Total predicted boxes after NMS:", total_pred_after_nms)
    print("Total matched GT boxes:", total_matched_gt)
    print("Total missed GT boxes:", total_missed_gt)
    print("Frames with at least one matched GT:", frames_with_match)

    if total_gt > 0:
        print("GT match rate:", total_matched_gt / total_gt)

    if len(all_records) == 0:
        print("No matched GT-pred pairs found.")
        print("Try lower --iou_thresh 0.3 or --score_thresh 0.05.")

        if args.csv_path is not None:
            print("CSV was not saved because there were no matched records.")

        return

    errors_mps = np.array([r["abs_error_mps"] for r in all_records], dtype=np.float32)
    errors_kmh = np.array([r["abs_error_raw_kmh"] for r in all_records], dtype=np.float32)

    print("\n========== SPEED ERROR FOR MATCHED GT BOXES ==========")
    print("Speed MAE:", float(np.mean(errors_mps)), "m/s")
    print("Speed RMSE:", float(np.sqrt(np.mean(errors_mps ** 2))), "m/s")
    print("Speed median AE:", float(np.median(errors_mps)), "m/s")
    print("Speed max AE:", float(np.max(errors_mps)), "m/s")
    print("Speed std AE:", float(np.std(errors_mps)), "m/s")

    print("\n========== RAW SPEED ERROR ==========")
    print("Raw speed appears to be km/h based on YAML displacement check.")
    print("Raw Speed MAE:", float(np.mean(errors_kmh)), "km/h")
    print("Raw Speed RMSE:", float(np.sqrt(np.mean(errors_kmh ** 2))), "km/h")
    print("Raw Speed median AE:", float(np.median(errors_kmh)), "km/h")
    print("Raw Speed max AE:", float(np.max(errors_kmh)), "km/h")
    print("Raw Speed std AE:", float(np.std(errors_kmh)), "km/h")

    print("\n========== WORST MATCHED GT SPEED ERRORS ==========")
    worst = sorted(all_records, key=lambda r: r["abs_error_mps"], reverse=True)

    for r in worst[:args.top_k]:
        print(
            f"dataset_idx={r['dataset_idx']} "
            f"scenario={r['scenario_index']} "
            f"timestamp={r['timestamp']} "
            f"gt_idx={r['gt_idx']} "
            f"pred_idx={r['pred_idx']} "
            f"iou={r['iou']:.3f} "
            f"score={r['score']:.3f} "
            f"err={r['abs_error_mps']:.2f} m/s "
            f"signed_err={r['signed_error_mps']:.2f} m/s "
            f"pred_speed={r['pred_speed_mps']:.2f} m/s "
            f"gt_speed={r['gt_speed_mps']:.2f} m/s "
            f"pred_raw={r['pred_speed_raw_kmh']:.2f} km/h "
            f"gt_raw={r['gt_speed_raw_kmh']:.2f} km/h "
            f"path={r['scenario_path']}"
        )

    if args.csv_path is not None:
        save_records_to_csv(all_records, args.csv_path)


if __name__ == "__main__":
    main()