import argparse
import os
import csv
import inspect
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils


KMH_TO_MPS = 1.0 / 3.6


# -----------------------------------------------------------------------------
# Small helpers for making this script work with BOTH:
#   1) old stacked-lidar model:      model(batch_data["ego"])
#   2) new two-frame model variants: model(batch_data["ego"]), where ego contains
#      separate current/previous tensors, OR model(cur_frame, prev_frame).
# -----------------------------------------------------------------------------


def tensor_shape_summary(x):
    if torch.is_tensor(x):
        return tuple(x.shape)
    if isinstance(x, dict):
        return {k: tensor_shape_summary(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tensor_shape_summary(v) for v in x]
    return type(x).__name__


def print_ego_debug(ego_data):
    print("\n========== EGO BATCH KEYS / SHAPES ==========")
    for k, v in ego_data.items():
        print(f"{k}: {tensor_shape_summary(v)}")


def find_first_key(d, candidates):
    for key in candidates:
        if key in d:
            return key
    return None


def extract_current_prev_inputs(ego_data):
    """
    Tries to find current-frame and previous-frame processed lidar inputs.

    Your peer's exact key names may differ. If this function fails, run:
        python opencood/tools/eval_gt_speed_two_frame.py ... --debug_batch
    then add the printed key names below.
    """
    cur_candidates = [
        "processed_lidar",             # common current-frame key
        "cur_processed_lidar",
        "current_processed_lidar",
        "processed_lidar_cur",
        "processed_lidar_t",
        "cur_lidar",
        "current_lidar",
    ]

    prev_candidates = [
        "prev_processed_lidar",
        "processed_lidar_prev",
        "previous_processed_lidar",
        "processed_lidar_t_1",
        "processed_lidar_t_minus_1",
        "prev_lidar",
        "previous_lidar",
    ]

    cur_key = find_first_key(ego_data, cur_candidates)
    prev_key = find_first_key(ego_data, prev_candidates)

    if cur_key is None or prev_key is None:
        raise KeyError(
            "Could not automatically find current/previous lidar inputs.\n"
            f"Available ego keys: {list(ego_data.keys())}\n"
            f"Tried current keys: {cur_candidates}\n"
            f"Tried previous keys: {prev_candidates}\n"
            "Fix: add your peer's key names to extract_current_prev_inputs()."
        )

    return ego_data[cur_key], ego_data[prev_key], cur_key, prev_key


def run_model_flexible(model, ego_data, debug_model_call=False):
    """
    Runs inference for different possible two-frame implementations.

    Priority:
      A) model(ego_data)
         This is most likely if the new dataset stores both frames inside ego_data.
      B) model(cur_input, prev_input)
         This handles models whose forward() explicitly takes two frame inputs.
      C) model(ego_data, prev_input)
         Less common, but included for peer code variants.

    Returns:
      output_dict with at least psm/rm, or nested output containing psm/rm.
    """
    errors = []

    try:
        if debug_model_call:
            print("Trying model(ego_data)")
        return model(ego_data)
    except Exception as e:
        errors.append(("model(ego_data)", repr(e)))

    cur_input, prev_input, cur_key, prev_key = extract_current_prev_inputs(ego_data)

    try:
        if debug_model_call:
            print(f"Trying model(ego_data[{cur_key}], ego_data[{prev_key}])")
        return model(cur_input, prev_input)
    except Exception as e:
        errors.append((f"model({cur_key}, {prev_key})", repr(e)))

    try:
        if debug_model_call:
            print(f"Trying model(ego_data, ego_data[{prev_key}])")
        return model(ego_data, prev_input)
    except Exception as e:
        errors.append((f"model(ego_data, {prev_key})", repr(e)))

    msg = "Could not run model with any known calling pattern. Tried:\n"
    for call, err in errors:
        msg += f"  - {call}: {err}\n"
    msg += "\nCheck your peer model's forward() signature in opencood/models/*.py."
    raise RuntimeError(msg)


def normalize_output_dict(output):
    """
    OpenCOOD models normally return {'psm': ..., 'rm': ...}.
    Some models return {'ego': {'psm': ..., 'rm': ...}} or a tuple/list.
    This normalizes the output so decode_pred_boxes_8d can use it.
    """
    if isinstance(output, dict):
        if "psm" in output and "rm" in output:
            return output
        if "ego" in output and isinstance(output["ego"], dict):
            if "psm" in output["ego"] and "rm" in output["ego"]:
                return output["ego"]
        # Search one level deep.
        for _, v in output.items():
            if isinstance(v, dict) and "psm" in v and "rm" in v:
                return v

    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return normalize_output_dict(item)
            except Exception:
                pass

    raise KeyError(
        "Model output does not contain psm/rm in a recognized format. "
        f"Output summary: {tensor_shape_summary(output)}"
    )


def decode_pred_boxes_8d(post_processor, ego_data, output_dict, score_threshold=None):
    """
    Decode model output to boxes:
        [x, y, z, h, w, l, yaw, speed]

    If box_code_size > 8, this keeps all columns for decoding/NMS but evaluates
    column 7 as speed by default.
    """
    output_dict = normalize_output_dict(output_dict)
    box_code_size = int(post_processor.params.get("box_code_size", output_dict["rm"].shape[1] // 2 if output_dict["rm"].dim() == 4 else 8))

    anchor_box = ego_data["anchor_box"]
    prob = output_dict["psm"]
    reg = output_dict["rm"]

    prob = torch.sigmoid(prob.permute(0, 2, 3, 1))
    prob = prob.reshape(1, -1)

    batch_box3d = post_processor.delta_to_boxes3d(reg, anchor_box)

    if score_threshold is None:
        score_threshold = post_processor.params["target_args"]["score_threshold"]

    mask = torch.gt(prob, score_threshold).view(1, -1)
    mask_reg = mask.unsqueeze(2).repeat(1, 1, box_code_size)

    boxes = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, box_code_size)
    scores = torch.masked_select(prob[0], mask[0])

    if boxes.shape[0] == 0:
        return None, None

    return boxes, scores


def apply_nms_to_boxes(post_processor, boxes, scores):
    if boxes is None or boxes.shape[0] == 0:
        return None, None

    boxes7d = boxes[:, :7]
    pred_corners = box_utils.boxes_to_corners_3d(
        boxes7d,
        order=post_processor.params["order"]
    )

    keep_idx = box_utils.nms_rotated(
        pred_corners,
        scores,
        post_processor.params["nms_thresh"]
    )

    return boxes[keep_idx], scores[keep_idx]


def get_gt_boxes8d(ego_data, speed_idx=7):
    """
    Gets GT boxes containing speed.

    Preferred shapes:
      - object_bbx_center: [B, max_obj, >=8]
      - object_bbx_center_8d / object_bbx_center_8d_debug: [B, max_obj, >=8]

    If your new dataset stores speed under a different key, add it to gt_candidates.
    """
    gt_candidates = [
        "object_bbx_center",
        "object_bbx_center_8d",
        "object_bbx_center_8d_debug",
        "object_bbx_center_with_speed",
        "object_bbx_center_speed",
    ]

    gt_key = None
    gt_boxes = None
    for key in gt_candidates:
        if key in ego_data and torch.is_tensor(ego_data[key]):
            candidate = ego_data[key]
            if candidate.dim() >= 3 and candidate.shape[-1] > speed_idx:
                gt_key = key
                gt_boxes = candidate[0]
                break

    if gt_boxes is None:
        raise KeyError(
            "Could not find GT boxes with speed column.\n"
            f"Available ego keys: {list(ego_data.keys())}\n"
            "Need one GT tensor with shape [B, max_obj, >=8], where column 7 is speed."
        )

    if "object_bbx_mask" not in ego_data:
        raise KeyError("ego_data is missing object_bbx_mask, needed to filter valid GT boxes.")

    gt_mask = ego_data["object_bbx_mask"][0].bool()
    return gt_boxes[gt_mask], gt_key


def standup_iou_matrix(pred_boxes, gt_boxes, post_processor):
    pred_boxes7d = pred_boxes[:, :7]
    gt_boxes7d = gt_boxes[:, :7]

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


def match_each_gt_to_prediction(pred_boxes, scores, gt_boxes, post_processor, iou_thresh=0.3, speed_idx=7, speed_unit="kmh"):
    records = []
    missed_gt_indices = []

    if gt_boxes is None or gt_boxes.shape[0] == 0:
        return records, missed_gt_indices

    if pred_boxes is None or pred_boxes.shape[0] == 0:
        return records, list(range(gt_boxes.shape[0]))

    if pred_boxes.shape[-1] <= speed_idx:
        raise ValueError(f"Predicted boxes only have {pred_boxes.shape[-1]} columns, but speed_idx={speed_idx}.")
    if gt_boxes.shape[-1] <= speed_idx:
        raise ValueError(f"GT boxes only have {gt_boxes.shape[-1]} columns, but speed_idx={speed_idx}.")

    iou = standup_iou_matrix(pred_boxes, gt_boxes, post_processor)
    used_pred = set()

    if speed_unit == "kmh":
        to_mps = KMH_TO_MPS
    elif speed_unit == "mps":
        to_mps = 1.0
    else:
        raise ValueError("speed_unit must be either 'kmh' or 'mps'")

    for gt_idx in range(gt_boxes.shape[0]):
        pred_order = np.argsort(-iou[:, gt_idx])
        matched = False

        for pred_idx in pred_order:
            pred_idx = int(pred_idx)
            if pred_idx in used_pred:
                continue

            best_iou = float(iou[pred_idx, gt_idx])
            if best_iou < iou_thresh:
                break

            pred_speed_raw = float(pred_boxes[pred_idx, speed_idx].detach().cpu())
            gt_speed_raw = float(gt_boxes[gt_idx, speed_idx].detach().cpu())

            pred_speed_mps = pred_speed_raw * to_mps
            gt_speed_mps = gt_speed_raw * to_mps

            signed_error_raw = pred_speed_raw - gt_speed_raw
            abs_error_raw = abs(signed_error_raw)
            signed_error_mps = pred_speed_mps - gt_speed_mps
            abs_error_mps = abs(signed_error_mps)

            records.append({
                "gt_idx": int(gt_idx),
                "pred_idx": int(pred_idx),
                "iou": best_iou,
                "score": float(scores[pred_idx].detach().cpu()),
                "pred_speed_raw": pred_speed_raw,
                "gt_speed_raw": gt_speed_raw,
                "signed_error_raw": signed_error_raw,
                "abs_error_raw": abs_error_raw,
                "pred_speed_mps": pred_speed_mps,
                "gt_speed_mps": gt_speed_mps,
                "signed_error_mps": signed_error_mps,
                "abs_error_mps": abs_error_mps,
                "pred_box": pred_boxes[pred_idx].detach().cpu().numpy(),
                "gt_box": gt_boxes[gt_idx].detach().cpu().numpy(),
            })

            used_pred.add(pred_idx)
            matched = True
            break

        if not matched:
            missed_gt_indices.append(int(gt_idx))

    return records, missed_gt_indices


def get_scenario_ranges(dataset):
    ranges = []
    len_record = list(dataset.len_record)
    if len(len_record) == 0:
        return ranges

    dataset_length = len(dataset)

    if int(len_record[-1]) == dataset_length:
        prev_end = 0
        for scenario_idx, end_idx in enumerate(len_record):
            end_idx = int(end_idx)
            ranges.append((scenario_idx, int(prev_end), end_idx))
            prev_end = end_idx
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
    timestamp_key = dataset.return_timestamp_key(scenario_database, timestamp_index)
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


def build_subset_for_scenarios(dataset, scenario_indices, list_only=False):
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

    if len(selected_indices) == 0 and not list_only:
        valid = [s for s, _, _ in scenario_ranges]
        raise ValueError(
            f"No frames found for scenario_indices={scenario_indices}. "
            f"Valid scenario indices are: {valid}"
        )

    return Subset(dataset, selected_indices), selected_indices


def save_records_to_csv(records, csv_path, speed_unit):
    if len(records) == 0:
        print("No matched records to save to CSV.")
        return

    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    fieldnames = [
        "dataset_idx", "scenario_index", "timestamp_index", "timestamp", "scenario_path",
        "gt_idx", "pred_idx", "iou", "score",
        "speed_unit_raw",
        "gt_speed_raw", "pred_speed_raw", "signed_error_raw", "abs_error_raw",
        "gt_speed_mps", "pred_speed_mps", "signed_error_mps", "abs_error_mps",
        "gt_x", "gt_y", "gt_z", "gt_h", "gt_w", "gt_l", "gt_yaw",
        "pred_x", "pred_y", "pred_z", "pred_h", "pred_w", "pred_l", "pred_yaw",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            gt_box = r["gt_box"]
            pred_box = r["pred_box"]
            writer.writerow({
                "dataset_idx": r["dataset_idx"],
                "scenario_index": r["scenario_index"],
                "timestamp_index": r["timestamp_index"],
                "timestamp": r["timestamp"],
                "scenario_path": r["scenario_path"],
                "gt_idx": r["gt_idx"],
                "pred_idx": r["pred_idx"],
                "iou": r["iou"],
                "score": r["score"],
                "speed_unit_raw": speed_unit,
                "gt_speed_raw": r["gt_speed_raw"],
                "pred_speed_raw": r["pred_speed_raw"],
                "signed_error_raw": r["signed_error_raw"],
                "abs_error_raw": r["abs_error_raw"],
                "gt_speed_mps": r["gt_speed_mps"],
                "pred_speed_mps": r["pred_speed_mps"],
                "signed_error_mps": r["signed_error_mps"],
                "abs_error_mps": r["abs_error_mps"],
                "gt_x": gt_box[0], "gt_y": gt_box[1], "gt_z": gt_box[2],
                "gt_h": gt_box[3], "gt_w": gt_box[4], "gt_l": gt_box[5], "gt_yaw": gt_box[6],
                "pred_x": pred_box[0], "pred_y": pred_box[1], "pred_z": pred_box[2],
                "pred_h": pred_box[3], "pred_w": pred_box[4], "pred_l": pred_box[5], "pred_yaw": pred_box[6],
            })

    print(f"\nSaved CSV to: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--hypes_yaml", default=None)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--score_thresh", type=float, default=None)
    parser.add_argument("--iou_thresh", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--scenario_indices", type=int, nargs="+", default=None)
    parser.add_argument("--list_scenarios", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--debug_batch", action="store_true", help="Print ego keys/shapes and stop before evaluation.")
    parser.add_argument("--debug_model_call", action="store_true", help="Print which model call pattern is attempted.")
    parser.add_argument("--speed_idx", type=int, default=7, help="Column index for speed in decoded boxes and GT boxes.")
    parser.add_argument(
        "--speed_unit",
        choices=["kmh", "mps"],
        default="kmh",
        help="Raw speed unit stored in prediction/GT. Old OPV2V velocity labels were treated as km/h."
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
        args.scenario_indices,
        list_only=args.list_scenarios
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
        drop_last=False,
    )

    print("Dataloader length:", len(data_loader))

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
    gt_key_used = None

    with torch.no_grad():
        for loader_idx, batch_data in enumerate(tqdm(data_loader)):
            if args.max_batches > 0 and evaluated_batches >= args.max_batches:
                break
            if batch_data is None:
                continue

            original_idx = selected_indices[loader_idx] if selected_indices is not None else loader_idx
            info = get_dataset_sample_info(dataset, original_idx)
            evaluated_batches += 1

            batch_data = train_utils.to_device(batch_data, torch.device("cuda"))
            ego_data = batch_data["ego"]

            if args.debug_batch:
                print_ego_debug(ego_data)
                print("\nModel forward signature:")
                try:
                    print(inspect.signature(model.forward))
                except Exception as e:
                    print("Could not inspect forward signature:", repr(e))
                print("\nStopping because --debug_batch was set.")
                return

            output_raw = run_model_flexible(
                model,
                ego_data,
                debug_model_call=args.debug_model_call
            )
            output_dict = normalize_output_dict(output_raw)

            pred_boxes, scores = decode_pred_boxes_8d(
                post_processor,
                ego_data,
                output_dict,
                score_threshold=args.score_thresh,
            )

            if pred_boxes is not None:
                pred_boxes, scores = apply_nms_to_boxes(post_processor, pred_boxes, scores)

            gt_boxes, gt_key = get_gt_boxes8d(ego_data, speed_idx=args.speed_idx)
            gt_key_used = gt_key

            total_gt += int(gt_boxes.shape[0])
            if pred_boxes is not None:
                total_pred_after_nms += int(pred_boxes.shape[0])

            records, missed = match_each_gt_to_prediction(
                pred_boxes,
                scores,
                gt_boxes,
                post_processor,
                iou_thresh=args.iou_thresh,
                speed_idx=args.speed_idx,
                speed_unit=args.speed_unit,
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
    print("Speed column index:", args.speed_idx)
    print("Raw speed unit:", args.speed_unit)
    print("GT key used:", gt_key_used)
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
    errors_raw = np.array([r["abs_error_raw"] for r in all_records], dtype=np.float32)

    print("\n========== SPEED ERROR FOR MATCHED GT BOXES ==========")
    print("Speed MAE:", float(np.mean(errors_mps)), "m/s")
    print("Speed RMSE:", float(np.sqrt(np.mean(errors_mps ** 2))), "m/s")
    print("Speed median AE:", float(np.median(errors_mps)), "m/s")
    print("Speed max AE:", float(np.max(errors_mps)), "m/s")
    print("Speed std AE:", float(np.std(errors_mps)), "m/s")

    print("\n========== RAW SPEED ERROR ==========")
    print(f"Raw speed interpreted as {args.speed_unit}.")
    print("Raw Speed MAE:", float(np.mean(errors_raw)), args.speed_unit)
    print("Raw Speed RMSE:", float(np.sqrt(np.mean(errors_raw ** 2))), args.speed_unit)
    print("Raw Speed median AE:", float(np.median(errors_raw)), args.speed_unit)
    print("Raw Speed max AE:", float(np.max(errors_raw)), args.speed_unit)
    print("Raw Speed std AE:", float(np.std(errors_raw)), args.speed_unit)

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
            f"pred_raw={r['pred_speed_raw']:.2f} {args.speed_unit} "
            f"gt_raw={r['gt_speed_raw']:.2f} {args.speed_unit} "
            f"path={r['scenario_path']}"
        )

    if args.csv_path is not None:
        save_records_to_csv(all_records, args.csv_path, args.speed_unit)


if __name__ == "__main__":
    main()
