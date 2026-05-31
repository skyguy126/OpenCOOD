"""
Assign persistent bbox IDs across frames using predicted speed + greedy IoU association.
"""
import argparse
import os
import sys

# Prefer this repo over a pip-installed OpenCOOD copy (e.g. /media/Disk2/OpenCOOD).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils
from opencood.utils.track_utils import SimpleTracker, compute_dt, standup_iou_matrix, greedy_iou_associate


def decode_pred_boxes_8d(post_processor, ego_data, output_dict, score_threshold=None):
    box_code_size = post_processor.params.get("box_code_size", 8)
    anchor_box = ego_data["anchor_box"]
    prob = torch.sigmoid(output_dict["psm"].permute(0, 2, 3, 1)).reshape(1, -1)
    batch_box3d = post_processor.delta_to_boxes3d(output_dict["rm"], anchor_box)

    if score_threshold is None:
        score_threshold = post_processor.params["target_args"]["score_threshold"]

    mask = torch.gt(prob, score_threshold).view(1, -1)
    mask_reg = mask.unsqueeze(2).repeat(1, 1, box_code_size)
    boxes8d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, box_code_size)
    scores = torch.masked_select(prob[0], mask[0])

    if boxes8d.shape[0] == 0:
        return None, None
    return boxes8d, scores


def apply_nms_to_8d_boxes(post_processor, boxes8d, scores):
    if boxes8d is None or boxes8d.shape[0] == 0:
        return None, None

    corners = box_utils.boxes_to_corners_3d(
        boxes8d[:, :7], order=post_processor.params["order"]
    )
    keep_idx = box_utils.nms_rotated(
        corners, scores, post_processor.params["nms_thresh"]
    )
    return boxes8d[keep_idx], scores[keep_idx]


def get_sample_info(dataset, idx):
    """Map dataloader index to scenario/timestamp (same logic as BaseDataset)."""
    scenario_index = 0
    for i, end_idx in enumerate(dataset.len_record):
        if idx < end_idx:
            scenario_index = i
            break

    scenario_database = dataset.scenario_database[scenario_index]
    timestamp_index = idx if scenario_index == 0 else \
        idx - dataset.len_record[scenario_index - 1]
    timestamp = dataset.return_timestamp_key(scenario_database, timestamp_index)

    return {
        "dataset_idx": idx,
        "scenario_index": scenario_index,
        "timestamp_index": timestamp_index,
        "timestamp": timestamp,
    }


def match_dets_to_gt(det_boxes8d, gt_boxes8d, post_processor, iou_thresh=0.3):
    """Return list of (det_idx, gt_idx) using greedy IoU."""
    if det_boxes8d is None or gt_boxes8d is None:
        return []
    if len(det_boxes8d) == 0 or len(gt_boxes8d) == 0:
        return []

    det_np = det_boxes8d.detach().cpu().numpy() if torch.is_tensor(det_boxes8d) else det_boxes8d
    gt_np = gt_boxes8d.detach().cpu().numpy() if torch.is_tensor(gt_boxes8d) else gt_boxes8d
    iou = standup_iou_matrix(det_np, gt_np, post_processor)
    matches, _, _ = greedy_iou_associate(iou, iou_thresh)
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--hypes_yaml", default=None)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--score_thresh", type=float, default=None)
    parser.add_argument("--iou_thresh", type=float, default=0.3,
                        help="IoU threshold for track-det association")
    parser.add_argument("--gt_iou_thresh", type=float, default=0.3,
                        help="IoU threshold for det-GT matching in eval")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    hypes_yaml = args.hypes_yaml or os.path.join(args.model_dir, "config.yaml")
    hypes = yaml_utils.load_yaml(hypes_yaml, None)
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

    model = train_utils.create_model(hypes)
    model.cuda()
    model.eval()
    _, model = train_utils.load_saved_model(args.model_dir, model)
    post_processor = dataset.post_processor

    tracker = SimpleTracker(iou_thresh=args.iou_thresh)
    prev_info = None
    prev_track_ids = None
    prev_det_to_gt = {}  # det_idx -> gt object_id string

    total_dets = 0
    total_tracks = 0
    id_switches = 0
    consistent_links = 0
    frames_processed = 0

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(data_loader)):
            if args.max_batches > 0 and i >= args.max_batches:
                break
            if batch_data is None:
                continue

            info = get_sample_info(dataset, i)
            batch_data = train_utils.to_device(batch_data, torch.device("cuda"))
            output_dict = model(batch_data["ego"])

            pred_boxes8d, scores = decode_pred_boxes_8d(
                post_processor, batch_data["ego"], output_dict,
                score_threshold=args.score_thresh,
            )
            if pred_boxes8d is not None:
                pred_boxes8d, scores = apply_nms_to_8d_boxes(
                    post_processor, pred_boxes8d, scores
                )

            if pred_boxes8d is None:
                tracker.reset()
                prev_info = info
                prev_track_ids = None
                prev_det_to_gt = {}
                continue

            pred_np = pred_boxes8d.detach().cpu().numpy()
            score_np = scores.detach().cpu().numpy()

            if prev_info is not None and info["scenario_index"] != prev_info["scenario_index"]:
                tracker.reset()
                prev_track_ids = None
                prev_det_to_gt = {}
            elif prev_info is not None:
                tracker.set_dt(compute_dt(prev_info["timestamp"], info["timestamp"]))

            track_ids = tracker.update(pred_np, score_np, post_processor)
            total_dets += len(track_ids)
            total_tracks = max(total_tracks, int(track_ids.max()) if len(track_ids) else 0)
            frames_processed += 1

            gt_boxes = batch_data["ego"]["object_bbx_center"][0]
            gt_mask = batch_data["ego"]["object_bbx_mask"][0].bool()
            gt_boxes8d = gt_boxes[gt_mask]
            object_ids = batch_data["ego"].get("object_ids", [])

            det_to_gt = {}
            for det_idx, gt_idx in match_dets_to_gt(
                pred_boxes8d, gt_boxes8d, post_processor, args.gt_iou_thresh
            ):
                if gt_idx < len(object_ids):
                    det_to_gt[det_idx] = str(object_ids[gt_idx])

            if prev_track_ids is not None and prev_info["scenario_index"] == info["scenario_index"]:
                prev_tid_to_gt = {}
                for det_idx, tid in enumerate(prev_track_ids):
                    if det_idx in prev_det_to_gt:
                        prev_tid_to_gt[int(tid)] = prev_det_to_gt[det_idx]

                for det_idx, tid in enumerate(track_ids):
                    gt_id = det_to_gt.get(det_idx)
                    prev_gt_id = prev_tid_to_gt.get(int(tid))
                    if gt_id is None or prev_gt_id is None:
                        continue
                    if gt_id == prev_gt_id:
                        consistent_links += 1
                    else:
                        id_switches += 1

            prev_info = info
            prev_track_ids = track_ids
            prev_det_to_gt = det_to_gt

    print("\n========== BBOX ID TRACKING ==========")
    print("Frames processed:", frames_processed)
    print("Total detections with IDs:", total_dets)
    print("Max track ID assigned:", total_tracks)
    print("Track-det IoU threshold:", args.iou_thresh)
    print("Det-GT IoU threshold:", args.gt_iou_thresh)
    print("GT-consistent track links (t -> t+1):", consistent_links)
    print("ID switches (same track, different GT object):", id_switches)
    if consistent_links + id_switches > 0:
        accuracy = consistent_links / (consistent_links + id_switches)
        print("Track consistency:", f"{accuracy:.3f}")


if __name__ == "__main__":
    main()
