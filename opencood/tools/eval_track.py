"""
Global-ID tracking eval: assign persistent IDs to predicted boxes and verify
they follow the correct GT vehicle (by object_id) using GT t+1 positions.
"""
import argparse
import os
import sys

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
from opencood.utils.track_utils import (
    KalmanHungarianTracker, compute_dt, standup_iou_matrix, greedy_iou_associate,
)


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


def to_numpy_boxes(boxes):
    if boxes is None:
        return None
    if torch.is_tensor(boxes):
        return boxes.detach().cpu().numpy()
    return np.asarray(boxes)


def greedy_match_with_iou(boxes_a8d, boxes_b8d, post_processor, iou_thresh):
    """Return matches as list of (idx_a, idx_b, iou)."""
    a = to_numpy_boxes(boxes_a8d)
    b = to_numpy_boxes(boxes_b8d)
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return []

    iou = standup_iou_matrix(a, b, post_processor)
    pairs, _, _ = greedy_iou_associate(iou, iou_thresh)
    return [(i, j, float(iou[i, j])) for i, j in pairs]


def build_gt_global_map(object_ids, gt_boxes8d):
    """GT global ID (vehicle object_id) -> (gt_idx, box8d)."""
    gt_np = to_numpy_boxes(gt_boxes8d)
    mapping = {}
    for gt_idx, gid in enumerate(object_ids):
        mapping[str(gid)] = (gt_idx, gt_np[gt_idx])
    return mapping


def build_pred_global_map(track_ids, det_boxes8d, gt_boxes8d, object_ids,
                          post_processor, iou_thresh):
    """
    Pred global ID (tracker ID) -> {det_idx, box, gt_global_id or None, iou}.
    """
    pred_np = to_numpy_boxes(det_boxes8d)
    matches = greedy_match_with_iou(det_boxes8d, gt_boxes8d, post_processor, iou_thresh)

    det_to_gt = {}
    det_to_iou = {}
    for det_idx, gt_idx, iou_val in matches:
        if gt_idx < len(object_ids):
            det_to_gt[det_idx] = str(object_ids[gt_idx])
            det_to_iou[det_idx] = iou_val

    mapping = {}
    for det_idx, gid in enumerate(track_ids):
        mapping[int(gid)] = {
            "det_idx": det_idx,
            "box": pred_np[det_idx],
            "gt_global_id": det_to_gt.get(det_idx),
            "gt_iou": det_to_iou.get(det_idx, 0.0),
        }
    return mapping


class TrackEvalStats:
    def __init__(self):
        self.frames = 0
        self.frame_pairs = 0
        self.total_dets = 0
        self.max_global_id = 0
        self.gt_links = 0
        self.correct = 0
        self.id_switch = 0
        self.track_lost = 0
        self.no_track_at_t = 0
        self.misaligned = 0
        self.center_dists = []
        self.t1_ious = []
        self.gt_to_pred_ids = {}
        self.episode_fragmentation = []  # per GT vehicle: num pred IDs used
        self._scenario_gt_map = {}

    def start_scenario(self):
        self._scenario_gt_map = {}

    def end_scenario(self):
        for gt_id, pred_ids in self._scenario_gt_map.items():
            self.episode_fragmentation.append(len(pred_ids))

    def record_pred_gt(self, gt_global_id, pred_global_id):
        self._scenario_gt_map.setdefault(gt_global_id, set()).add(pred_global_id)

    def eval_link(self, gt_global_id, gt_box_t1, pred_t, pred_t1,
                  post_processor, iou_thresh):
        self.gt_links += 1

        pred_gid_t = None
        for gid, info in pred_t.items():
            if info["gt_global_id"] == gt_global_id:
                pred_gid_t = gid
                break

        if pred_gid_t is None:
            self.no_track_at_t += 1
            return

        self.record_pred_gt(gt_global_id, pred_gid_t)

        if pred_gid_t not in pred_t1:
            self.track_lost += 1
            return

        info_t1 = pred_t1[pred_gid_t]
        if info_t1["gt_global_id"] != gt_global_id:
            self.id_switch += 1
            return

        iou = standup_iou_matrix(
            info_t1["box"].reshape(1, -1),
            gt_box_t1.reshape(1, -1),
            post_processor,
        )[0, 0]

        if iou < iou_thresh:
            self.misaligned += 1
            return

        self.correct += 1
        self.t1_ious.append(float(iou))
        pred_xy = info_t1["box"][:2]
        gt_xy = gt_box_t1[:2]
        self.center_dists.append(float(np.linalg.norm(pred_xy - gt_xy)))

    def report(self, track_iou_thresh, gt_iou_thresh):
        print("\n========== GLOBAL ID TRACKING EVAL ==========")
        print(f"Frames: {self.frames}  |  Frame pairs: {self.frame_pairs}")
        print(f"Track-det IoU thresh: {track_iou_thresh}  |  GT IoU thresh: {gt_iou_thresh}")
        print("Tracker: Kalman (CV) + Hungarian (IoU + velocity cost)")
        print(f"Detections with global ID: {self.total_dets}  |  Max global ID: {self.max_global_id}")

        print("\n--- GT t→t+1 links (same vehicle both frames) ---")
        print(f"Total GT links: {self.gt_links}")

        tracked_at_t = self.gt_links - self.no_track_at_t
        print(f"No pred matched GT at t: {self.no_track_at_t}")
        print(f"Pred tracked at t: {tracked_at_t}")

        print("\n--- Global ID preservation (verified at GT t+1 position) ---")
        print(f"Correct (ID kept + matched GT at t+1 + IoU ok): {self.correct}")
        print(f"ID switch (wrong GT at t+1): {self.id_switch}")
        print(f"Track lost (ID missing at t+1): {self.track_lost}")
        print(f"Misaligned (right GT, IoU with GT t+1 box too low): {self.misaligned}")

        if tracked_at_t > 0:
            print(f"ID preservation rate (correct / tracked@t): {self.correct / tracked_at_t:.3f}")
        if self.gt_links > 0:
            print(f"GT link recall (correct / all GT links): {self.correct / self.gt_links:.3f}")
        denom = self.correct + self.id_switch + self.misaligned
        if denom > 0:
            print(f"Identity accuracy (correct / resolved@t+1): {self.correct / denom:.3f}")

        if self.center_dists:
            d = np.array(self.center_dists)
            print(f"\n--- Position at t+1 when ID correct ---")
            print(f"Center dist (m): mean={d.mean():.2f}  median={np.median(d):.2f}  max={d.max():.2f}")
        if self.t1_ious:
            i = np.array(self.t1_ious)
            print(f"IoU(pred, GT t+1): mean={i.mean():.3f}  median={np.median(i):.3f}")

        if self.episode_fragmentation:
            f = np.array(self.episode_fragmentation)
            print(f"\n--- Episode fragmentation (pred global IDs per GT vehicle) ---")
            print(f"Mean IDs/GT vehicle: {f.mean():.2f}  |  >1 ID: {(f > 1).sum()} / {len(f)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--hypes_yaml", default=None)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--score_thresh", type=float, default=None)
    parser.add_argument("--iou_thresh", type=float, default=0.3,
                        help="IoU for track-det association")
    parser.add_argument("--gt_iou_thresh", type=float, default=0.3,
                        help="IoU for det-GT matching and t+1 position verify")
    parser.add_argument("--max_age", type=int, default=3,
                        help="Frames to keep unmatched Kalman tracks")
    parser.add_argument("--vel_weight", type=float, default=0.3,
                        help="Velocity mismatch weight in Hungarian cost")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    hypes_yaml = args.hypes_yaml or os.path.join(args.model_dir, "config.yaml")
    hypes = yaml_utils.load_yaml(hypes_yaml, None)
    dataset = build_dataset(hypes, visualize=False, train=False)

    data_loader = DataLoader(
        dataset, batch_size=1, num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False, pin_memory=False, drop_last=False,
    )

    model = train_utils.create_model(hypes)
    model.cuda()
    model.eval()
    _, model = train_utils.load_saved_model(args.model_dir, model)
    post_processor = dataset.post_processor

    tracker = KalmanHungarianTracker(
        iou_thresh=args.iou_thresh,
        max_age=args.max_age,
        vel_weight=args.vel_weight,
    )
    stats = TrackEvalStats()

    prev_info = None
    prev_gt_map = None
    prev_pred_map = None
    prev_scenario = None

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

            gt_boxes = batch_data["ego"]["object_bbx_center"][0]
            gt_mask = batch_data["ego"]["object_bbx_mask"][0].bool()
            gt_boxes8d = gt_boxes[gt_mask]
            object_ids = batch_data["ego"].get("object_ids", [])
            gt_map = build_gt_global_map(object_ids, gt_boxes8d)

            # Scenario boundary
            if prev_scenario is not None and info["scenario_index"] != prev_scenario:
                stats.end_scenario()
                tracker.reset()
                prev_gt_map = None
                prev_pred_map = None
            if prev_scenario != info["scenario_index"]:
                stats.start_scenario()
            prev_scenario = info["scenario_index"]

            if pred_boxes8d is None:
                prev_info = info
                prev_gt_map = gt_map
                prev_pred_map = None
                continue

            if prev_info is not None and info["scenario_index"] == prev_info["scenario_index"]:
                tracker.set_dt(compute_dt(prev_info["timestamp"], info["timestamp"]))

            pred_np = pred_boxes8d.detach().cpu().numpy()
            score_np = scores.detach().cpu().numpy()
            global_ids = tracker.update(pred_np, score_np, post_processor)

            stats.frames += 1
            stats.total_dets += len(global_ids)
            if len(global_ids):
                stats.max_global_id = max(stats.max_global_id, int(global_ids.max()))

            pred_map = build_pred_global_map(
                global_ids, pred_boxes8d, gt_boxes8d, object_ids,
                post_processor, args.gt_iou_thresh,
            )

            # Evaluate t -> t+1 links within same scenario
            if (prev_gt_map is not None and prev_pred_map is not None
                    and prev_info is not None
                    and info["scenario_index"] == prev_info["scenario_index"]):
                stats.frame_pairs += 1
                common_gt_ids = set(prev_gt_map.keys()) & set(gt_map.keys())
                for gt_gid in common_gt_ids:
                    _, gt_box_t1 = gt_map[gt_gid]
                    stats.eval_link(
                        gt_gid, gt_box_t1, prev_pred_map, pred_map,
                        post_processor, args.gt_iou_thresh,
                    )

            prev_info = info
            prev_gt_map = gt_map
            prev_pred_map = pred_map

    if prev_scenario is not None:
        stats.end_scenario()

    stats.report(args.iou_thresh, args.gt_iou_thresh)


if __name__ == "__main__":
    main()
