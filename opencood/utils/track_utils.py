"""
Simple multi-frame bbox tracking using predicted speed and greedy IoU association.
"""
import numpy as np
import torch

from opencood.utils import box_utils

KMH_TO_MPS = 1.0 / 3.6
DEFAULT_DT = 0.1  # OPV2V is 10 Hz


def compute_dt(prev_timestamp, cur_timestamp, default_dt=DEFAULT_DT):
    """Frame interval in seconds from consecutive timestamp keys."""
    try:
        return (int(cur_timestamp) - int(prev_timestamp)) / 10.0
    except (ValueError, TypeError):
        return default_dt


def predict_boxes_next_frame(boxes8d, dt=DEFAULT_DT):
    """
    Propagate boxes to t+1 using predicted scalar speed (km/h) and yaw.

    boxes8d: (N, 8) [x, y, z, h, w, l, yaw, speed_kmh]
    """
    if boxes8d is None or len(boxes8d) == 0:
        return boxes8d

    predicted = np.array(boxes8d, dtype=np.float32, copy=True)
    speed_mps = predicted[:, 7] * KMH_TO_MPS
    yaw = predicted[:, 6]
    predicted[:, 0] += speed_mps * np.cos(yaw) * dt
    predicted[:, 1] += speed_mps * np.sin(yaw) * dt
    return predicted


def standup_iou_matrix(boxes_a8d, boxes_b8d, post_processor):
    """BEV standup-box IoU matrix, shape (len(a), len(b))."""
    if len(boxes_a8d) == 0 or len(boxes_b8d) == 0:
        return np.zeros((len(boxes_a8d), len(boxes_b8d)), dtype=np.float32)

    def to_standup(boxes8d):
        boxes7d = torch.from_numpy(boxes8d[:, :7].astype(np.float32))
        corners = box_utils.boxes_to_corners_3d(
            boxes7d, order=post_processor.params["order"]
        )
        standup = box_utils.corner_to_standup_box_torch(corners)
        return standup.detach().cpu().numpy()

    a = to_standup(boxes_a8d)
    b = to_standup(boxes_b8d)
    iou = np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    for i in range(a.shape[0]):
        ax1, ay1, ax2, ay2 = a[i]
        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        for j in range(b.shape[0]):
            bx1, by1, bx2, by2 = b[j]
            b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = a_area + b_area - inter
            if union > 0:
                iou[i, j] = inter / union

    return iou


def greedy_iou_associate(iou, iou_thresh=0.3):
    """
    Greedy highest-IoU-first matching.

    Returns:
        matches: list of (row_idx, col_idx)
        unmatched_rows: list
        unmatched_cols: list
    """
    if iou.size == 0:
        return [], list(range(iou.shape[0])), list(range(iou.shape[1]))

    candidates = []
    for r in range(iou.shape[0]):
        for c in range(iou.shape[1]):
            if iou[r, c] >= iou_thresh:
                candidates.append((float(iou[r, c]), r, c))

    candidates.sort(reverse=True, key=lambda x: x[0])

    used_rows = set()
    used_cols = set()
    matches = []

    for _, r, c in candidates:
        if r in used_rows or c in used_cols:
            continue
        matches.append((r, c))
        used_rows.add(r)
        used_cols.add(c)

    unmatched_rows = [r for r in range(iou.shape[0]) if r not in used_rows]
    unmatched_cols = [c for c in range(iou.shape[1]) if c not in used_cols]
    return matches, unmatched_rows, unmatched_cols


class SimpleTracker:
    """
    Assign persistent IDs across frames.

    At each step:
      1. Predict active tracks to t+1 with predicted speed.
      2. Greedy IoU match predicted boxes to current detections.
      3. Keep matched IDs, drop lost tracks, spawn IDs for new detections.
    """

    def __init__(self, iou_thresh=0.3, dt=DEFAULT_DT):
        self.iou_thresh = iou_thresh
        self.dt = dt
        self.next_id = 1
        self.tracks = []  # each: {id, box (8,), score}

    def reset(self):
        self.next_id = 1
        self.tracks = []

    def set_dt(self, dt):
        self.dt = dt

    def update(self, det_boxes8d, det_scores, post_processor):
        """
        Args:
            det_boxes8d: np.ndarray (N, 8)
            det_scores: np.ndarray (N,)

        Returns:
            track_ids: np.ndarray (N,) int, -1 if no detection passed in
        """
        if det_boxes8d is None or len(det_boxes8d) == 0:
            self.tracks = []
            return np.array([], dtype=np.int32)

        det_boxes8d = np.asarray(det_boxes8d, dtype=np.float32)
        det_scores = np.asarray(det_scores, dtype=np.float32)

        if len(self.tracks) == 0:
            track_ids = []
            for i in range(len(det_boxes8d)):
                tid = self.next_id
                self.next_id += 1
                self.tracks.append({
                    "id": tid,
                    "box": det_boxes8d[i].copy(),
                    "score": float(det_scores[i]),
                })
                track_ids.append(tid)
            return np.array(track_ids, dtype=np.int32)

        track_boxes = np.stack([t["box"] for t in self.tracks], axis=0)
        predicted_boxes = predict_boxes_next_frame(track_boxes, self.dt)
        iou = standup_iou_matrix(predicted_boxes, det_boxes8d, post_processor)
        matches, _, unmatched_dets = greedy_iou_associate(iou, self.iou_thresh)

        track_ids = [-1] * len(det_boxes8d)
        matched_track_idx = set()
        matched_det_idx = set()

        for track_idx, det_idx in matches:
            tid = self.tracks[track_idx]["id"]
            track_ids[det_idx] = tid
            self.tracks[track_idx]["box"] = det_boxes8d[det_idx].copy()
            self.tracks[track_idx]["score"] = float(det_scores[det_idx])
            matched_track_idx.add(track_idx)
            matched_det_idx.add(det_idx)

        new_tracks = [
            self.tracks[i] for i in range(len(self.tracks)) if i in matched_track_idx
        ]

        for det_idx in unmatched_dets:
            tid = self.next_id
            self.next_id += 1
            new_tracks.append({
                "id": tid,
                "box": det_boxes8d[det_idx].copy(),
                "score": float(det_scores[det_idx]),
            })
            track_ids[det_idx] = tid

        self.tracks = new_tracks
        return np.array(track_ids, dtype=np.int32)
