"""
Multi-frame bbox tracking: CV Kalman filter + Hungarian (IoU + velocity cost).
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from opencood.utils import box_utils

KMH_TO_MPS = 1.0 / 3.6
DEFAULT_DT = 0.1  # OPV2V is 10 Hz
MAX_COST = 1e6


def compute_dt(prev_timestamp, cur_timestamp, default_dt=DEFAULT_DT):
    try:
        return (int(cur_timestamp) - int(prev_timestamp)) / 10.0
    except (ValueError, TypeError):
        return default_dt


def velocity_from_box(box8d):
    """Return (vx, vy) in m/s from 8D box speed (km/h) and yaw."""
    speed_mps = float(box8d[7]) * KMH_TO_MPS
    yaw = float(box8d[6])
    return speed_mps * np.cos(yaw), speed_mps * np.sin(yaw)


def predict_boxes_next_frame(boxes8d, dt=DEFAULT_DT):
    """Constant-velocity propagation using predicted speed and yaw."""
    if boxes8d is None or len(boxes8d) == 0:
        return boxes8d

    predicted = np.array(boxes8d, dtype=np.float32, copy=True)
    for i in range(len(predicted)):
        vx, vy = velocity_from_box(predicted[i])
        predicted[i, 0] += vx * dt
        predicted[i, 1] += vy * dt
    return predicted


def standup_iou_matrix(boxes_a8d, boxes_b8d, post_processor):
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


class CVKalmanFilter:
    """Constant-velocity Kalman filter on BEV center [x, y, vx, vy]."""

    def __init__(self, x, y, vx, vy, dt=DEFAULT_DT,
                 pos_var=1.0, vel_var=4.0, meas_var=0.5, proc_var=1.0):
        self.dt = dt
        self.x = np.array([x, y, vx, vy], dtype=np.float64)
        self.P = np.diag([pos_var, pos_var, vel_var, vel_var]).astype(np.float64)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        q = proc_var
        self.Q = np.diag([q, q, q * 0.5, q * 0.5]).astype(np.float64)
        self.R = np.eye(2, dtype=np.float64) * meas_var

    def set_dt(self, dt):
        self.dt = dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        z = np.asarray(z, dtype=np.float64).reshape(2)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P


class _Track:
    """Single track with Kalman state and last 8D box."""

    def __init__(self, global_id, box8d, score, dt):
        vx, vy = velocity_from_box(box8d)
        self.global_id = global_id
        self.kf = CVKalmanFilter(box8d[0], box8d[1], vx, vy, dt=dt)
        self.box8d = np.asarray(box8d, dtype=np.float32).copy()
        self.score = float(score)
        self.time_since_update = 0
        self.hits = 1

    def set_dt(self, dt):
        self.kf.set_dt(dt)

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1

    def update(self, box8d, score):
        self.kf.update([box8d[0], box8d[1]])
        self.box8d = np.asarray(box8d, dtype=np.float32).copy()
        self.score = float(score)
        self.time_since_update = 0
        self.hits += 1
        # Sync speed/yaw in box with filtered velocity for next prediction.
        speed_mps = np.sqrt(self.kf.x[2] ** 2 + self.kf.x[3] ** 2)
        self.box8d[7] = speed_mps / KMH_TO_MPS
        self.box8d[6] = np.arctan2(self.kf.x[3], self.kf.x[2])

    def predicted_box8d(self):
        box = self.box8d.copy()
        box[0] = self.kf.x[0]
        box[1] = self.kf.x[1]
        speed_mps = np.sqrt(self.kf.x[2] ** 2 + self.kf.x[3] ** 2)
        box[7] = speed_mps / KMH_TO_MPS
        box[6] = np.arctan2(self.kf.x[3], self.kf.x[2])
        return box


class KalmanHungarianTracker:
    """
    Global-ID tracker using CV Kalman prediction and Hungarian assignment.

    Association cost = (1 - IoU) + vel_weight * ||v_track - v_det|| / max_speed
    """

    def __init__(self, iou_thresh=0.3, dt=DEFAULT_DT, max_age=3,
                 vel_weight=0.3, max_speed_mps=30.0):
        self.iou_thresh = iou_thresh
        self.dt = dt
        self.max_age = max_age
        self.vel_weight = vel_weight
        self.max_speed_mps = max_speed_mps
        self.next_id = 1
        self.tracks = []

    def reset(self):
        self.next_id = 1
        self.tracks = []

    def set_dt(self, dt):
        self.dt = dt

    def _association_cost(self, pred_boxes, det_boxes, post_processor):
        n_t = len(pred_boxes)
        n_d = len(det_boxes)
        cost = np.full((n_t, n_d), MAX_COST, dtype=np.float64)
        if n_t == 0 or n_d == 0:
            return cost

        iou = standup_iou_matrix(pred_boxes, det_boxes, post_processor)
        for i in range(n_t):
            tvx, tvy = self.tracks[i].kf.x[2], self.tracks[i].kf.x[3]
            for j in range(n_d):
                if iou[i, j] < self.iou_thresh:
                    continue
                dvx, dvy = velocity_from_box(det_boxes[j])
                vel_diff = np.sqrt((tvx - dvx) ** 2 + (tvy - dvy) ** 2)
                vel_cost = vel_diff / self.max_speed_mps
                cost[i, j] = (1.0 - iou[i, j]) + self.vel_weight * vel_cost
        return cost

    def _hungarian_match(self, cost):
        if cost.size == 0:
            return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        unmatched_tracks = set(range(cost.shape[0]))
        unmatched_dets = set(range(cost.shape[1]))

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= MAX_COST:
                continue
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)

    def update(self, det_boxes8d, det_scores, post_processor):
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
                self.tracks.append(_Track(tid, det_boxes8d[i], det_scores[i], self.dt))
                track_ids.append(tid)
            return np.array(track_ids, dtype=np.int32)

        for trk in self.tracks:
            trk.set_dt(self.dt)
            trk.predict()

        pred_boxes = np.stack([t.predicted_box8d() for t in self.tracks], axis=0)
        cost = self._association_cost(pred_boxes, det_boxes8d, post_processor)
        matches, unmatched_tracks, unmatched_dets = self._hungarian_match(cost)

        track_ids = [-1] * len(det_boxes8d)
        matched_track_idx = set()

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(det_boxes8d[det_idx], det_scores[det_idx])
            track_ids[det_idx] = self.tracks[track_idx].global_id
            matched_track_idx.add(track_idx)

        # Keep matched tracks and unmatched tracks within max_age (SORT-style).
        self.tracks = [
            self.tracks[i] for i in range(len(self.tracks))
            if i in matched_track_idx or self.tracks[i].time_since_update <= self.max_age
        ]

        for det_idx in unmatched_dets:
            tid = self.next_id
            self.next_id += 1
            self.tracks.append(_Track(tid, det_boxes8d[det_idx], det_scores[det_idx], self.dt))
            track_ids[det_idx] = tid

        return np.array(track_ids, dtype=np.int32)


# Default tracker used by eval / inference pipelines.
SimpleTracker = KalmanHungarianTracker
