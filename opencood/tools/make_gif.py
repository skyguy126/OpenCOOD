"""
Generate BEV visualization frames (and optional GIF) for a dual-frame model.

Uses separate current/previous lidar inputs (dual_frame) — no stacked 5th feature.
Predictions are smoothed with KalmanHungarianTracker before drawing.
"""
import argparse
import glob
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.eval_track import (
    apply_nms_to_8d_boxes,
    decode_pred_boxes_8d,
    get_sample_info,
)
from opencood.utils import box_utils, common_utils
from opencood.utils.track_utils import KalmanHungarianTracker, compute_dt


def boxes8d_to_corners(boxes8d, post_processor):
    if boxes8d is None or len(boxes8d) == 0:
        return None
    boxes7d = torch.from_numpy(np.asarray(boxes8d[:, :7], dtype=np.float32))
    corners = box_utils.boxes_to_corners_3d(
        boxes7d, order=post_processor.params['order'])
    return corners.detach().cpu().numpy()


def get_tracker_vis_boxes(tracker):
    """Return Kalman-smoothed 8D boxes for all active tracks."""
    if not tracker.tracks:
        return None
    boxes = []
    for trk in tracker.tracks:
        if trk.time_since_update == 0:
            boxes.append(trk.box8d)
        else:
            boxes.append(trk.predicted_box8d())
    return np.stack(boxes, axis=0)


def advance_tracker(tracker):
    """Predict all tracks forward one step (empty-detection frame)."""
    for trk in tracker.tracks:
        trk.set_dt(tracker.dt)
        trk.predict()


def render_bev_frame(pcd, pred_corners, gt_corners, cav_lidar_range,
                     res=0.1, out_width=800):
    L1, W1, _, L2, W2, _ = cav_lidar_range
    img_w = int((L2 - L1) / res)
    img_h = int((W2 - W1) / res)
    bev = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    if pcd is not None and len(pcd) > 0:
        xs = np.clip(((pcd[:, 0] - L1) / res).astype(int), 0, img_w - 1)
        ys = np.clip(((W2 - pcd[:, 1]) / res).astype(int), 0, img_h - 1)
        bev[ys, xs] = [120, 120, 120]

    if gt_corners is not None and len(gt_corners) > 0:
        for box in gt_corners:
            pts = box[:4, :2]
            px = np.clip(((pts[:, 0] - L1) / res).astype(int), 0, img_w - 1)
            py = np.clip(((W2 - pts[:, 1]) / res).astype(int), 0, img_h - 1)
            cv2.polylines(bev, [np.stack([px, py], axis=1).reshape(-1, 1, 2)],
                          True, (0, 255, 0), 2)

    if pred_corners is not None and len(pred_corners) > 0:
        for box in pred_corners:
            pts = box[:4, :2]
            px = np.clip(((pts[:, 0] - L1) / res).astype(int), 0, img_w - 1)
            py = np.clip(((W2 - pts[:, 1]) / res).astype(int), 0, img_h - 1)
            cv2.polylines(bev, [np.stack([px, py], axis=1).reshape(-1, 1, 2)],
                          True, (0, 0, 255), 2)

    out_h = int(img_h * out_width / img_w)
    return cv2.resize(bev, (out_width, out_h))


def to_numpy_pcd(pcd):
    if pcd is None:
        return None
    if not isinstance(pcd, np.ndarray):
        pcd = common_utils.torch_tensor_to_numpy(pcd)
    if pcd.ndim == 3:
        pcd = pcd[0]
    return pcd


def build_gif(frame_dir, gif_path, fps=10):
    pattern = os.path.join(frame_dir, '*.png')
    frames = sorted(glob.glob(pattern))
    if not frames:
        print(f'No PNG frames found in {frame_dir}; skipping GIF.')
        return

    images = [Image.open(f) for f in frames]
    duration_ms = int(1000 / fps)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f'GIF saved to {gif_path} ({len(frames)} frames @ {fps} fps)')


def parse_args():
    parser = argparse.ArgumentParser(description='BEV frames + GIF with tracking')
    parser.add_argument('--model_dir', type=str,
                        default='/home/project/baseline_multiframe')
    parser.add_argument('--hypes_yaml', default=None)
    parser.add_argument('--max_batches', type=int, default=-1)
    parser.add_argument('--score_thresh', type=float, default=None)
    parser.add_argument('--no_track', action='store_true',
                        help='Draw raw detections instead of tracked boxes')
    parser.add_argument('--iou_thresh', type=float, default=0.3)
    parser.add_argument('--max_age', type=int, default=3)
    parser.add_argument('--vel_weight', type=float, default=0.3)
    parser.add_argument('--fps', type=int, default=10,
                        help='GIF playback rate')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gif_name', type=str, default='vis.gif')
    return parser.parse_args()


def main():
    args = parse_args()

    hypes_yaml = args.hypes_yaml or os.path.join(args.model_dir, 'config.yaml')
    hypes = yaml_utils.load_yaml(hypes_yaml, None)

    print('Dataset Building')
    dataset = build_dataset(hypes, visualize=True, train=False)
    print(f'{len(dataset)} samples found.')

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    _, model = train_utils.load_saved_model(args.model_dir, model)

    post_processor = dataset.post_processor
    cav_lidar_range = hypes['preprocess']['cav_lidar_range']

    vis_dir = os.path.join(args.model_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    tracker = KalmanHungarianTracker(
        iou_thresh=args.iou_thresh,
        max_age=args.max_age,
        vel_weight=args.vel_weight,
    )

    prev_info = None
    prev_scenario = None

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(data_loader)):
            if args.max_batches > 0 and i >= args.max_batches:
                break
            if batch_data is None:
                continue

            info = get_sample_info(dataset, i)

            if prev_scenario is not None and info['scenario_index'] != prev_scenario:
                tracker.reset()
            prev_scenario = info['scenario_index']

            batch_data = train_utils.to_device(batch_data, device)
            output_dict = model(batch_data['ego'])

            pred_boxes8d, scores = decode_pred_boxes_8d(
                post_processor, batch_data['ego'], output_dict,
                score_threshold=args.score_thresh,
            )
            if pred_boxes8d is not None:
                pred_boxes8d, scores = apply_nms_to_8d_boxes(
                    post_processor, pred_boxes8d, scores)

            gt_box_tensor = post_processor.generate_gt_bbx(batch_data)
            gt_corners = None
            if gt_box_tensor is not None:
                gt_corners = common_utils.torch_tensor_to_numpy(gt_box_tensor)

            if args.no_track:
                vis_boxes8d = (
                    pred_boxes8d.detach().cpu().numpy()
                    if pred_boxes8d is not None else None)
            elif pred_boxes8d is not None:
                if (prev_info is not None
                        and info['scenario_index'] == prev_info['scenario_index']):
                    tracker.set_dt(
                        compute_dt(prev_info['timestamp'], info['timestamp']))
                tracker.update(
                    pred_boxes8d.detach().cpu().numpy(),
                    scores.detach().cpu().numpy(),
                    post_processor,
                )
                vis_boxes8d = get_tracker_vis_boxes(tracker)
            else:
                if tracker.tracks:
                    if (prev_info is not None
                            and info['scenario_index'] == prev_info['scenario_index']):
                        tracker.set_dt(
                            compute_dt(prev_info['timestamp'], info['timestamp']))
                    advance_tracker(tracker)
                    tracker.tracks = [
                        trk for trk in tracker.tracks
                        if trk.time_since_update <= tracker.max_age
                    ]
                vis_boxes8d = get_tracker_vis_boxes(tracker)

            pred_corners = boxes8d_to_corners(vis_boxes8d, post_processor)
            pcd = to_numpy_pcd(batch_data['ego']['origin_lidar'][0])

            frame = render_bev_frame(pcd, pred_corners, gt_corners, cav_lidar_range)
            frame_path = os.path.join(vis_dir, '%05d.png' % i)
            cv2.imwrite(frame_path, frame)

            prev_info = info

    gif_path = os.path.join(args.model_dir, args.gif_name)
    build_gif(vis_dir, gif_path, fps=args.fps)


if __name__ == '__main__':
    main()
