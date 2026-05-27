import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils


def decode_pred_boxes_8d(post_processor, ego_data, output_dict, score_threshold=None):
    """
    Decode early-fusion output to 8D boxes before NMS.

    Returns:
        boxes8d: torch.Tensor, shape (N, 8)
        scores: torch.Tensor, shape (N,)
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
    Apply rotated NMS using 7D geometry while preserving speed column.
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


def compute_iou_matrix_2d(pred_corners, gt_corners):
    """
    Compute standup 2D IoU matrix using BEV standup boxes.

    This is not exact rotated IoU, but it is much better than nearest-center
    matching and is simple/stable.
    """
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


def greedy_iou_match(pred_boxes8d, gt_boxes8d, post_processor, iou_thresh=0.5):
    """
    Match predictions to GT by IoU, then compare speed.

    Returns:
        list of speed absolute errors
    """
    if pred_boxes8d is None or gt_boxes8d is None:
        return []

    if len(pred_boxes8d) == 0 or len(gt_boxes8d) == 0:
        return []

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

    iou = compute_iou_matrix_2d(pred_corners, gt_corners)

    matched_errors = []
    used_pred = set()
    used_gt = set()

    candidates = []
    for p in range(iou.shape[0]):
        for g in range(iou.shape[1]):
            if iou[p, g] >= iou_thresh:
                candidates.append((iou[p, g], p, g))

    candidates.sort(reverse=True, key=lambda x: x[0])

    for _, p, g in candidates:
        if p in used_pred or g in used_gt:
            continue

        pred_speed = float(pred_boxes8d[p, 7].detach().cpu())
        gt_speed = float(gt_boxes8d[g, 7].detach().cpu())

        matched_errors.append(abs(pred_speed - gt_speed))

        used_pred.add(p)
        used_gt.add(g)

    return matched_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--hypes_yaml", default=None)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--score_thresh", type=float, default=None)
    parser.add_argument("--iou_thresh", type=float, default=0.3)
    args = parser.parse_args()

    hypes_yaml = args.hypes_yaml
    if hypes_yaml is None:
        hypes_yaml = os.path.join(args.model_dir, "config.yaml")

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

    all_errors = []
    total_gt = 0
    total_pred_after_nms = 0
    frames_with_matches = 0

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

            if pred_boxes8d is None:
                continue

            total_pred_after_nms += int(pred_boxes8d.shape[0])

            errors = greedy_iou_match(
                pred_boxes8d,
                gt_boxes8d,
                post_processor,
                iou_thresh=args.iou_thresh
            )

            if len(errors) > 0:
                frames_with_matches += 1
                all_errors.extend(errors)

    print("\n========== IOU-BASED SPEED EVALUATION ==========")
    print("Total GT boxes:", total_gt)
    print("Total predicted boxes after NMS:", total_pred_after_nms)
    print("Frames with matched pred-GT:", frames_with_matches)
    print("Total matched pairs:", len(all_errors))
    print("IoU threshold:", args.iou_thresh)

    if len(all_errors) == 0:
        print("No matches found.")
        print("Try --iou_thresh 0.3 or --score_thresh 0.05 for debugging.")
        return

    errors = np.array(all_errors)

    print("Speed MAE:", float(np.mean(errors)))
    print("Speed RMSE:", float(np.sqrt(np.mean(errors ** 2))))
    print("Speed median AE:", float(np.median(errors)))
    print("Speed max AE:", float(np.max(errors)))
    print("Speed std AE:", float(np.std(errors)))


if __name__ == "__main__":
    main()