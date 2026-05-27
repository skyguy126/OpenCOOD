import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset


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


def nearest_gt_speed_match(pred_boxes_8d, gt_boxes_8d, max_dist=3.0):
    """
    Match each predicted box to the nearest GT center and compare speed.

    This is a rough sanity metric, not official detection-matched evaluation.
    """
    errors = []

    if pred_boxes_8d is None or gt_boxes_8d is None:
        return errors

    if len(pred_boxes_8d) == 0 or len(gt_boxes_8d) == 0:
        return errors

    pred_centers = pred_boxes_8d[:, :2]
    gt_centers = gt_boxes_8d[:, :2]

    used_gt = set()

    for i in range(len(pred_boxes_8d)):
        dists = np.linalg.norm(gt_centers - pred_centers[i], axis=1)
        nearest = int(np.argmin(dists))

        if dists[nearest] <= max_dist and nearest not in used_gt:
            pred_speed = float(pred_boxes_8d[i, 7])
            gt_speed = float(gt_boxes_8d[nearest, 7])
            errors.append(abs(pred_speed - gt_speed))
            used_gt.add(nearest)

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True,
                        help="Path to OpenCOOD log folder containing checkpoint and config.yaml")
    parser.add_argument("--hypes_yaml", default=None,
                        help="Optional yaml path. If omitted, uses config.yaml in model_dir.")
    parser.add_argument("--max_batches", type=int, default=-1,
                        help="Limit number of validation batches. Use -1 for all.")
    parser.add_argument("--max_match_dist", type=float, default=3.0,
                        help="Max XY center distance for pred-GT speed matching.")
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
    num_frames_with_matches = 0
    total_preds = 0
    total_gt = 0

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(data_loader)):
            if args.max_batches > 0 and i >= args.max_batches:
                break

            if batch_data is None:
                continue

            batch_data = train_utils.to_device(batch_data, torch.device("cuda"))

            output_dict = model(batch_data["ego"])

            pred_boxes_8d, pred_scores = extract_pred_boxes_8d(
                post_processor,
                batch_data["ego"],
                output_dict
            )

            gt_boxes = batch_data["ego"]["object_bbx_center"]
            gt_mask = batch_data["ego"]["object_bbx_mask"]

            gt_boxes = gt_boxes[0].detach().cpu().numpy()
            gt_mask = gt_mask[0].detach().cpu().numpy().astype(bool)
            gt_boxes_valid = gt_boxes[gt_mask]

            total_gt += len(gt_boxes_valid)

            if pred_boxes_8d is None:
                continue

            pred_boxes_8d_np = pred_boxes_8d.numpy()
            total_preds += len(pred_boxes_8d_np)

            errors = nearest_gt_speed_match(
                pred_boxes_8d_np,
                gt_boxes_valid,
                max_dist=args.max_match_dist
            )

            if len(errors) > 0:
                num_frames_with_matches += 1
                all_abs_errors.extend(errors)

    print("\n========== SPEED EVALUATION ==========")
    print("Total GT boxes:", total_gt)
    print("Total predicted boxes before NMS:", total_preds)
    print("Frames with at least one matched pred-GT pair:", num_frames_with_matches)
    print("Total matched pairs:", len(all_abs_errors))

    if len(all_abs_errors) == 0:
        print("No speed matches found.")
        print("Try increasing --max_match_dist or lowering score_threshold in YAML.")
        return

    errors = np.array(all_abs_errors)

    print("Speed MAE:", float(np.mean(errors)))
    print("Speed RMSE:", float(np.sqrt(np.mean(errors ** 2))))
    print("Speed median AE:", float(np.median(errors)))
    print("Speed max AE:", float(np.max(errors)))
    print("Speed std AE:", float(np.std(errors)))


if __name__ == "__main__":
    main()