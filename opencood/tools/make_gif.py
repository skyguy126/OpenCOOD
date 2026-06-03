import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import cv2

sys.path.insert(0, '/media/Disk2/OpenCOOD_vamsi_2')

# Patch 1 : ajouter la 5ème feature (frame_id)
import opencood.data_utils.datasets.early_fusion_dataset as efd
_orig = efd.EarlyFusionDataset.get_item_single_car

def _patched(self, selected_cav_base, ego_pose):
    result = _orig(self, selected_cav_base, ego_pose)
    lidar = result['projected_lidar']
    prev = result['projected_lidar_prev']
    result['projected_lidar'] = np.hstack([lidar, np.ones((len(lidar), 1), dtype=np.float32)])
    result['projected_lidar_prev'] = np.hstack([prev, np.zeros((len(prev), 1), dtype=np.float32)])
    return result

efd.EarlyFusionDataset.get_item_single_car = _patched

# Patch 2 : visualisation BEV headless
import opencood.data_utils.datasets.basedataset as baseds
from opencood.utils import common_utils

def _headless_visualize_result(self, pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
    if not save_path:
        return

    if pcd is not None and not isinstance(pcd, np.ndarray):
        pcd = common_utils.torch_tensor_to_numpy(pcd)
    if pred_box_tensor is not None and not isinstance(pred_box_tensor, np.ndarray):
        pred_box_tensor = common_utils.torch_tensor_to_numpy(pred_box_tensor)
    if gt_tensor is not None and not isinstance(gt_tensor, np.ndarray):
        gt_tensor = common_utils.torch_tensor_to_numpy(gt_tensor)

    # Enlever la dim batch [1, N, 5] -> [N, 5]
    if pcd is not None and pcd.ndim == 3:
        pcd = pcd[0]

    L1, W1, H1, L2, W2, H2 = self.params["preprocess"]["cav_lidar_range"]
    res = 0.1

    img_w = int((L2 - L1) / res)
    img_h = int((W2 - W1) / res)
    bev = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    if pcd is not None and len(pcd) > 0:
        xs = np.clip(((pcd[:, 0] - L1) / res).astype(int), 0, img_w - 1)
        ys = np.clip(((W2 - pcd[:, 1]) / res).astype(int), 0, img_h - 1)
        bev[ys, xs] = [120, 120, 120]

    if gt_tensor is not None and len(gt_tensor) > 0:
        for box in gt_tensor:
            pts = box[:4, :2]
            px = np.clip(((pts[:, 0] - L1) / res).astype(int), 0, img_w - 1)
            py = np.clip(((W2 - pts[:, 1]) / res).astype(int), 0, img_h - 1)
            cv2.polylines(bev, [np.stack([px, py], axis=1).reshape(-1, 1, 2)], True, (0, 255, 0), 2)

    if pred_box_tensor is not None and len(pred_box_tensor) > 0:
        for box in pred_box_tensor:
            pts = box[:4, :2]
            px = np.clip(((pts[:, 0] - L1) / res).astype(int), 0, img_w - 1)
            py = np.clip(((W2 - pts[:, 1]) / res).astype(int), 0, img_h - 1)
            cv2.polylines(bev, [np.stack([px, py], axis=1).reshape(-1, 1, 2)], True, (0, 0, 255), 2)

    out_h = int(img_h * 800 / img_w)
    cv2.imwrite(save_path, cv2.resize(bev, (800, out_h)))

baseds.BaseDataset.visualize_result = _headless_visualize_result

sys.argv = [
    'inference.py',
    '--model_dir', '/media/Disk2/OpenCOOD_vamsi_2/opencood/logs/point_pillar_early_fusion_2026_05_22_11_30_26',
    '--fusion_method', 'early',
    '--save_vis',
]

exec(open('/media/Disk2/OpenCOOD_vamsi_2/opencood/tools/inference.py').read())