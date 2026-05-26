# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
3D Anchor Generator for Voxel
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from opencood.data_utils.post_processor.base_postprocessor \
    import BasePostprocessor
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils


class VoxelPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, train):
        super(VoxelPostprocessor, self).__init__(anchor_params, train)
        self.anchor_num = self.params['anchor_args']['num']

    def generate_anchor_box(self):
        W = self.params['anchor_args']['W']
        H = self.params['anchor_args']['H']

        l = self.params['anchor_args']['l']
        w = self.params['anchor_args']['w']
        h = self.params['anchor_args']['h']
        r = self.params['anchor_args']['r']

        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]

        vh = self.params['anchor_args']['vh']
        vw = self.params['anchor_args']['vw']

        xrange = [self.params['anchor_args']['cav_lidar_range'][0],
                  self.params['anchor_args']['cav_lidar_range'][3]]
        yrange = [self.params['anchor_args']['cav_lidar_range'][1],
                  self.params['anchor_args']['cav_lidar_range'][4]]

        if 'feature_stride' in self.params['anchor_args']:
            feature_stride = self.params['anchor_args']['feature_stride']
        else:
            feature_stride = 2

        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride)
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)

        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num)
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0

        w = np.ones_like(cx) * w
        l = np.ones_like(cx) * l
        h = np.ones_like(cx) * h

        r_ = np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]

        if self.params['order'] == 'hwl':
            anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1)
        elif self.params['order'] == 'lhw':
            anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
        else:
            sys.exit('Unknown bbx order.')

        return anchors

    def generate_label(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        gt_box_center : np.ndarray
            Shape: (max_num, 8)
            Format: [x, y, z, h, w, l, yaw, speed]

        anchors : np.ndarray
            Shape: (H, W, anchor_num, 7)

        mask : np.ndarray
            Shape: (max_num)

        Returns
        -------
        label_dict : dict
            Dictionary that contains target tensors.
        """
        assert self.params['order'] == 'hwl', \
            'Currently Voxel only supports hwl bbx order.'

        # (max_num, 8)
        gt_box_center = kwargs['gt_box_center']

        # (H, W, anchor_num, 7)
        anchors = kwargs['anchors']

        # (max_num)
        masks = kwargs['mask']

        # Use 8D box target by default.
        # 7D = [x, y, z, h, w, l, yaw]
        # 8D = [x, y, z, h, w, l, yaw, speed]
        box_code_size = self.params.get('box_code_size', 8)

        # Speed normalization. Add this in yaml:
        # postprocess:
        #   box_code_size: 8
        #   target_args:
        #     speed_norm: 30.0
        speed_norm = self.params['target_args'].get('speed_norm', 30.0)

        # (H, W)
        feature_map_shape = anchors.shape[:2]

        # (H*W*anchor_num, 7)
        anchors = anchors.reshape(-1, 7)

        # Normalization factor, (H*W*anchor_num)
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

        # (H, W, anchor_num)
        pos_equal_one = np.zeros((*feature_map_shape, self.anchor_num))
        neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num))

        # (H, W, anchor_num * 8)
        targets = np.zeros((*feature_map_shape,
                        self.anchor_num * box_code_size))

        # Only valid GT boxes.
        # Shape: (n, 8)
        gt_box_center_valid = gt_box_center[masks == 1]

        # IMPORTANT:
        # Geometry utilities only understand 7D boxes.
        # Use [:, :7] for IoU/corner computation.
        gt_box_center_valid_7d = gt_box_center_valid[:, :7]

        # (n, 8, 3)
        gt_box_corner_valid = box_utils.boxes_to_corners_3d(
            gt_box_center_valid_7d,
            self.params['order']
        )

        # (H*W*anchor_num, 8, 3)
        anchors_corner = box_utils.boxes_to_corners_3d(
            anchors,
            order=self.params['order']
        )

        # (H*W*anchor_num, 4)
        anchors_standup_2d = box_utils.corner2d_to_standup_box(
            anchors_corner
        )

        # (n, 4)
        gt_standup_2d = box_utils.corner2d_to_standup_box(
            gt_box_corner_valid
        )

        # (H*W*anchor_num, n)
        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        # Anchors with largest IoU for each GT box.
        # Shape: (n)
        id_highest = np.argmax(iou.T, axis=1)

        # [0, 1, 2, ..., n-1]
        id_highest_gt = np.arange(iou.T.shape[0])

        # Make sure all highest IoUs are larger than 0.
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # Find anchors with IoU > positive threshold.
        id_pos, id_pos_gt = np.where(
            iou > self.params['target_args']['pos_threshold']
        )

        # Find anchors with IoU < negative threshold for all GT boxes.
        id_neg = np.where(
            np.sum(
                iou < self.params['target_args']['neg_threshold'],
                axis=1
            ) == iou.shape[1]
        )[0]

        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])

        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]

        id_neg.sort()

        # Set positive anchors.
        index_x, index_y, index_z = np.unravel_index(
            id_pos,
            (*feature_map_shape, self.anchor_num)
        )
        pos_equal_one[index_x, index_y, index_z] = 1

        # Base channel index for each selected anchor.
        base = np.array(index_z) * box_code_size

        # x
        targets[index_x, index_y, base + 0] = (
            gt_box_center_valid[id_pos_gt, 0] - anchors[id_pos, 0]
        ) / anchors_d[id_pos]

        # y
        targets[index_x, index_y, base + 1] = (
            gt_box_center_valid[id_pos_gt, 1] - anchors[id_pos, 1]
        ) / anchors_d[id_pos]

        # z
        targets[index_x, index_y, base + 2] = (
            gt_box_center_valid[id_pos_gt, 2] - anchors[id_pos, 2]
        ) / anchors[id_pos, 3]

        # h
        targets[index_x, index_y, base + 3] = np.log(
            gt_box_center_valid[id_pos_gt, 3] / anchors[id_pos, 3]
        )

        # w
        targets[index_x, index_y, base + 4] = np.log(
            gt_box_center_valid[id_pos_gt, 4] / anchors[id_pos, 4]
        )

        # l
        targets[index_x, index_y, base + 5] = np.log(
            gt_box_center_valid[id_pos_gt, 5] / anchors[id_pos, 5]
        )

        # yaw
        targets[index_x, index_y, base + 6] = (
            gt_box_center_valid[id_pos_gt, 6] - anchors[id_pos, 6]
        )

        # speed
        if box_code_size >= 8:
            targets[index_x, index_y, base + 7] = (
                gt_box_center_valid[id_pos_gt, 7] / speed_norm
            )

        # Set negative anchors.
        index_x, index_y, index_z = np.unravel_index(
            id_neg,
            (*feature_map_shape, self.anchor_num)
        )
        neg_equal_one[index_x, index_y, index_z] = 1

        # Avoid a box being positive and negative at the same time.
        index_x, index_y, index_z = np.unravel_index(
            id_highest,
            (*feature_map_shape, self.anchor_num)
        )
        neg_equal_one[index_x, index_y, index_z] = 0

        label_dict = {
            'pos_equal_one': pos_equal_one,
            'neg_equal_one': neg_equal_one,
            'targets': targets
        }

        return label_dict

    @staticmethod
    def collate_batch(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = \
            torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = \
            torch.from_numpy(np.array(neg_equal_one))
        targets = \
            torch.from_numpy(np.array(targets))

        return {'targets': targets,
                'pos_equal_one': pos_equal_one,
                'neg_equal_one': neg_equal_one}

    def post_process(self, data_dict, output_dict):
        """
        Process model outputs to predicted 3D bounding boxes.

        Modified for 8D regression:
            decoded box = [x, y, z, h, w, l, yaw, speed]

        Geometry/NMS/AP still use only:
            [x, y, z, h, w, l, yaw]
        """
        pred_box3d_list = []
        pred_box2d_list = []

        box_code_size = self.params.get('box_code_size', 8)

        for cav_id, cav_content in data_dict.items():
            assert cav_id in output_dict

            transformation_matrix = cav_content['transformation_matrix']

            # anchor_box shape: (H, W, anchor_num, 7)
            anchor_box = cav_content['anchor_box']

            # classification probability
            prob = output_dict[cav_id]['psm']
            prob = F.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)

            # regression map
            reg = output_dict[cav_id]['rm']

            # Decode regression map.
            # batch_box3d shape: (N, H*W*anchor_num, 8)
            batch_box3d = self.delta_to_boxes3d(reg, anchor_box)

            mask = torch.gt(
                prob,
                self.params['target_args']['score_threshold']
            )
            mask = mask.view(1, -1)

            # Repeat mask for 8D box regression, not 7D.
            mask_reg = mask.unsqueeze(2).repeat(1, 1, box_code_size)

            # during validation/testing, batch size should be 1
            assert batch_box3d.shape[0] == 1

            # boxes3d_all shape: (num_selected, 8)
            boxes3d_all = torch.masked_select(
                batch_box3d[0],
                mask_reg[0]
            ).view(-1, box_code_size)

            scores = torch.masked_select(prob[0], mask[0])

            if len(boxes3d_all) != 0:
                # Geometry utilities only support 7D boxes.
                boxes3d = boxes3d_all[:, :7]

                # Optional: keep speed if you want to inspect/save it later.
                if box_code_size >= 8:
                    pred_speed = boxes3d_all[:, 7]

                # Convert 7D box to 8 corners.
                boxes3d_corner = box_utils.boxes_to_corners_3d(
                    boxes3d,
                    order=self.params['order']
                )

                # Project boxes to ego frame.
                projected_boxes3d = box_utils.project_box3d(
                    boxes3d_corner,
                    transformation_matrix
                )

                # Convert 3D corners to 2D standup boxes.
                projected_boxes2d = box_utils.corner_to_standup_box_torch(
                    projected_boxes3d
                )

                boxes2d_score = torch.cat(
                    (projected_boxes2d, scores.unsqueeze(1)),
                    dim=1
                )

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)

        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            return None, None

        pred_box2d_list = torch.vstack(pred_box2d_list)

        # scores
        scores = pred_box2d_list[:, -1]

        # predicted 3D boxes as corners: (N, 8, 3)
        pred_box3d_tensor = torch.vstack(pred_box3d_list)

        # remove abnormal boxes
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]

        # NMS
        keep_index = box_utils.nms_rotated(
            pred_box3d_tensor,
            scores,
            self.params['nms_thresh']
        )

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]

        # filter predictions outside range
        mask = box_utils.get_mask_for_boxes_within_range_torch(
            pred_box3d_tensor
        )

        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        scores = scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        return pred_box3d_tensor, scores

    def delta_to_boxes3d(self, deltas, anchors, channel_swap=True):
        """
        Convert model regression output to 3D bounding boxes.

        Original:
            deltas: anchor_num * 7

        Modified:
            deltas: anchor_num * 8
            box = [x, y, z, h, w, l, yaw, speed]
        """
        box_code_size = self.params.get('box_code_size', 8)
        speed_norm = self.params['target_args'].get('speed_norm', 30.0)

        # batch size
        N = deltas.shape[0]

        if channel_swap:
            deltas = deltas.permute(0, 2, 3, 1).contiguous()
            deltas = deltas.view(N, -1, box_code_size)
        else:
            deltas = deltas.contiguous().view(N, -1, box_code_size)

        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # anchors are still 7D: [x, y, z, h, w, l, yaw]
        anchors_reshaped = anchors.view(-1, 7).float()

        # diagonal of anchor box on BEV plane
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2
        )

        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # x, y
        boxes3d[..., [0, 1]] = (
            deltas[..., [0, 1]] * anchors_d
            + anchors_reshaped[..., [0, 1]]
        )

        # z
        boxes3d[..., [2]] = (
            deltas[..., [2]] * anchors_reshaped[..., [3]]
            + anchors_reshaped[..., [2]]
        )

        # h, w, l
        boxes3d[..., [3, 4, 5]] = (
            torch.exp(deltas[..., [3, 4, 5]])
            * anchors_reshaped[..., [3, 4, 5]]
        )

        # yaw
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        # speed
        if box_code_size >= 8:
            boxes3d[..., 7] = deltas[..., 7] * speed_norm

        return boxes3d

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """
        vis_utils.visualize_single_sample_output_gt(pred_box_tensor,
                                                    gt_tensor,
                                                    pcd,
                                                    show_vis,
                                                    save_path)