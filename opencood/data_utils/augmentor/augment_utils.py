# -*- coding: utf-8 -*-
# Author: OpenPCDet

import numpy as np

from opencood.utils import common_utils


def random_flip_along_x(gt_boxes, points, prev_points=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
        prev_points: optional second point cloud to apply the same flip.
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

        if prev_points is not None:
            prev_points[:, 1] = -prev_points[:, 1]

    if prev_points is not None:
        return gt_boxes, points, prev_points
    return gt_boxes, points


def random_flip_along_y(gt_boxes, points, prev_points=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
        prev_points: optional second point cloud to apply the same flip.
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

        if prev_points is not None:
            prev_points[:, 0] = -prev_points[:, 0]

    if prev_points is not None:
        return gt_boxes, points, prev_points
    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range, prev_points=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
        prev_points: optional second point cloud to apply the same rotation.
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0],
                                       rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :],
                                                np.array([noise_rotation]))[0]

    gt_boxes[:, 0:3] = \
        common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3],
                                           np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation

    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[
            np.newaxis, :, :],
            np.array([noise_rotation]))[0][:, 0:2]

    if prev_points is not None:
        prev_points = common_utils.rotate_points_along_z(
            prev_points[np.newaxis, :, :],
            np.array([noise_rotation]))[0]

    if prev_points is not None:
        return gt_boxes, points, prev_points
    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range, prev_points=None):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
        prev_points: optional second point cloud to apply the same scaling.
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        if prev_points is not None:
            return gt_boxes, points, prev_points
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale

    if prev_points is not None:
        prev_points[:, :3] *= noise_scale

    if prev_points is not None:
        return gt_boxes, points, prev_points
    return gt_boxes, points
