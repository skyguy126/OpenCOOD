# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for early fusion
"""
import random
import math
from collections import OrderedDict

import numpy as np
import torch

import opencood.data_utils.datasets
from opencood.utils import box_utils
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2

def get_speed_by_object_id(vehicles, obj_id): #TODO jk
    if obj_id in vehicles:
        return float(vehicles[obj_id].get('speed', 0.0))

    obj_id_str = str(obj_id)
    if obj_id_str in vehicles:
        return float(vehicles[obj_id_str].get('speed', 0.0))

    try:
        obj_id_int = int(obj_id)
        if obj_id_int in vehicles:
            return float(vehicles[obj_id_int].get('speed', 0.0))
    except Exception:
        pass

    return 0.0 


class EarlyFusionDataset(basedataset.BaseDataset):
    """
    This dataset is used for early fusion, where each CAV transmit the raw
    point cloud to the ego vehicle.
    """
    def __init__(self, params, visualize, train=True):
        super(EarlyFusionDataset, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = build_postprocessor(params['postprocess'], train)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        projected_lidar_stack = []
        object_stack = []
        object_id_stack = []
        # object_stack_8d_debug = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)
            # all these lidar and object coordinates are projected to ego
            # already.
            projected_lidar_stack.append(
                selected_cav_processed['projected_lidar'])
            # object_stack.append(selected_cav_processed['object_bbx_center'])
            object_stack.append(selected_cav_processed['object_bbx_center'])
            # object_stack_8d_debug.append(selected_cav_processed['object_bbx_center_8d_debug'])
            object_id_stack += selected_cav_processed['object_ids']

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]
        # object_stack_8d_debug = np.vstack(object_stack_8d_debug)
        # object_stack_8d_debug = object_stack_8d_debug[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 8)) #TODO jk
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # convert list to numpy array, (N, 4)
        projected_lidar_stack = np.vstack(projected_lidar_stack)

        # data augmentation
        # projected_lidar_stack, object_bbx_center, mask = \
        #     self.augment(projected_lidar_stack, object_bbx_center, mask)
        # Data augmentation only supports 7D boxes:
        # [x, y, z, h, w, l, yaw]
        # Our new 8D box is:
        # [x, y, z, h, w, l, yaw, speed]
        if object_bbx_center.shape[1] == 8:
            speed_col = object_bbx_center[:, 7:8].copy()
            object_bbx_center_7d = object_bbx_center[:, :7].copy()

            projected_lidar_stack, object_bbx_center_7d, mask = \
                self.augment(projected_lidar_stack, object_bbx_center_7d, mask)

            object_bbx_center = np.concatenate(
                [object_bbx_center_7d, speed_col],
                axis=1
            )
        else:
            projected_lidar_stack, object_bbx_center, mask = \
                self.augment(projected_lidar_stack, object_bbx_center, mask)

        # we do lidar filtering in the stacked lidar
        projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                     self.params['preprocess'][
                                                         'cav_lidar_range'])
        # augmentation may remove some of the bbx out of range
        object_bbx_center_valid = object_bbx_center[mask == 1]
        # object_bbx_center_valid_7d = object_bbx_center_valid[:, :7]

        # object_bbx_center_valid, range_mask = \
        #     box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
        #                                              self.params['preprocess'][
        #                                                  'cav_lidar_range'],
        #                                              self.params['postprocess'][
        #                                                  'order'],
        #                                              return_mask=True
        #                                              )
        # object_bbx_center_valid = object_bbx_center_valid[range_mask]
        # object_bbx_center_valid[:, :7] = object_bbx_center_valid_7d

        object_bbx_center_valid = object_bbx_center[mask == 1]

        object_bbx_center_valid_7d = object_bbx_center_valid[:, :7]

        object_bbx_center_valid_7d, range_mask = \
            box_utils.mask_boxes_outside_range_numpy(
                object_bbx_center_valid_7d,
                self.params['preprocess']['cav_lidar_range'],
                self.params['postprocess']['order'],
                return_mask=True
            )

        # Apply range mask to full 8D boxes, preserving speed.
        object_bbx_center_valid = object_bbx_center_valid[range_mask]

        # Replace filtered geometry with filtered 7D geometry.
        object_bbx_center_valid[:, :7] = object_bbx_center_valid_7d

        mask[object_bbx_center_valid.shape[0]:] = 0
        object_bbx_center[:object_bbx_center_valid.shape[0]] = \
            object_bbx_center_valid
        object_bbx_center[object_bbx_center_valid.shape[0]:] = 0
        unique_indices = list(np.array(unique_indices)[range_mask])
        # object_stack_8d_debug = object_stack_8d_debug[range_mask]

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(projected_lidar_stack)

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
            # 'object_bbx_center_8d_debug': object_stack_8d_debug,
            'object_bbx_mask': mask,
            'object_ids': [object_id_stack[i] for i in unique_indices],
            'anchor_box': anchor_box,
            'processed_lidar': lidar_dict,
            'label_dict': label_dict})

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                                                   projected_lidar_stack})

        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = \
            x1_to_x2(selected_cav_base['params']['lidar_pose'],
                     ego_pose)

        # retrieve objects under ego coordinates
        # object_bbx_center, object_bbx_mask, object_ids = \
        #     self.post_processor.generate_object_center([selected_cav_base],
        #                                                ego_pose)

        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base], ego_pose)

        valid_object_bbx_center = object_bbx_center[object_bbx_mask == 1]

        vehicles = selected_cav_base['params']['vehicles']

        speed_list = []
        for obj_id in object_ids:
            #speed = self.get_speed_by_object_id(vehicles, obj_id)
            speed = get_speed_by_object_id(vehicles, obj_id) #TODO
            speed_list.append(speed)

        speed_array = np.array(speed_list, dtype=np.float32).reshape(-1, 1)

        # shape: (num_objects, 8)
        valid_object_bbx_center = np.concatenate(
            [valid_object_bbx_center, speed_array],
            axis=1
        )
        # filter lidar
        # lidar_np = selected_cav_base['lidar_np']
        # lidar_np = shuffle_points(lidar_np)
        # # remove points that hit itself
        # lidar_np = mask_ego_points(lidar_np)
        # # project the lidar to ego space
        # lidar_np[:, :3] = \
        #     box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
        #                                              transformation_matrix)

        # -----------------------------
        # Current frame LiDAR
        # -----------------------------
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np)

        lidar_np[:, :3] = \
            box_utils.project_points_by_matrix_torch(
                lidar_np[:, :3],
                transformation_matrix
            )

        # Add time-lag feature: current frame = 0.0
        cur_time_lag = np.zeros((lidar_np.shape[0], 1), dtype=np.float32)
        lidar_np = np.concatenate([lidar_np, cur_time_lag], axis=1)

        # -----------------------------
        # Previous frame LiDAR
        # -----------------------------
        prev_lidar_np = selected_cav_base['prev_lidar_np']
        prev_lidar_np = shuffle_points(prev_lidar_np)
        prev_lidar_np = mask_ego_points(prev_lidar_np)

        prev_transformation_matrix = \
            selected_cav_base['prev_params']['transformation_matrix']

        prev_lidar_np[:, :3] = \
            box_utils.project_points_by_matrix_torch(
                prev_lidar_np[:, :3],
                prev_transformation_matrix
            )

        # Add time-lag feature: previous frame = -0.1
        prev_time_lag = -0.1 * np.ones((prev_lidar_np.shape[0], 1), dtype=np.float32)
        prev_lidar_np = np.concatenate([prev_lidar_np, prev_time_lag], axis=1)

        # -----------------------------
        # Stack previous + current
        # -----------------------------
        lidar_np = np.vstack([prev_lidar_np, lidar_np]).astype(np.float32)

        # selected_cav_processed.update(
        #     {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
        #      'object_ids': object_ids,
        #      'projected_lidar': lidar_np})
        selected_cav_processed.update({
            'object_bbx_center': valid_object_bbx_center,
            'object_ids': object_ids,
            'projected_lidar': lidar_np
        })
        # if selected_cav_base['ego']:
        #     print("DEBUG cur timestamp:", selected_cav_base.get('cur_timestamp', None))
        #     print("DEBUG prev timestamp:", selected_cav_base.get('prev_timestamp', None))
        #     print("DEBUG temporal lidar shape:", lidar_np.shape)
        #     print("DEBUG time_lag min/max:", lidar_np[:, 4].min(), lidar_np[:, 4].max())
        #     print("DEBUG 8D box shape:", valid_object_bbx_center.shape)
        #     if valid_object_bbx_center.shape[0] > 0:
        #         print("DEBUG first 8D box:", valid_object_bbx_center[0])

        return selected_cav_processed

    def collate_batch_test(self, batch):
        """
        Customized collate function for pytorch dataloader during testing
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # currently, we only support batch size of 1 during testing
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]

        output_dict = {}

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            # shape: (1, max_num, 7)
            object_bbx_center = \
                torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = \
                torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box':
                    torch.from_numpy(np.array(
                        cav_content[
                            'anchor_box']))})
            if self.visualize:
                origin_lidar = [cav_content['origin_lidar']]

            # processed lidar dictionary
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(
                    [cav_content['processed_lidar']])
            # label dictionary
            label_torch_dict = \
                self.post_processor.collate_batch([cav_content['label_dict']])

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()

            output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,
                                        'processed_lidar': processed_lidar_torch_dict,
                                        'label_dict': label_torch_dict,
                                        'object_ids': object_ids,
                                        'transformation_matrix': transformation_matrix_torch})

            if self.visualize:
                origin_lidar = \
                    np.array(
                        downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({'origin_lidar': origin_lidar})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor
