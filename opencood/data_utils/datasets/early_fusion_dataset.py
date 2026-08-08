# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for early fusion
"""
import math
from collections import OrderedDict

import numpy as np
import numpy.linalg as npl
import torch

import opencood.data_utils.datasets
from opencood.utils import box_utils
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2, x_to_world


def world_xy_to_ego(xy, ego_pose):
    """Transform a world-frame xy point into the ego lidar frame."""
    Tw = x_to_world(ego_pose)
    pt = npl.inv(Tw) @ np.array(
        [xy[0], xy[1], ego_pose[2], 1.0], dtype=np.float64)
    return np.asarray(pt[:2], dtype=np.float32)


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
        planning_args = params.get('model', {}).get('args', {}).get(
            'planning_head', {})
        self.planning_enabled = planning_args.get('enabled', False)
        self.num_future_waypoints = planning_args.get('num_waypoints', 10)
        self.input_frame = int(
            params.get('model', {}).get('args', {}).get(
                'history_frames',
                planning_args.get('input_frame', self.history_frames)))

    def _scenario_timestamp_info(self, idx, ego_id):
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break

        scenario_database = self.scenario_database[scenario_index]
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]

        ego_timestamps = [
            timestamp for timestamp, timestamp_content
            in scenario_database[ego_id].items()
            if isinstance(timestamp_content, OrderedDict)
        ]
        return scenario_database, timestamp_index, ego_timestamps

    def get_future_ego_waypoints(self, idx, ego_id, current_ego_pose,
                                 num_waypoints=10):
        """
        Future ego xy positions in the *current ego frame* (rotation-aware).
        Labels only — never fed to the planner as an input feature.

        Samples the next ``num_waypoints`` consecutive stored frames
        (OPV2V: 0.2s apart).
        """
        scenario_database, timestamp_index, ego_timestamps = \
            self._scenario_timestamp_info(idx, ego_id)
        last_timestamp_index = len(ego_timestamps) - 1

        future_waypoints = []
        for step in range(1, num_waypoints + 1):
            future_timestamp_index = min(timestamp_index + step,
                                         last_timestamp_index)
            future_timestamp = ego_timestamps[future_timestamp_index]
            future_yaml = scenario_database[ego_id][future_timestamp]['yaml']
            future_pose = load_yaml(future_yaml)['lidar_pose']
            # Transform future ego origin into the current ego frame.
            T = x1_to_x2(future_pose, current_ego_pose)
            future_waypoints.append([T[0, 3], T[1, 3]])

        return np.asarray(future_waypoints, dtype=np.float32)

    def get_planning_target(self, idx, ego_id, current_ego_pose):
        """
        V2Xverse-style navigation command target in the current ego frame.

        Uses OPV2V `plan_trajectory` (route hint available at the current
        timestamp). Falls back to a fixed forward command if missing.
        Never uses future GT waypoints.
        """
        scenario_database, timestamp_index, ego_timestamps = \
            self._scenario_timestamp_info(idx, ego_id)
        cur_yaml = scenario_database[ego_id][
            ego_timestamps[timestamp_index]]['yaml']
        cur_params = load_yaml(cur_yaml)

        plan_traj = cur_params.get('plan_trajectory', None)
        if plan_traj is not None and len(plan_traj) > 0:
            # Far command point, analogous to CARLA x_command/y_command.
            cmd_xy = plan_traj[-1][:2]
            target = world_xy_to_ego(cmd_xy, current_ego_pose)
        else:
            # History-only fallback: extrapolate along recent ego motion.
            hist_idx = max(0, timestamp_index - (self.input_frame - 1))
            past_yaml = scenario_database[ego_id][
                ego_timestamps[hist_idx]]['yaml']
            past_pose = load_yaml(past_yaml)['lidar_pose']
            past_in_ego = world_xy_to_ego(past_pose[:2], current_ego_pose)
            direction = -past_in_ego  # from past toward current origin
            norm = float(np.linalg.norm(direction))
            if norm < 1e-3:
                direction = np.array([1.0, 0.0], dtype=np.float32)
            else:
                direction = (direction / norm).astype(np.float32)
            target = direction * 20.0

        # Clip to a V2Xverse-like local command window in ego frame
        # (front-heavy; OpenCOOD x-forward / y-left).
        target = np.clip(target, a_min=[-12.0, -12.0],
                         a_max=[36.0, 12.0]).astype(np.float32)
        target[np.isnan(target)] = 0.0
        return target

    def get_history_occupancy_meta(self, idx, ego_id, current_ego_pose):
        """
        Per-history-frame actor/ego positions in the current ego frame for
        V2Xverse occupancy channels 0/1.
        """
        scenario_database, timestamp_index, ego_timestamps = \
            self._scenario_timestamp_info(idx, ego_id)
        max_num = self.params['postprocess']['max_num']
        t_len = self.input_frame

        history_actor_xy = np.zeros((t_len, max_num, 2), dtype=np.float32)
        history_actor_mask = np.zeros((t_len, max_num), dtype=np.float32)
        history_ego_xy = np.zeros((t_len, 2), dtype=np.float32)

        for t, hist_offset in enumerate(range(t_len - 1, -1, -1)):
            hist_index = max(0, timestamp_index - hist_offset)
            hist_key = ego_timestamps[hist_index]
            hist_yaml = scenario_database[ego_id][hist_key]['yaml']
            hist_params = load_yaml(hist_yaml)
            hist_pose = hist_params['lidar_pose']
            history_ego_xy[t] = world_xy_to_ego(hist_pose[:2], current_ego_pose)

            vehicles = hist_params.get('vehicles', {})
            count = 0
            for _, veh in vehicles.items():
                if count >= max_num:
                    break
                loc = veh.get('location', None)
                if loc is None:
                    continue
                history_actor_xy[t, count] = world_xy_to_ego(
                    loc[:2], current_ego_pose)
                history_actor_mask[t, count] = 1.0
                count += 1

        return history_actor_xy, history_actor_mask, history_ego_xy

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
        future_waypoints = self.get_future_ego_waypoints(
            idx,
            ego_id,
            ego_lidar_pose,
            self.num_future_waypoints)
        planning_target = self.get_planning_target(
            idx, ego_id, ego_lidar_pose)
        history_actor_xy, history_actor_mask, history_ego_xy = \
            self.get_history_occupancy_meta(idx, ego_id, ego_lidar_pose)

        projected_lidar_stack = []
        projected_lidar_prev_stack = []
        projected_lidar_history_stacks = [
            [] for _ in range(self.input_frame)]
        object_stack = []
        object_id_stack = []

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
            projected_lidar_stack.append(
                selected_cav_processed['projected_lidar'])
            projected_lidar_prev_stack.append(
                selected_cav_processed['projected_lidar_prev'])
            for t, hist_lidar in enumerate(
                    selected_cav_processed['projected_lidar_history']):
                projected_lidar_history_stacks[t].append(hist_lidar)
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 8)) #TODO jk
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # convert list to numpy array, (N, 4)
        projected_lidar_stack = np.vstack(projected_lidar_stack)
        projected_lidar_prev_stack = np.vstack(projected_lidar_prev_stack)
        projected_lidar_history = [
            np.vstack(stack) if len(stack) > 0 else
            np.zeros((1, 4), dtype=np.float32)
            for stack in projected_lidar_history_stacks
        ]

        # Data augmentation only supports 7D boxes:
        # [x, y, z, h, w, l, yaw]
        # Our new 8D box is:
        # [x, y, z, h, w, l, yaw, speed]
        if object_bbx_center.shape[1] == 8:
            speed_col = object_bbx_center[:, 7:8].copy()
            object_bbx_center_7d = object_bbx_center[:, :7].copy()

            projected_lidar_stack, projected_lidar_prev_stack, \
                object_bbx_center_7d, mask = \
                self.augment_dual(projected_lidar_stack,
                                  projected_lidar_prev_stack,
                                  object_bbx_center_7d, mask)

            object_bbx_center = np.concatenate(
                [object_bbx_center_7d, speed_col],
                axis=1
            )
        else:
            projected_lidar_stack, projected_lidar_prev_stack, \
                object_bbx_center, mask = \
                self.augment_dual(projected_lidar_stack,
                                  projected_lidar_prev_stack,
                                  object_bbx_center, mask)

        # Keep history lidar consistent with current/prev when no augment
        # is configured (planner baseline). If augment is enabled, history
        # clouds are left unaugmented — prefer empty data_augment for planning.
        if len(self.data_augmentor.data_augmentor_queue) == 0:
            projected_lidar_history[-1] = projected_lidar_stack
            if len(projected_lidar_history) >= 2:
                projected_lidar_history[-2] = projected_lidar_prev_stack

        cav_lidar_range = self.params['preprocess']['cav_lidar_range']
        projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                     cav_lidar_range)
        projected_lidar_prev_stack = mask_points_by_range(
            projected_lidar_prev_stack, cav_lidar_range)
        projected_lidar_history = [
            mask_points_by_range(pc, cav_lidar_range)
            for pc in projected_lidar_history
        ]

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

        # pre-process current, previous, and full history lidars
        lidar_dict = self.pre_processor.preprocess(projected_lidar_stack)
        lidar_prev_dict = self.pre_processor.preprocess(
            projected_lidar_prev_stack)
        lidar_history_dicts = [
            self.pre_processor.preprocess(pc)
            for pc in projected_lidar_history
        ]

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
            'object_bbx_mask': mask,
            'object_ids': [object_id_stack[i] for i in unique_indices],
            'anchor_box': anchor_box,
            'processed_lidar': lidar_dict,
            'processed_lidar_prev': lidar_prev_dict,
            'processed_lidar_history': lidar_history_dicts,
            'label_dict': label_dict,
            'future_waypoints': future_waypoints,
            'planning_target': planning_target,
            'history_actor_xy': history_actor_xy,
            'history_actor_mask': history_actor_mask,
            'history_ego_xy': history_ego_xy})

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

        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base], ego_pose)

        valid_object_bbx_center = object_bbx_center[object_bbx_mask == 1]

        vehicles = selected_cav_base['params']['vehicles']

        speed_list = []
        for obj_id in object_ids:
            speed = get_speed_by_object_id(vehicles, obj_id) #TODO
            speed_list.append(speed)

        speed_array = np.array(speed_list, dtype=np.float32).reshape(-1, 1)

        # shape: (num_objects, 8)
        valid_object_bbx_center = np.concatenate(
            [valid_object_bbx_center, speed_array],
            axis=1
        )

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

        lidar_np = lidar_np.astype(np.float32)

        # -----------------------------
        # Previous frame LiDAR (separate input)
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
        prev_lidar_np = prev_lidar_np.astype(np.float32)

        # -----------------------------
        # Full chronological history (oldest -> current)
        # -----------------------------
        projected_lidar_history = []
        history_lidar_list = selected_cav_base.get(
            'history_lidar_list', [lidar_np])
        history_params_list = selected_cav_base.get(
            'history_params_list', [selected_cav_base['params']])
        for hist_lidar, hist_params in zip(
                history_lidar_list, history_params_list):
            hist_pc = shuffle_points(hist_lidar.copy())
            hist_pc = mask_ego_points(hist_pc)
            hist_T = hist_params['transformation_matrix']
            hist_pc[:, :3] = box_utils.project_points_by_matrix_torch(
                hist_pc[:, :3], hist_T)
            projected_lidar_history.append(hist_pc.astype(np.float32))

        selected_cav_processed.update({
            'object_bbx_center': valid_object_bbx_center,
            'object_ids': object_ids,
            'projected_lidar': lidar_np,
            'projected_lidar_prev': prev_lidar_np,
            'projected_lidar_history': projected_lidar_history,
        })

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
            processed_lidar_prev_torch_dict = \
                self.pre_processor.collate_batch(
                    [cav_content['processed_lidar_prev']])
            processed_lidar_history = None
            if 'processed_lidar_history' in cav_content:
                processed_lidar_history = [
                    self.pre_processor.collate_batch([hist])
                    for hist in cav_content['processed_lidar_history']
                ]
            # label dictionary
            label_torch_dict = \
                self.post_processor.collate_batch([cav_content['label_dict']])

            # planning GT (only exists for the ego vehicle)
            planning_fields = {}
            if 'future_waypoints' in cav_content:
                planning_fields['future_waypoints'] = torch.from_numpy(
                    np.array([cav_content['future_waypoints']])
                ).float()
            if 'planning_target' in cav_content:
                planning_fields['planning_target'] = torch.from_numpy(
                    np.array([cav_content['planning_target']])
                ).float()
                planning_fields['history_actor_xy'] = torch.from_numpy(
                    np.array([cav_content['history_actor_xy']])
                ).float()
                planning_fields['history_actor_mask'] = torch.from_numpy(
                    np.array([cav_content['history_actor_mask']])
                ).float()
                planning_fields['history_ego_xy'] = torch.from_numpy(
                    np.array([cav_content['history_ego_xy']])
                ).float()

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()

            output_dict[cav_id].update({
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': object_bbx_mask,
                'processed_lidar': processed_lidar_torch_dict,
                'processed_lidar_prev': processed_lidar_prev_torch_dict,
                'label_dict': label_torch_dict,
                'object_ids': object_ids,
                'transformation_matrix': transformation_matrix_torch,
                **planning_fields})
            if processed_lidar_history is not None:
                output_dict[cav_id]['processed_lidar_history'] = \
                    processed_lidar_history

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
