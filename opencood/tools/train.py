# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import sys

# Prefer this repo over another OpenCOOD copy on PYTHONPATH.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import statistics

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

# Default off for the original detection+speed training entrypoint.
# Path-only training is enabled via point_pillar_early_fusion_x2_path.yaml.
ENABLE_PLANNING_HEAD = False


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path or output directory')
    parser.add_argument('--pretrained_dir', default='',
                        help='Directory with a pretrained detection checkpoint '
                             'to load before path-head training')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    train_params = hypes.get('train_params', {})
    pretrained_dir = opt.pretrained_dir or train_params.get('pretrained_dir', '')

    # Resume an existing run (uses model_dir/config.yaml).
    if opt.model_dir and not pretrained_dir:
        hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
        train_params = hypes.get('train_params', {})

    pretrained_dir = opt.pretrained_dir or train_params.get('pretrained_dir', '')
    freeze_backbone = train_params.get('freeze_backbone', False)

    planning_head_cfg = hypes['model']['args'].setdefault('planning_head', {})
    if ENABLE_PLANNING_HEAD:
        planning_head_cfg['enabled'] = True
        hypes['loss']['args']['enable_planning'] = True
    else:
        planning_head_cfg['enabled'] = planning_head_cfg.get('enabled', False)
        hypes['loss']['args']['enable_planning'] = planning_head_cfg['enabled']

    hypes['model']['args']['freeze_backbone'] = freeze_backbone
    planning_only = hypes['loss']['args'].get('planning_only', False)

    if planning_head_cfg.get('enabled', False):
        if freeze_backbone:
            print('Planning head enabled with frozen backbone. '
                  'Training objectives: future waypoints only.')
        elif planning_only:
            print('Planning head enabled. Training objectives: '
                  'future waypoints only.')
        else:
            print('Planning head enabled. Training objectives: detection (cls), '
                  'box regression (x,y,z,h,w,l,yaw), speed, future waypoints.')
    else:
        print('Planning head disabled. Training objectives: detection (cls), '
              'box regression (x,y,z,h,w,l,yaw), speed.')

    multi_gpu_utils.init_distributed_mode(opt)

    if hypes.get('model', {}).get('args', {}).get('dual_frame'):
        import inspect
        import opencood.data_utils.datasets.early_fusion_dataset as efd_mod
        efd_path = inspect.getfile(efd_mod.EarlyFusionDataset)
        print('dual_frame enabled; EarlyFusionDataset loaded from:', efd_path)
        if 'projected_lidar_prev' not in inspect.getsource(
                efd_mod.EarlyFusionDataset.get_item_single_car):
            raise RuntimeError(
                "dual_frame is enabled in the yaml but EarlyFusionDataset "
                "does not provide separate prev/current lidar. Train from "
                "OpenCOOD_vamsi_2, not the legacy OpenCOOD tree that stacks "
                "frames with a time-lag channel.")

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
        print("DEBUG validation dataset length:", len(opencood_validate_dataset))
        print("DEBUG validation dataloader length:", len(val_loader)) 
    
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)

        print("DEBUG validate_dir:", hypes['validate_dir'])
        print("DEBUG validation dataset length:", len(opencood_validate_dataset))
        print("DEBUG validation dataloader length:", len(val_loader))

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_epoch = 0
    if pretrained_dir:
        pretrained_epoch = train_params.get('pretrained_epoch', 15)
        checkpoint_path = os.path.join(
            pretrained_dir, 'net_epoch%d.pth' % pretrained_epoch)
        model = train_utils.load_pretrained_weights(model, checkpoint_path)
        if freeze_backbone:
            model = train_utils.freeze_backbone(model)
        if opt.model_dir:
            if os.path.abspath(opt.model_dir) == os.path.abspath(pretrained_dir):
                print('model_dir matches pretrained_dir; writing checkpoints '
                      'to a new logs folder to avoid overwriting pretrained '
                      'weights.')
                saved_path = train_utils.setup_train(hypes)
            else:
                saved_path = opt.model_dir
                if not os.path.exists(saved_path):
                    os.makedirs(saved_path)
                yaml_utils.save_yaml(hypes,
                                     os.path.join(saved_path, 'config.yaml'))
        else:
            saved_path = train_utils.setup_train(hypes)
    elif opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
    else:
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):
            train_utils.configure_frozen_training(model_without_ddp,
                                                  freeze_backbone)
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                target_dict = batch_data['ego']['label_dict']
                if planning_head_cfg.get('enabled', False):
                    target_dict['future_waypoints'] = \
                        batch_data['ego']['future_waypoints']
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict, target_dict)
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    target_dict = batch_data['ego']['label_dict']
                    if planning_head_cfg.get('enabled', False):
                        target_dict['future_waypoints'] = \
                            batch_data['ego']['future_waypoints']
                    final_loss = criterion(ouput_dict, target_dict)


            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    train_utils.configure_frozen_training(
                        model_without_ddp, freeze_backbone)

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])
                    target_dict = batch_data['ego']['label_dict']
                    if planning_head_cfg.get('enabled', False):
                        target_dict['future_waypoints'] = \
                            batch_data['ego']['future_waypoints']

                    final_loss = criterion(ouput_dict, target_dict)
                    valid_ave_loss.append(final_loss.item())
            # valid_ave_loss = statistics.mean(valid_ave_loss)
            # print('At epoch %d, the validation loss is %f' % (epoch,
            #                                                   valid_ave_loss))

            if len(valid_ave_loss) > 0:
                valid_ave_loss = statistics.mean(valid_ave_loss)
                print('At epoch %d, the validation loss is %f' % (epoch, valid_ave_loss))
            else:
                print('At epoch %d, validation skipped: no validation batches.' % epoch)
                valid_ave_loss = None
            
            if valid_ave_loss is not None:
                writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
