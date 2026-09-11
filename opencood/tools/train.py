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
    # Overlay DataLoader keys from the explicitly passed --hypes_yaml so
    # loader patches take effect without rewriting the saved config.
    if opt.model_dir and not pretrained_dir:
        repo_train_params = dict(train_params)
        hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
        train_params = hypes.get('train_params', {})
        loader_keys = (
            'num_workers', 'val_num_workers', 'pin_memory',
            'prefetch_factor', 'val_prefetch_factor',
            'persistent_workers', 'val_persistent_workers',
            'max_workers_per_rank_train', 'max_workers_per_rank_val',
        )
        for key in loader_keys:
            if key in repo_train_params:
                train_params[key] = repo_train_params[key]
        hypes['train_params'] = train_params

    pretrained_dir = opt.pretrained_dir or train_params.get('pretrained_dir', '')
    freeze_backbone = train_params.get('freeze_backbone', False)
    gradient_clip_norm = train_params.get('gradient_clip_norm', None)

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

    loader_train_kwargs = train_utils.dataloader_kwargs(
        train_params, distributed=opt.distributed, shuffle=True,
        drop_last=True, is_train=True)
    loader_val_kwargs = train_utils.dataloader_kwargs(
        train_params, distributed=opt.distributed, shuffle=False,
        drop_last=False, is_train=False)
    loader_val_kwargs['drop_last'] = False

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        dist_train_kwargs = {
            k: v for k, v in loader_train_kwargs.items()
            if k not in ('shuffle', 'drop_last')
        }
        dist_val_kwargs = {
            k: v for k, v in loader_val_kwargs.items()
            if k not in ('shuffle',)
        }

        train_loader = DataLoader(
            opencood_train_dataset,
            batch_sampler=batch_sampler_train,
            collate_fn=opencood_train_dataset.collate_batch_train,
            **dist_train_kwargs)
        val_loader = DataLoader(
            opencood_validate_dataset,
            sampler=sampler_val,
            batch_size=hypes['train_params']['batch_size'],
            collate_fn=opencood_train_dataset.collate_batch_train,
            **dist_val_kwargs)
        print("DEBUG validation dataset length:", len(opencood_validate_dataset))
        print("DEBUG validation dataloader length:", len(val_loader))

    else:
        train_loader = DataLoader(
            opencood_train_dataset,
            batch_size=hypes['train_params']['batch_size'],
            collate_fn=opencood_train_dataset.collate_batch_train,
            **loader_train_kwargs)
        val_loader = DataLoader(
            opencood_validate_dataset,
            batch_size=hypes['train_params']['batch_size'],
            collate_fn=opencood_train_dataset.collate_batch_train,
            **loader_val_kwargs)

        print("DEBUG validate_dir:", hypes['validate_dir'])
        print("DEBUG validation dataset length:", len(opencood_validate_dataset))
        print("DEBUG validation dataloader length:", len(val_loader))

    print('DataLoader train: num_workers=%d pin_memory=%s prefetch_factor=%s '
          'persistent_workers=%s'
          % (loader_train_kwargs.get('num_workers'),
             loader_train_kwargs.get('pin_memory'),
             loader_train_kwargs.get('prefetch_factor', 'n/a'),
             loader_train_kwargs.get('persistent_workers', False)))
    print('DataLoader val:   num_workers=%d pin_memory=%s prefetch_factor=%s '
          'persistent_workers=%s'
          % (loader_val_kwargs.get('num_workers'),
             loader_val_kwargs.get('pin_memory'),
             loader_val_kwargs.get('prefetch_factor', 'n/a'),
             loader_val_kwargs.get('persistent_workers', False)))

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
        # Resume: load full checkpoint (backbone + planner). Detection/speed
        # weights are already in net_epoch*.pth from the earlier pretrained
        # init — do not pass --pretrained_dir or training restarts at epoch 0
        # with a randomly initialized planner.
        saved_path = opt.model_dir
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        config_path = os.path.join(saved_path, 'config.yaml')
        if not os.path.exists(config_path):
            yaml_utils.save_yaml(hypes, config_path)
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        if freeze_backbone:
            model = train_utils.freeze_backbone(model)
    else:
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[opt.gpu],
                find_unused_parameters=False)
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
    scheduler_method = hypes['lr_scheduler']['core_method'].lower().replace(
        '_', '')
    # CosineAnnealingLR is stepped once per epoch *after* the epoch (V2Xverse).
    # MultiStep/Step still use the historical OpenCOOD begin-of-epoch step.
    step_scheduler_at_epoch_start = scheduler_method not in (
        'cosineannealinglr', 'cosineannealing', 'cosineannealwarm')

    is_main_process = (not opt.distributed) or (getattr(opt, 'rank', 0) == 0)
    best_val_loss = float('inf')
    best_epoch = -1
    best_meta_path = os.path.join(saved_path, 'best_epoch.yaml')
    if os.path.exists(best_meta_path):
        try:
            import yaml as _yaml
            with open(best_meta_path, 'r') as f:
                best_meta = _yaml.safe_load(f) or {}
            best_epoch = int(best_meta.get('best_epoch', -1))
            best_val_loss = float(best_meta.get('metric_value', best_val_loss))
            print('Resumed best-epoch tracker: epoch=%d val_loss=%.6f'
                  % (best_epoch, best_val_loss))
        except Exception as e:
            print('Warning: could not load best_epoch.yaml (%s)' % e)

    use_cuda = torch.cuda.is_available()

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if step_scheduler_at_epoch_start:
            if scheduler_method != 'cosineannealwarm':
                scheduler.step(epoch)
        if scheduler_method == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        train_utils.configure_frozen_training(model_without_ddp,
                                              freeze_backbone)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(
                batch_data, device, non_blocking=use_cuda)

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
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad,
                               model_without_ddp.parameters()),
                        gradient_clip_norm)
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad,
                               model_without_ddp.parameters()),
                        gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            if scheduler_method == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

        # V2Xverse CosineAnnealingLR: step once at end of epoch.
        if scheduler_method in ('cosineannealinglr', 'cosineannealing'):
            scheduler.step()

        if epoch % hypes['train_params']['save_freq'] == 0 and is_main_process:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            model_without_ddp.eval()

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    batch_data = train_utils.to_device(
                        batch_data, device, non_blocking=use_cuda)
                    ouput_dict = model(batch_data['ego'])
                    target_dict = batch_data['ego']['label_dict']
                    if planning_head_cfg.get('enabled', False):
                        target_dict['future_waypoints'] = \
                            batch_data['ego']['future_waypoints']

                    final_loss = criterion(ouput_dict, target_dict)
                    valid_ave_loss.append(final_loss.item())

            if len(valid_ave_loss) > 0:
                valid_ave_loss = statistics.mean(valid_ave_loss)
                print('At epoch %d, the validation loss is %f' % (epoch,
                                                                  valid_ave_loss))
            else:
                print('At epoch %d, validation skipped: no validation batches.'
                      % epoch)
                valid_ave_loss = None

            if valid_ave_loss is not None:
                writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)
                if is_main_process and valid_ave_loss < best_val_loss:
                    best_val_loss = valid_ave_loss
                    best_epoch = epoch + 1
                    train_utils.save_best_checkpoint(
                        model_without_ddp, saved_path, best_epoch,
                        'val_loss', best_val_loss)
                    print('New best epoch %d (val_loss=%.6f) -> net_best.pth'
                          % (best_epoch, best_val_loss))
                    writer.add_scalar('Best_Val_Loss', best_val_loss, epoch)

            train_utils.configure_frozen_training(model_without_ddp,
                                                  freeze_backbone)

    if is_main_process and best_epoch > 0:
        print('Best epoch: %d (val_loss=%.6f) saved as net_best.pth'
              % (best_epoch, best_val_loss))
    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
