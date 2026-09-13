# -*- coding: utf-8 -*-
"""Train the original V2Xverse mean-pool planner on the no-velocity backbone.

Single-process only. Does not modify the attention planner. Checkpoints load
point_pillar_v2xverse_mean (uniform spatial mean, no spatial_attn).
"""
import os
import sys

# Before torch is imported: one OpenMP thread per worker so 12 loaders do not
# oversubscribe the machine.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

HYPES = os.path.join(
    _REPO_ROOT,
    'opencood/hypes_yaml/point_pillar_early_fusion_baseline_det_only_mean.yaml',
)
PRETRAINED = '/home/project/x2_detection_only'
MODEL_DIR = '/home/project/path_v2xverse_det_only'


def _force_single_process():
    """Skip NCCL even if launched under torch.distributed.run --nproc=1."""
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world > 1:
        raise SystemExit(
            'train_v2xverse_mean_planner.py is single-GPU only '
            '(WORLD_SIZE=%d). Do not use torch.distributed.run.' % world)
    for key in (
            'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE',
            'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE',
            'MASTER_ADDR', 'MASTER_PORT', 'TORCHELASTIC_RUN_ID'):
        os.environ.pop(key, None)


def main():
    extra = sys.argv[1:]
    if '--hypes_yaml' not in extra:
        extra = ['--hypes_yaml', HYPES] + extra
    if '--model_dir' not in extra:
        extra += ['--model_dir', MODEL_DIR]
    if '--pretrained_dir' not in extra:
        extra += ['--pretrained_dir', PRETRAINED]
    sys.argv = [sys.argv[0]] + extra

    _force_single_process()
    print('Original V2Xverse mean-pool planner (no spatial attention).')
    print('Single process on CUDA_VISIBLE_DEVICES=%s'
          % os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    print('Config:', HYPES)
    print('Pretrained:', PRETRAINED)
    print('Output:', MODEL_DIR)

    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    from opencood.tools.train import main as train_main
    train_main()


if __name__ == '__main__':
    main()
