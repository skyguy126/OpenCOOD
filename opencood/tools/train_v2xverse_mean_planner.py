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
# GPU 0 is NUMA 0 (0-19,40-59). The other active jobs hold 12-19,52-59
# and 20-28,60-68. Same scheme as those jobs: main thread on the reserved
# set; each loader worker on one logical CPU, starting at the next core
# and using both HT threads as separate slots.
CPU_AFFINITY = '0-11'


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


def bind_like_active_jobs(spec, num_workers):
    """Match the live planner jobs: wide main mask, one logical CPU per worker."""
    from opencood.tools import train_utils
    groups = train_utils.cpu_core_groups(spec)
    if not groups:
        raise ValueError('cpu_affinity %r did not match any CPUs' % spec)
    main_cpus = sorted(cpu for group in groups for cpu in group)
    train_utils._limit_blas_threads()
    train_utils.apply_cpu_affinity(main_cpus, label='main process')

    slots = []
    for group in groups[1:]:
        for cpu in group:
            slots.append(cpu)
            if len(slots) >= num_workers:
                break
        if len(slots) >= num_workers:
            break
    if not slots:
        slots = list(main_cpus)
    print('CPU affinity: main on %s; workers on %s'
          % (','.join(str(cpu) for cpu in main_cpus),
             ','.join(str(cpu) for cpu in slots)))
    if num_workers > len(slots):
        print('Warning: %d workers share %d CPUs'
              % (num_workers, len(slots)))

    def _init(worker_id):
        train_utils._limit_blas_threads()
        cpu = slots[worker_id % len(slots)]
        train_utils.apply_cpu_affinity(
            [cpu], label='dataloader worker %d' % worker_id)

    return _init


def main():
    extra = sys.argv[1:]
    if '--hypes_yaml' not in extra:
        extra = ['--hypes_yaml', HYPES] + extra
    if '--model_dir' not in extra:
        extra += ['--model_dir', MODEL_DIR]
    if '--pretrained_dir' not in extra:
        extra += ['--pretrained_dir', PRETRAINED]
    if '--cpu_affinity' not in extra:
        extra += ['--cpu_affinity', CPU_AFFINITY]
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

    import opencood.tools.train_utils as train_utils
    train_utils.bind_job_affinity = bind_like_active_jobs

    from opencood.tools.train import main as train_main
    train_main()


if __name__ == '__main__':
    main()
