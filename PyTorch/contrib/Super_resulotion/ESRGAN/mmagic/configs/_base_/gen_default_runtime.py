# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import CheckpointHook, LoggerHook
from mmengine.model import MMSeparateDistributedDataParallel

from mmagic.engine import (IterTimerHook, LogProcessor,
                           MultiOptimWrapperConstructor, MultiTestLoop,
                           MultiValLoop)
from mmagic.evaluation import Evaluator
from mmagic.visualization import VisBackend, Visualizer

default_scope = 'mmagic'

randomness = dict(seed=2022, diff_rank_seed=True)
# env settings
dist_params = dict(backend='tccl')
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# configure for default hooks
default_hooks = dict(
    # record time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 100 iterations.
    logger=dict(type=LoggerHook, interval=100, log_metric_by_epoch=False),
    # save checkpoint per 10000 iterations
    checkpoint=dict(
        type=CheckpointHook,
        interval=10000,
        by_epoch=False,
        max_keep_ckpts=20,
        less_keys=['FID-Full-50k/fid', 'FID-50k/fid', 'swd/avg'],
        greater_keys=['IS-50k/is', 'ms-ssim/avg'],
        save_optimizer=True))

# config for environment
env_cfg = dict(
    # whether to enable cudnn benchmark.
    cudnn_benchmark=True,
    # set multi process parameters.
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters.
    dist_cfg=dict(backend='tccl'))

# set log level
log_level = 'INFO'
log_processor = dict(type=LogProcessor, by_epoch=False)

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = None

# config for model wrapper
model_wrapper_cfg = dict(
    type=MMSeparateDistributedDataParallel,
    broadcast_buffers=False,
    find_unused_parameters=False)

# set visualizer
vis_backends = [dict(type=VisBackend)]
visualizer = dict(type=Visualizer, vis_backends=vis_backends)

# config for training
train_cfg = dict(by_epoch=False, val_begin=1, val_interval=10000)

# config for val
val_cfg = dict(type=MultiValLoop)
val_evaluator = dict(type=Evaluator)

# config for test
test_cfg = dict(type=MultiTestLoop)
test_evaluator = dict(type=Evaluator)

# config for optim_wrapper_constructor
optim_wrapper = dict(constructor=MultiOptimWrapperConstructor)
