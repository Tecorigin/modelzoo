EIGVAL = [
    0.2175,
    0.0188,
    0.0045,
]
EIGVEC = [
    [
        -0.5836,
        -0.6948,
        0.4203,
    ],
    [
        -0.5808,
        -0.0045,
        -0.814,
    ],
    [
        -0.5675,
        0.7192,
        0.4009,
    ],
]
auto_scale_lr = dict(base_batch_size=1024)
custom_hooks = [
    dict(type='CustomLogHook'),
]
data_preprocessor = dict(
    mean=[
        103.53,
        116.28,
        123.675,
    ],
    num_classes=1000,
    std=[
        57.375,
        57.12,
        58.395,
    ],
    to_rgb=False)
dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(arch='regnetx_400mf', type='RegNet'),
    head=dict(
        in_channels=384,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=1000,
        topk=(
            1,
            5,
        ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(
        lr=0.8, momentum=0.9, nesterov=True, type='SGD', weight_decay=5e-05))
param_scheduler = [
    dict(begin=0, by_epoch=True, end=5, start_factor=0.1, type='LinearLR'),
    dict(T_max=95, begin=5, by_epoch=True, end=100, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=128,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/imagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='val',
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(edge='short', scale=256, type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=128,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/imagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                alphastd=25.5,
                eigval=[
                    0.2175,
                    0.0188,
                    0.0045,
                ],
                eigvec=[
                    [
                        -0.5836,
                        -0.6948,
                        0.4203,
                    ],
                    [
                        -0.5808,
                        -0.0045,
                        -0.814,
                    ],
                    [
                        -0.5675,
                        0.7192,
                        0.4009,
                    ],
                ],
                to_rgb=False,
                type='Lighting'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=224, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(
        alphastd=25.5,
        eigval=[
            0.2175,
            0.0188,
            0.0045,
        ],
        eigvec=[
            [
                -0.5836,
                -0.6948,
                0.4203,
            ],
            [
                -0.5808,
                -0.0045,
                -0.814,
            ],
            [
                -0.5675,
                0.7192,
                0.4009,
            ],
        ],
        to_rgb=False,
        type='Lighting'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=128,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/imagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='val',
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/regnetx-400mf_8xb128_in1k'
