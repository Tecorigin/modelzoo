# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='dinov2-small',
        img_size=518,
        patch_size=14,
        layer_scale_init_value=1e-5,
    ),
    neck=None,
    # head 被替换为下面的分类头配置
    head=dict(
        type='LinearClsHead',
        num_classes=1000,  # ImageNet-1k 数据集的类别数
        in_channels=384,   # ViT-Small backbone的输出维度
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5), # 计算 Top-1 和 Top-5 准确率
    ))

# data pre-processor
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# train dataloader
train_dataloader = dict(
    batch_size=16,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='/data/teco-data/imagenet',
        data_prefix='train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale=518),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs'),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

# validation dataloader
val_dataloader = dict(
    batch_size=16,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='/data/teco-data/imagenet',
        data_prefix='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=518, edge='short'),
            dict(type='CenterCrop', crop_size=518),
            dict(type='PackInputs'),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# evaluator
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# training schedule
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=90,
        by_epoch=True,
        begin=10,
        end=100,
    )
]

# default runtime settings
default_scope = 'mmpretrain'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)

# environment settings
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# visualization settings
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')])

# set log level
log_level = 'INFO'
load_from = None
resume = False

# work directory
work_dir = './work_dirs/dinov2_small_finetune'