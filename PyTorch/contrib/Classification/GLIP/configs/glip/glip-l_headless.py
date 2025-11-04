model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='large',
        img_size=224,
        out_indices=(3,),
        stage_cfgs=dict(
            block_cfgs=dict(
                window_size=7,  # 减小窗口大小
                pad_small_map=True,  # 添加小特征图填充选项
            )
        ),
        drop_path_rate=0.2
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)

data_preprocessor = dict(
    type='ClsDataPreprocessor',
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    to_rgb=False,
)

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ImageNet',
        data_root='/data/teco-data/imagenet',
        split='train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                scale=224,
                crop_ratio_range=(0.08, 1.0),
            ),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(
                type='PackInputs',
                algorithm_keys=['gt_label']
            )
        ]
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ImageNet',
        data_root='/data/teco-data/imagenet',
        split='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeEdge',
                scale=224,
                edge='short',
                backend='pillow',
            ),
            dict(
                type='CenterCrop',
                crop_size=224,
            ),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(
                type='PackInputs',
                algorithm_keys=['gt_label']
            )
        ]
    )
)

# 优化器配置
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.0001,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=1.0),
    loss_scale='dynamic',
)

# 学习率调度器
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    dict(
        type='MultiStepLR',
        milestones=[8, 11],
        by_epoch=True,
        gamma=0.1
    )
]

train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='Accuracy', topk=(1, 5))

default_scope = 'mmpretrain'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Visualizer',
    vis_backends=vis_backends
)