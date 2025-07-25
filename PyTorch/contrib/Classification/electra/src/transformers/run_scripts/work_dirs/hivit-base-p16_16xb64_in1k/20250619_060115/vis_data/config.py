auto_scale_lr = dict(base_batch_size=1024)
bgr_mean = [
    103.53,
    116.28,
    123.675,
]
bgr_std = [
    57.375,
    57.12,
    58.395,
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=1000,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
data_root = 'data/imagenet/'
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
    backbone=dict(
        ape=True,
        arch='base',
        drop_path_rate=0.5,
        img_size=224,
        rpe=True,
        type='HiViT'),
    head=dict(
        cal_acc=False,
        in_channels=512,
        init_cfg=None,
        loss=dict(
            label_smooth_val=0.1, mode='original', type='LabelSmoothLoss'),
        num_classes=1000,
        type='LinearClsHead'),
    init_cfg=[
        dict(bias=0.0, layer='Linear', std=0.02, type='TruncNormal'),
        dict(bias=0.0, layer='LayerNorm', type='Constant', val=1.0),
    ],
    neck=dict(type='GlobalAveragePooling'),
    train_cfg=dict(augments=[
        dict(alpha=0.8, type='Mixup'),
        dict(alpha=1.0, type='CutMix'),
    ]),
    type='ImageClassifier')
optim_wrapper = dict(
    clip_grad=dict(max_norm=5.0),
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.001,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
        flat_decay_mult=0.0,
        norm_decay_mult=0.0),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        start_factor=0.001,
        type='LinearLR'),
    dict(begin=20, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='val_copy_list.txt',
        data_prefix='',
        data_root='data/imagenet/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
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
    dict(
        backend='pillow',
        edge='short',
        interpolation='bicubic',
        scale=256,
        type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
train_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='train_list.txt',
        data_prefix='',
        data_root='/data/teco-data/imagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=224,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                hparams=dict(
                    interpolation='bicubic', pad_val=[
                        104,
                        116,
                        124,
                    ]),
                magnitude_level=9,
                magnitude_std=0.5,
                num_policies=2,
                policies='timm_increasing',
                total_level=10,
                type='RandAugment'),
            dict(
                erase_prob=0.25,
                fill_color=[
                    103.53,
                    116.28,
                    123.675,
                ],
                fill_std=[
                    57.375,
                    57.12,
                    58.395,
                ],
                max_area_ratio=0.3333333333333333,
                min_area_ratio=0.02,
                mode='rand',
                type='RandomErasing'),
            dict(type='PackInputs'),
        ],
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=224,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(
        hparams=dict(interpolation='bicubic', pad_val=[
            104,
            116,
            124,
        ]),
        magnitude_level=9,
        magnitude_std=0.5,
        num_policies=2,
        policies='timm_increasing',
        total_level=10,
        type='RandAugment'),
    dict(
        erase_prob=0.25,
        fill_color=[
            103.53,
            116.28,
            123.675,
        ],
        fill_std=[
            57.375,
            57.12,
            58.395,
        ],
        max_area_ratio=0.3333333333333333,
        min_area_ratio=0.02,
        mode='rand',
        type='RandomErasing'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='val_copy_list.txt',
        data_prefix='',
        data_root='/data/teco-data/imagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
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
work_dir = './work_dirs/hivit-base-p16_16xb64_in1k'
