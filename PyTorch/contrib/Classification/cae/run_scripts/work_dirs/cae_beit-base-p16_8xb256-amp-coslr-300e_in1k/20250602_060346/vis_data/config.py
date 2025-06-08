auto_scale_lr = dict(base_batch_size=2048)
custom_hooks = [
    dict(type='CustomLogHook'),
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    second_mean=[
        -31.875,
        -31.875,
        -31.875,
    ],
    second_std=[
        318.75,
        318.75,
        318.75,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True,
    type='TwoNormDataPreprocessor')
data_root = '/data/teco-data/imagenet/'
dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=3, type='CheckpointHook'),
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
find_unused_parameters = True
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='b',
        bias='qv_bias',
        layer_scale_init_value=0.1,
        patch_size=16,
        type='CAEPretrainViT'),
    base_momentum=0.0,
    head=dict(loss=dict(lambd=2, type='CAELoss'), type='CAEHead'),
    neck=dict(
        decoder_depth=4,
        embed_dims=768,
        layer_scale_init_value=0.1,
        mlp_ratio=4,
        num_heads=12,
        regressor_depth=4,
        type='CAENeck'),
    target_generator=dict(
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/dalle_encoder.pth',
            type='Pretrained'),
        type='DALL-E'),
    type='CAE')
optim_wrapper = dict(
    clip_grad=dict(max_norm=3.0),
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0015, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0.0, flat_decay_mult=0.0, norm_decay_mult=0.0),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=10,
        start_factor=0.0001,
        type='LinearLR'),
    dict(
        T_max=290,
        begin=10,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        eta_min=1e-05,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=True, seed=0)
resume = False
train_cfg = dict(max_iters=100, type='IterBasedTrainLoop', val_interval=4000)
train_dataloader = dict(
    batch_size=50,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='train_list.txt',
        data_prefix=dict(img_path=''),
        data_root='/data/teco-data/imagenet/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                interpolation='bicubic',
                scale=(
                    0.08,
                    1.0,
                ),
                second_interpolation='lanczos',
                second_size=112,
                size=224,
                type='RandomResizedCropAndInterpolationWithTwoPic'),
            dict(
                input_size=(
                    14,
                    14,
                ),
                max_num_patches=None,
                min_num_patches=16,
                num_masking_patches=75,
                type='BEiTMaskGenerator'),
            dict(type='PackInputs'),
        ],
        type='ImageNet'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        interpolation='bicubic',
        scale=(
            0.08,
            1.0,
        ),
        second_interpolation='lanczos',
        second_size=112,
        size=224,
        type='RandomResizedCropAndInterpolationWithTwoPic'),
    dict(
        input_size=(
            14,
            14,
        ),
        max_num_patches=None,
        min_num_patches=16,
        num_masking_patches=75,
        type='BEiTMaskGenerator'),
    dict(type='PackInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/cae_beit-base-p16_8xb256-amp-coslr-300e_in1k'
