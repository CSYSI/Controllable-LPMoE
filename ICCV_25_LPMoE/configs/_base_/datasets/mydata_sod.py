# dataset settings
dataset_type = 'MydataDataset'
data_root = ''
img_norm_cfg = dict(
    mean=[125.76, 118.39, 101.49], std=[69.81, 67.87, 72.41], to_rgb=True)
img_norm_cfg_test = dict(    mean=[117.28, 112.61, 92.87], std=[64.76, 61.91, 66.10], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),

        flip=True,
        transforms=[
            dict(type='SETR_Resize', keep_ratio=True, crop_size=(512, 512), setr_multi_scale=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip', prob=1.0),
            dict(type='Normalize', **img_norm_cfg_test),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation_ecssd',
        ann_dir='annotations/validation_ecssd',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation_ecssd',
        ann_dir='annotations/validation_ecssd',
        pipeline=test_pipeline))
