_base_ = [
    '../_base_/models/ssd300_ReAct.py', '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py'
]


model = dict(
    type='SingleStageDetector_PCGrad',
    backbone=dict(
        out_feature_indices=(13,24)
    ),
    bbox_head=dict(
        num_classes=20, 
        anchor_generator=dict(basesize_ratio_range=(0.2,0.9))
            ))

train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)

test_cfg = dict(
    nms=dict(type='nms', iou_threshold=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)


# dataset settings
dataset_type = 'VOCDataset'
data_root = '../data/VOCdevkit/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=15,
    workers_per_gpu=3,
    train=dict(
        type='RepeatDataset', times=1, dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
auto_scale_lr = dict(base_batch_size=60)
optimizer = dict(type='SAM_BNN',lr=1e-3, weight_decay=0, c = 0.2, reduction='sum')
optimizer_config = dict(type="GradientCumulativeOptimizerHookForPC", cumulative_iters=5)

lr_config = dict(
    policy = 'Poly',
    power = 1,
    min_lr = 0,
    )

checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

total_epochs =300 
runner= dict(type = 'BopRunner', max_epochs=300)