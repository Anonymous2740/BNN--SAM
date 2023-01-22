# model settings
model = dict(
    type='YOLOV2',
    # pretrained='./checkpoints/darknet19.pt',
    # pretrained='./checkpoints/darknet19_convert.pth',
    # pretrained='./checkpoints/darknet19_72.96.pth',

    backbone=dict(type='DarkNet19'),
    neck=dict(
        type='YOLOV2Neck',
        in_channels=[1024, 512, 1280],
        out_channels=[1024, 64, 1024]),
    bbox_head=dict(
        type='YOLOV2Head',
        num_classes=80,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(18, 21), (60, 66), (106, 175), (252, 112), (312, 293)]],
            strides=[32]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='GridAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0))
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    conf_thr=0.005,
    nms=dict(type='nms', iou_threshold=0.45),
    max_per_img=100)