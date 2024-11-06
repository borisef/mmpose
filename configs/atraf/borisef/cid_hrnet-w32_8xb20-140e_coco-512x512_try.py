_base_ = ['/home/borisef/projects/mm/mmpose/configs/_base_/default_runtime.py']

MY_BATCH_SIZE = 1

work_dir = '/home/borisef/projects/mm/mmpose/tools/atraf/borisef/work_dirs/cid'
load_from = "/home/borisef/models/mmpose/cid_hrnet-w32_8xb20-140e_coco-512x512_42b7e6e6-20230207.pth"

# runtime
train_cfg = dict(max_epochs=140, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-3,
))

# learning policy
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=140,
        milestones=[90, 120],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=160)

# hooks
# hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=1,
        save_best='coco/AP', rule='greater', max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=True, interval =100, out_dir = work_dir + '/vvv'),
    badcase=dict(
        type='BadCaseAnalysisHook',
        enable=False,
        out_dir='badcase',
        metric_type='loss',
        badcase_thr=5)
)

# codec settings
codec = dict(
    type='DecoupledHeatmap', input_size=(512, 512), heatmap_size=(128, 128))

# model settings
model = dict(
    type='BottomupPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
                multiscale_output=True)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    neck=dict(
        type='FeatureMapProcessor',
        concat=True,
    ),
    head=dict(
        type='CIDHead',
        in_channels=480,
        num_keypoints=17,
        gfd_channels=32,
        coupled_heatmap_loss=dict(type='FocalHeatmapLoss', loss_weight=1.0),
        decoupled_heatmap_loss=dict(type='FocalHeatmapLoss', loss_weight=4.0),
        contrastive_loss=dict(
            type='InfoNCELoss', temperature=0.05, loss_weight=1.0),
        decoder=codec,
    ),
    train_cfg=dict(max_train_instances=200),
    test_cfg=dict(
        multiscale_test=False,
        flip_test=True,
        shift_heatmap=False,
        align_corners=False))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'bottomup'


# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupRandomAffine', input_size=codec['input_size']),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='BottomupGetHeatmapMask'),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize',
        input_size=codec['input_size'],
        size_factor=64,
        resize_mode='expand'),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'crowd_index', 'ori_shape',
                   'img_shape', 'input_size', 'input_center', 'input_scale',
                   'flip', 'flip_direction', 'flip_indices', 'raw_ann_info',
                   'skeleton_links'))
]

data_root = '/home/borisef/data/coco/'
MY_TRAIN_DATASET =dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='images/val2017/'),
        pipeline=train_pipeline,
    )
MY_VAL_DATASET =dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='images/val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    )
# data loaders
train_dataloader = dict(
    batch_size=MY_BATCH_SIZE,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=MY_TRAIN_DATASET
)

val_dataloader = dict(
    batch_size=1,
    num_workers=3,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=MY_VAL_DATASET)
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json',
    nms_thr=0.8,
    score_mode='keypoint',
)
test_evaluator = val_evaluator
