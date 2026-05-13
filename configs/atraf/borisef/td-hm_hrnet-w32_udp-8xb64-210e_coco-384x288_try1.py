_base_ = ['/home/borisef/projects/mm/mmpose/configs/_base_/default_runtime.py']

MY_BATCH = 1
work_dir =  '/home/borisef/projects/mm/mmpose/tools/atraf/borisef/work_dirs/hrnet_UDP_w32_try3'
resume = True

# runtime
train_cfg = dict(max_epochs=1210, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-3,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=2),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=1,
        save_best='coco/AP', rule='greater', max_keep_ckpts=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    #visualization=dict(type='PoseVisualizationHook', enable=True, interval =2, out_dir = work_dir + '/vvv'),
    visualization=dict(type='PoseVisualizationHookWithClassifiers', enable=True, interval=2, out_dir=work_dir + '/vvv'),
    badcase=dict(
        type='BadCaseAnalysisHook',
        enable=False,
        out_dir='badcase',
        metric_type='loss',
        badcase_thr=5)
)

# codec settings
codec = dict(
    type='UDPHeatmap', input_size=(288, 384), heatmap_size=(72, 96), sigma=3)

# model settings
model = dict(
    type='TopdownPoseEstimator',
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
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    head=dict(
        #type='HeatmapHead',
        type='HeatmapHeadWithClassifiers',
        classifiers = [
            dict(num_classes = 3, weight = 0.5, field_name = "gender",labels = ["M", "W", "S"],
                 loss_cfg=dict(type='CrossEntropyLoss', reduction='none'),
                 #loss_cfg=dict(type='FocalLoss', gamma=2.0, reduction='none'),
                 #loss_cfg=dict(type='BCEWithLogitsLoss', reduction='none'), # num classes = 1
                 num_convs = 2,  num_fcs = 2, conv_out_channels = 256, fc_out_channels = 256), #TODO: params of loss
            dict(num_classes = 2, weight = 0.5, field_name = "shape", labels = ["round", "rectangular"],
                 loss_cfg=dict(type='FocalLoss', gamma=2.0, reduction='none'),
                 num_convs=1, num_fcs=1, conv_out_channels=128, fc_out_channels=64 ), #TODO: params of loss
        ],
        in_channels=32,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        #TODO: define classifier's loss
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '/home/borisef/data/coco/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]
pipeline = val_pipeline
# data loaders
train_dataloader = dict(
    batch_size=MY_BATCH,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset = dict(
        type = 'ConcatDataset',
        datasets = [
            dict(
                type=dataset_type,
                indices = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18],
                data_root=data_root,
                data_mode=data_mode,
                ann_file='annotations/w1_person_keypoints_with_gender.json',
                data_prefix=dict(img='images/val2017/'),
                pipeline=train_pipeline,
            )
        ]*10
    )
)
val_dataloader = dict(
    batch_size=MY_BATCH,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type = 'ConcatDataset',
        datasets = [
            dict(
                type=dataset_type,
                indices=[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                data_root=data_root,
                data_mode=data_mode,
                ann_file='annotations/person_keypoints_with_gender.json',
                # bbox_file=data_root + 'person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
                data_prefix=dict(img='images/val2017/'),
                test_mode=True,
                pipeline=val_pipeline,
            ),
            dict(
                type=dataset_type,
                indices=[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                data_root=data_root,
                data_mode=data_mode,
                ann_file='annotations/person_keypoints_with_gender.json',
                data_prefix=dict(img='images/val2017/'),
                test_mode=True,
                pipeline=val_pipeline,
            )
        ],
        # type=dataset_type,
        # indices = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18],
        # data_root=data_root,
        # data_mode=data_mode,
        # ann_file='annotations/person_keypoints_with_gender.json',
        # # bbox_file=data_root + 'person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
        # data_prefix=dict(img='images/val2017/'),
        # test_mode=True,
        # pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
# val_evaluator = dict(
#     type='CocoMetric',
#     #ann_file=data_root + 'annotations/person_keypoints_with_gender.json'
# )

val_evaluator = [
    dict(type='PCKAccuracy', thr=0.2),
    dict(type='AUC'),
    dict(type='EPE'),
    dict(type='CocoMetric'),
    dict(
        type='ClassificationMetric',
        classifiers=[
            dict(field_name='gender', num_classes=3),
            dict(field_name='shape', num_classes=2),
        ]
    )
]
test_evaluator = val_evaluator

# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
