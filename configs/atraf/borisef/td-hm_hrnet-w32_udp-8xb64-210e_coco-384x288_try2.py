_base_ = ['/home/borisef/projects/mm/mmpose/configs/_base_/default_runtime.py']

MY_BATCH = 1
work_dir =  '/home/borisef/projects/mm/mmpose/tools/atraf/borisef/work_dirs/hrnet_UDP_w32_try2'
resume = True

# runtime
train_cfg = dict(max_epochs=40, val_interval=1)
    # Confusion matrix visualization for classifier heads

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
        end=40,
        milestones=[15, 18],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

custom_hooks = [
    dict(type='ClassifierLRSchedulerHook'),
    dict(type='KeypointFreezeHook', freeze_epoch=20),
]


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
                 num_convs = 2,  num_fcs = 2, conv_out_channels = 256, fc_out_channels = 256,
                 lr_schedule=[(0, 0.0), (20, 0.001)],
                 ),
            dict(num_classes = 2, weight = 0.5, field_name = "shape", labels = ["round", "rectangular"],
                 loss_cfg=dict(type='FocalLoss', gamma=2.0, reduction='none'),
                 num_convs=1, num_fcs=1, conv_out_channels=128, fc_out_channels=64,
                 lr_schedule=[(0, 0.0), (20, 0.001)],
                 ),
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
    ),
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

    # =====================================================================
    # NEW FEATURE: ignore_gt_out_of_image
    # Skip keypoints whose GT coordinates are outside the image boundaries
    # (negative or exceeding image dimensions). Default: False
    # =====================================================================
    dict(type='AtrafPCKAccuracy', kpt_indexes=[0,1,2,3], prefix='pck_no_ooi', thr=0.2,
         ignore_gt_out_of_image=True),
    dict(type='AtrafAUC', kpt_indexes=[0,1,2,3], prefix='auc_no_ooi',
         ignore_gt_out_of_image=True),
    dict(type='AtrafEPE', kpt_indexes=[0,1,2,3], prefix='epe_no_ooi',
         ignore_gt_out_of_image=True),

    # =====================================================================
    # NEW FEATURE: ignore_gt_out_of_bbox
    # Skip keypoints whose GT coordinates are outside the detection bbox.
    # Stricter than ignore_gt_out_of_image. Default: False
    # Both can be used together.
    # =====================================================================
    dict(type='AtrafPCKAccuracy', kpt_indexes=[0,1,2,3], prefix='pck_no_oob', thr=0.2,
         ignore_gt_out_of_image=True, ignore_gt_out_of_bbox=True),
    dict(type='AtrafAUC', kpt_indexes=[0,1,2,3], prefix='auc_no_oob',
         ignore_gt_out_of_image=True, ignore_gt_out_of_bbox=True),
    dict(type='AtrafEPE', kpt_indexes=[0,1,2,3], prefix='epe_no_oob',
         ignore_gt_out_of_image=True, ignore_gt_out_of_bbox=True),

    # =====================================================================
    # NEW FEATURE: Accuracy metric (correct_high / total_valid)
    # Recall_Atraf now also reports Accuracy alongside Recall/FAR/Precision/F1.
    # With ignore_gt_out_of_bbox to exclude out-of-bbox GT keypoints.
    # =====================================================================
    dict(type='Recall_Atraf', kpt_indexes=[0,1,2,3], prefix='recall_oob', thr=0.2,
         score_threshold=0.5, ignore_gt_out_of_bbox=True),
    dict(type='FAR_atraf', kpt_indexes=[0,1,2,3], prefix='far_oob', thr=0.2,
         score_threshold=0.5, ignore_gt_out_of_bbox=True),

    # =====================================================================
    # NEW FEATURE: SmartAccuracy (accuracy at the optimal F1 threshold)
    # Smart_F1 now also reports SmartAccuracy alongside SmartF1/SmartThreshold.
    # =====================================================================
    dict(type='Smart_F1', kpt_indexes=[0,1,2,3], prefix='smart_oob',
         thr=0.2, num_steps=101, ignore_gt_out_of_bbox=True,
         generate_chart=True, chart_images_folder=work_dir + '/smart_f1_charts'),

    # =====================================================================
    # NEW FEATURE: torso_keypoint_indexes
    # Custom keypoint pair for torso normalization (replaces hardcoded [4,5]).
    # Here we use keypoints [1,10] as the torso reference pair.
    # =====================================================================
    dict(type='AtrafPCKAccuracy', kpt_indexes=[0,1,2,3], prefix='pck_torso_custom',
         thr=0.2, norm_item='torso', torso_keypoint_indexes=[1, 10]),
    dict(type='Recall_Atraf', kpt_indexes=[0,1,2,3], prefix='recall_torso_custom',
         thr=0.2, score_threshold=0.5, norm_item='torso', torso_keypoint_indexes=[1, 10]),
    dict(type='Smart_F1', kpt_indexes=[0,1,2,3], prefix='smart_torso_custom',
         thr=0.2, num_steps=101, norm_item='torso', torso_keypoint_indexes=[1, 10],
         generate_chart=True, chart_images_folder=work_dir + '/smart_f1_torso_charts'),

    # =====================================================================
    # COMBINED EXAMPLE: all new features together
    # ignore_gt_out_of_image + ignore_gt_out_of_bbox + torso_keypoint_indexes
    # + twin_keypoints + Accuracy/SmartAccuracy
    # =====================================================================
    dict(type='Recall_Atraf', kpt_indexes=[0,1,2,3], prefix='recall_all', thr=0.2,
         score_threshold=0.5, norm_item=['bbox', 'torso'],
         ignore_gt_out_of_image=True, ignore_gt_out_of_bbox=True,
         torso_keypoint_indexes=[1, 10], twin_keypoints=[[0, 1], [2, 3]]),
    dict(type='Smart_F1', kpt_indexes=[0,1,2,3], prefix='smart_all', thr=0.2,
         num_steps=101, norm_item=['bbox', 'torso'],
         ignore_gt_out_of_image=True, ignore_gt_out_of_bbox=True,
         torso_keypoint_indexes=[1, 10], twin_keypoints=[[0, 1], [2, 3]],
         generate_chart=True, chart_images_folder=work_dir + '/smart_f1_all_charts'),

    # Classification metrics (unchanged)
    dict(
        type='ClassificationMetric',
        prefix='a',
        classifiers=[
            dict(field_name='gender', num_classes=3),
            dict(field_name='shape', num_classes=2),
        ]
    ),
    dict(
        type='ClassificationMetricConfusionMatrix',
        prefix='b',
        classifiers=[
            dict(field_name='gender', num_classes=3),
            dict(field_name='shape', num_classes=2),
        ],
        generate_chart=True,
        chart_images_folder=work_dir + '/confusion_matrices'
    ),
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
