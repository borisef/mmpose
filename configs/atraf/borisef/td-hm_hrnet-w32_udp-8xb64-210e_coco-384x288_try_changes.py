_base_ = ['/home/borisef/projects/mm/mmpose/configs/_base_/default_runtime.py']

MY_BATCH = 1
work_dir =  '/home/borisef/projects/mm/mmpose/tools/atraf/borisef/work_dirs/hrnet_UDP_w32_try_changes'
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
        # SmartF2 was removed; F2 (now on Recall_Atraf) is its replacement.
        # SmartF1 is also emitted by Recall_Atraf now. Keys must match a
        # metric+prefix present in val_evaluator below (here: 'recall_oob').
        save_best=['recall_oob/F2', 'recall_oob/SmartF1', 'pck_no_oob/PCK'],
        rule=['greater', 'greater', 'greater'],
        max_keep_ckpts=10),
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

    # #####################################################################
    # #####################################################################
    # GROUP 1 - Recall_Atraf
    #   Outputs (per norm suffix ''/h/t): Recall, Precision, F1, F2,
    #                                     SmartF1, SmartThreshold
    #   All options:
    #     thr                    (float)  PCK correctness threshold
    #     score_threshold        (float)  "high-score" cutoff for Recall/F1
    #     num_steps              (int)    thresholds sampled in [0,1] for SmartF1
    #     norm_item              str | list of {'bbox','head','torso'}
    #     kpt_indexes            (list)   evaluate only these keypoints
    #     ignore_gt_out_of_image (bool)   mask GT outside the image
    #     ignore_gt_out_of_bbox  (bool)   mask GT outside the bbox
    #     torso_keypoint_indexes (list)   pair used when norm_item='torso'
    #     twin_keypoints         (list)   symmetric pairs, min-distance match
    #     generate_chart         (bool)   save a Precision-Recall PNG
    #     chart_images_folder    (str)    where to save charts
    #     collect_device         (str)    'cpu' or 'gpu'
    #     prefix                 (str)    key namespace
    # #####################################################################

    # 1a. Minimal: all defaults (norm bbox, thr .05, score_thr .5, num_steps 101)
    dict(type='Recall_Atraf', prefix='recall_default'),

    # 1b. Filter to a subset of keypoints only
    dict(type='Recall_Atraf', prefix='recall_kpts', thr=0.2,
         kpt_indexes=[0, 1, 2, 3]),

    # 1c. Custom PCK threshold + score threshold + finer/coarser sweep
    dict(type='Recall_Atraf', prefix='recall_thr', thr=0.1,
         kpt_indexes=[0, 1, 2, 3], score_threshold=0.3, num_steps=201),

    # 1d. ignore_gt_out_of_image only
    dict(type='Recall_Atraf', prefix='recall_ooi', thr=0.2,
         kpt_indexes=[0, 1, 2, 3], ignore_gt_out_of_image=True),

    # 1e. ignore_gt_out_of_bbox only
    dict(type='Recall_Atraf', prefix='recall_oob', thr=0.2,
         kpt_indexes=[0, 1, 2, 3], ignore_gt_out_of_bbox=True),

    # 1f. Both spatial masks together
    dict(type='Recall_Atraf', prefix='recall_ooi_oob', thr=0.2,
         kpt_indexes=[0, 1, 2, 3],
         ignore_gt_out_of_image=True, ignore_gt_out_of_bbox=True),

    # 1g. Torso normalization with a custom keypoint pair (default pair is [4,5])
    #     (norm_item='head' also exists but needs 'head_size' in the annotations)
    dict(type='Recall_Atraf', prefix='recall_torso', thr=0.2,
         kpt_indexes=[0, 1, 2, 3], norm_item='torso',
         torso_keypoint_indexes=[1, 10]),

    # 1h. Multiple norms at once -> emits '' (bbox) and 't' (torso) keys
    dict(type='Recall_Atraf', prefix='recall_multi', thr=0.2,
         kpt_indexes=[0, 1, 2, 3], norm_item=['bbox', 'torso'],
         torso_keypoint_indexes=[1, 10]),

    # 1i. twin_keypoints: symmetric pairs matched by minimum distance
    dict(type='Recall_Atraf', prefix='recall_twin', thr=0.2,
         kpt_indexes=[0, 1, 2, 3], twin_keypoints=[[0, 1], [2, 3]]),

    # 1j. Chart generation (Precision-Recall + best-F1 marker) to disk/TensorBoard
    dict(type='Recall_Atraf', prefix='recall_chart', thr=0.2,
         kpt_indexes=[0, 1, 2, 3],
         generate_chart=True, chart_images_folder=work_dir + '/recall_charts'),

    # 1k. Everything together
    dict(type='Recall_Atraf', prefix='recall_all', thr=0.2,
         kpt_indexes=[0, 1, 2, 3], score_threshold=0.5, num_steps=101,
         norm_item=['bbox', 'torso'], torso_keypoint_indexes=[1, 10],
         twin_keypoints=[[0, 1], [2, 3]],
         ignore_gt_out_of_image=True, ignore_gt_out_of_bbox=True,
         collect_device='cpu',
         generate_chart=True, chart_images_folder=work_dir + '/recall_all_charts'),

    # #####################################################################
    # GROUP 2 - Success_Rate
    #   Outputs (per norm suffix ''/h/t): PD_Success_Rate, FAR_Error_Rate,
    #     Skip_Rate, Combined_Success_Rate, Best_Combined_Success_Rate,
    #     BestThreshold
    #   Extra options on top of the GROUP 1 options:
    #     weight_FAR  (float) weight of the (1 - FAR_Error_Rate) term
    #     weight_skip (float) weight of the (1 - Skip_Rate) term
    #     weight_PD   (float) weight of the PD_Success_Rate term
    #   Combined = [w_FAR*(1-FAR) + w_skip*(1-Skip) + w_PD*PD] / (w_FAR+w_skip+w_PD)
    # #####################################################################

    # 2a. Minimal: all defaults, equal weights (1/1/1)
    dict(type='Success_Rate', prefix='success_default'),

    # 2b. Keypoint subset + custom thresholds/sweep
    dict(type='Success_Rate', prefix='success_kpts', thr=0.2,
         kpt_indexes=[0, 1, 2, 3], score_threshold=0.3, num_steps=201),

    # 2c. FAR-dominant weighting (penalize false alarms hardest)
    dict(type='Success_Rate', prefix='success_far_heavy', thr=0.2,
         kpt_indexes=[0, 1, 2, 3],
         weight_FAR=3.0, weight_skip=1.0, weight_PD=1.0),

    # 2d. Skip-dominant weighting (penalize skipped/low-score hardest)
    dict(type='Success_Rate', prefix='success_skip_heavy', thr=0.2,
         kpt_indexes=[0, 1, 2, 3],
         weight_FAR=1.0, weight_skip=3.0, weight_PD=1.0),

    # 2e. PD-dominant weighting (reward correct detections hardest)
    dict(type='Success_Rate', prefix='success_pd_heavy', thr=0.2,
         kpt_indexes=[0, 1, 2, 3],
         weight_FAR=1.0, weight_skip=1.0, weight_PD=3.0),

    # 2f. Single-term weighting (zero out two terms -> Combined = PD only)
    dict(type='Success_Rate', prefix='success_pd_only', thr=0.2,
         kpt_indexes=[0, 1, 2, 3],
         weight_FAR=0.0, weight_skip=0.0, weight_PD=1.0),

    # 2g. Spatial masks
    dict(type='Success_Rate', prefix='success_ooi_oob', thr=0.2,
         kpt_indexes=[0, 1, 2, 3],
         ignore_gt_out_of_image=True, ignore_gt_out_of_bbox=True),

    # 2h. Torso normalization
    dict(type='Success_Rate', prefix='success_torso', thr=0.2,
         kpt_indexes=[0, 1, 2, 3], norm_item='torso',
         torso_keypoint_indexes=[1, 10]),

    # 2i. Chart generation (Combined-vs-threshold curve with best marker)
    dict(type='Success_Rate', prefix='success_chart', thr=0.2,
         kpt_indexes=[0, 1, 2, 3],
         generate_chart=True, chart_images_folder=work_dir + '/success_charts'),

    # 2j. Everything together
    dict(type='Success_Rate', prefix='success_all', thr=0.2,
         kpt_indexes=[0, 1, 2, 3], score_threshold=0.5, num_steps=101,
         norm_item=['bbox', 'torso'], torso_keypoint_indexes=[1, 10],
         twin_keypoints=[[0, 1], [2, 3]],
         weight_FAR=2.0, weight_skip=1.0, weight_PD=1.5,
         ignore_gt_out_of_image=True, ignore_gt_out_of_bbox=True,
         collect_device='cpu',
         generate_chart=True, chart_images_folder=work_dir + '/success_all_charts'),

    # #####################################################################
    # Reference PCK/AUC/EPE with the same torso setup (for comparison)
    # #####################################################################
    dict(type='AtrafPCKAccuracy', kpt_indexes=[0, 1, 2, 3], prefix='pck_torso',
         thr=0.2, norm_item='torso', torso_keypoint_indexes=[1, 10]),

    # #####################################################################
    # GROUP 3 - Classification metrics
    # #####################################################################
    dict(
        type='ClassificationMetric',
        prefix='a',
        classifiers=[
            dict(field_name='gender', num_classes=3),
            dict(field_name='shape', num_classes=2),
        ]
    ),

    # 3a. Confusion matrix as ROW-NORMALIZED RATIOS + class names on axes
    dict(
        type='ClassificationMetricConfusionMatrix',
        prefix='b_ratios',
        classifiers=[
            dict(field_name='gender', num_classes=3,
                 class_names=['male', 'female', 'unknown']),
            dict(field_name='shape', num_classes=2,
                 class_names=['slim', 'wide']),
        ],
        generate_chart=True,
        use_ratios=True,
        chart_images_folder=work_dir + '/confusion_matrices_ratios'
    ),

    # 3b. Confusion matrix as RAW COUNTS (use_ratios=False) with class names
    dict(
        type='ClassificationMetricConfusionMatrix',
        prefix='b_counts',
        classifiers=[
            dict(field_name='gender', num_classes=3,
                 class_names=['male', 'female', 'unknown']),
            dict(field_name='shape', num_classes=2,
                 class_names=['slim', 'wide']),
        ],
        generate_chart=True,
        use_ratios=False,
        chart_images_folder=work_dir + '/confusion_matrices_counts'
    ),

    # 3c. No class_names -> axes fall back to numeric class indices
    dict(
        type='ClassificationMetricConfusionMatrix',
        prefix='b_numeric',
        classifiers=[
            dict(field_name='gender', num_classes=3),
            dict(field_name='shape', num_classes=2),
        ],
        generate_chart=True,
        use_ratios=True,
        chart_images_folder=work_dir + '/confusion_matrices_numeric'
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
