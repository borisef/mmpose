_base_ = ['/home/borisef/projects/mm/mmpose/configs/_base_/default_runtime.py']

MY_BATCH = 1
work_dir =  '/home/borisef/projects/mm/mmpose/tools/atraf/borisef/work_dirs/hrnet_UDP_w32_like_try_rfdetr_large_with_aug'
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
# ============================================================================
# Augmentation pipeline mirroring rf-detr's
#   atraf/configs/train_config_large_with_aug.py  (AUG_CONFIG)
# driven through AtrafAlbumentationsWrapper. Structure (9 stages, each
# SomeOf(n=1, p=pX) unless noted) and probabilities match rf-detr.
#
# ADAPTED FOR THIS ENVIRONMENT (albumentations 1.4.18, not 2.0.8):
#   * GEOMETRIC transforms are COMMENTED OUT. The wrapper is image-only
#     (keypoints/bboxes are not transformed), so D4 / Affine / Perspective /
#     GridElasticDeform / RandomSizedBBoxSafeCrop would desync keypoint labels
#     (and RandomSizedBBoxSafeCrop needs bboxes). Kept here for reference.
#   * MISSING in album 1.4.18 (commented out): ConstrainedCoarseDropout
#     (-> replaced by plain CoarseDropout), AutoContrast,
#     PlasmaBrightnessContrast, ShotNoise, AdditiveNoise.
#   * AtrafStyleTransfer is COMMENTED OUT: its FDA / HistogramMatching /
#     PixelDistributionAdaptation backends use the album 2.0.8 API and raise on
#     1.4.18. Re-enable only on a Python 3.9+ / albumentations 2.0.8 env.
#   * Param renames vs 2.0.8: Affine/Perspective fill->cval/pad_val,
#     MotionBlur drops angle_range/direction_range, GaussNoise
#     std_range->var_limit (var = (std*255)^2, mean=0).
#
# Must run before TopdownAffine so a sibling mask/<filename> (if present) still
# matches the original-resolution image. Mask-dependent augs
# (AtrafBurningSpot*, AtrafRandomEllipseSpot) are no-ops when no mask exists.
# ============================================================================

# Stage apply-probabilities (tune without touching structure) — from rf-detr.
_pStyle = 0.25   # style transfer (disabled below: needs album 2.0.8)
_pSafe = 0.5     # safe pixel transform (D4 is geometric -> disabled)
_p1 = 0.7        # OneOfSet1: appearance distortions
_p2 = 0.5        # OneOfSet2: optical / sensor distortions
_p3 = 0.4        # OneOfSet3: color / weather effects
_pNsmall = 0.4   # small noise  (SNR ~ 5-25)
_pNlarge = 0.2   # heavy noise  (SNR ~ 0.2-4)

# Populate and point at a folder of reference images to use style transfer
# (only on an album 2.0.8 env).
STYLE_REFERENCE_DIR = '/home/borisef/data/large_rf_detr_data/style_reference_images'

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='AtrafAlbumentationsWrapper',
        # rfdetr_path='/home/borisef/projects/rf-detr',  # or RFDETR_ATRAF_PATH env
        transforms=[

            # -- Stage 1: Style transfer (DISABLED: album 2.0.8 only) ---------
            # dict(type='SomeOf', transforms=[
            #     dict(type='AtrafStyleTransfer',
            #          reference_images_dir=STYLE_REFERENCE_DIR,
            #          methods=['FDA', 'HistogramMatching',
            #                   'PixelDistributionAdaptation'],
            #          p=1.0),
            # ], n=1, p=_pStyle),

            # -- Stage 2: Safe transforms (D4 is geometric -> disabled) -------
            dict(type='SomeOf', transforms=[
                # dict(type='D4', p=1.0),  # geometric: desyncs keypoints
                dict(type='InvertImg', p=1.0),
            ], n=1, p=_pSafe),

            # -- Stage 3: OneOfSet1 — appearance distortions ------------------
            dict(type='SomeOf', transforms=[
                dict(type='AdvancedBlur', blur_limit=[3, 13],
                     sigma_x_limit=[0.2, 1.0], sigma_y_limit=[0.2, 1.0],
                     rotate_limit=[-90, 90], beta_limit=[0.5, 8.0],
                     noise_limit=[0.9, 1.1], p=1.0),
                dict(type='MotionBlur', blur_limit=[3, 17],
                     allow_shifted=False, p=1.0),
                dict(type='Spatter', mean=[0.4, 0.85], std=[0.1, 0.4],
                     gauss_sigma=[1.0, 2.0], cutout_threshold=[0.6, 0.68],
                     intensity=[0.2, 0.7], mode='mud', p=1.0),
                # ConstrainedCoarseDropout missing in 1.4.18 -> plain CoarseDropout
                dict(type='CoarseDropout', num_holes_range=[3, 15],
                     hole_height_range=[0.05, 0.1],
                     hole_width_range=[0.05, 0.1], fill_value=0, p=1.0),
                # dict(type='Affine', ...),           # geometric: disabled
                # dict(type='GridElasticDeform', ...), # geometric: disabled
                # dict(type='Perspective', ...),       # geometric: disabled
                dict(type='AtrafRandomEllipseSpot', p=1.0),
                dict(type='AtrafLinearBrightnessGradient',
                     strength_range=[30.0, 80.0], angle_range=[0.0, 360.0],
                     p=1.0),
                dict(type='AtrafBurningSpotWithTrail', p=1.0),
            ], n=1, p=_p1),

            # -- Stage 4: OneOfSet2 — optical / sensor distortions ------------
            dict(type='SomeOf', transforms=[
                dict(type='ChromaticAberration',
                     primary_distortion_limit=[-0.5, 0.5],
                     secondary_distortion_limit=[-0.5, 0.5],
                     mode='random', interpolation=1, p=1.0),
                dict(type='Defocus', radius=[1, 5], alias_blur=[0.1, 0.5],
                     p=1.0),
                dict(type='MedianBlur', blur_limit=[3, 13], p=1.0),
                dict(type='PixelDropout', dropout_prob=0.2, per_channel=False,
                     drop_value=0, p=1.0),
                dict(type='RandomGamma', gamma_limit=[20, 180], p=1.0),
                dict(type='RandomShadow', shadow_roi=[0.0, 0.0, 1.0, 1.0],
                     num_shadows_limit=[1, 1], shadow_dimension=3,
                     shadow_intensity_range=[0.1, 0.8], p=1.0),
                # RandomSizedBBoxSafeCrop: geometric + needs bboxes -> disabled
                # dict(type='RandomSizedBBoxSafeCrop', height=384, width=384,
                #      erosion_rate=0.0, interpolation=1, p=1.0),
            ], n=1, p=_p2),

            # -- Stage 5: OneOfSet3 — color / weather effects -----------------
            dict(type='SomeOf', transforms=[
                # dict(type='AutoContrast', p=1.0),            # missing in 1.4.18
                # dict(type='PlasmaBrightnessContrast', ...),  # missing in 1.4.18
                dict(type='CLAHE', clip_limit=[1.0, 7.0],
                     tile_grid_size=[8, 8], p=1.0),
                dict(type='Equalize', mode='cv', by_channels=True, p=1.0),
                dict(type='FancyPCA', alpha=13.0, p=1.0),
                dict(type='RandomFog', alpha_coef=0.05,
                     fog_coef_range=[0.5, 0.9], p=1.0),
                dict(type='RandomSunFlare', flare_roi=[0.0, 0.0, 1.0, 0.5],
                     src_radius=100, src_color=[255, 255, 255],
                     angle_range=[0.0, 1.0], num_flare_circles_range=[2, 2],
                     method='physics_based', p=1.0),
            ], n=1, p=_p3),

            # -- Stage 6: Small noise (SNR ~ 5-25) ----------------------------
            # std_range [0.02,0.10] -> var_limit [(0.02*255)^2,(0.10*255)^2]
            dict(type='SomeOf', transforms=[
                dict(type='GaussNoise', var_limit=[26.0, 650.0], mean=0,
                     per_channel=False, p=1.0),
                # ShotNoise missing in 1.4.18
                dict(type='MultiplicativeNoise', multiplier=[0.90, 1.10],
                     per_channel=False, elementwise=True, p=1.0),
                # AdditiveNoise missing in 1.4.18
            ], n=1, p=_pNsmall),

            # -- Stage 7: Heavy noise (SNR ~ 0.2-4) ---------------------------
            # std_range [0.125,0.70] -> var_limit [(0.125*255)^2,(0.70*255)^2]
            dict(type='SomeOf', transforms=[
                dict(type='GaussNoise', var_limit=[1016.0, 31862.0], mean=0,
                     per_channel=False, p=1.0),
                # ShotNoise missing in 1.4.18
                dict(type='MultiplicativeNoise', multiplier=[0.50, 1.50],
                     per_channel=False, elementwise=True, p=1.0),
                # AdditiveNoise missing in 1.4.18
            ], n=1, p=_pNlarge),

            # -- Stage 8: Grayscale (keep 3 channels, always) -----------------
            dict(type='ToGray', num_output_channels=3,
                 method='weighted_average', p=1.0),

            # -- Stage 9: Final exposure stretch (always) ---------------------
            dict(type='AtrafClipDynamicRange',
                 low_percentile_range=[0.0, 0.05],
                 high_percentile_range=[0.92, 0.999], p=1.0),
        ]),
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
