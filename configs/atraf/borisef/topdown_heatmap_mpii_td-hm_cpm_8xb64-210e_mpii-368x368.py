
MY_BATCH_SIZE = 4

work_dir = '/home/borisef/projects/mm/mmpose/tools/atraf/borisef/work_dirs/cpm1'


auto_scale_lr = dict(base_batch_size=512)
backend_args = dict(backend='local')
codec = dict(
    heatmap_size=(
        46,
        46,
    ),
    input_size=(
        368,
        368,
    ),
    sigma=2,
    type='MSRAHeatmap')
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
data_mode = 'topdown'
my_data_root = '/home/borisef/projects/mm/mmpose/data/mpii/'
dataset_type = 'MpiiDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        interval=10, rule='greater', save_best='PCK', type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        feat_channels=128,
        in_channels=3,
        num_stages=6,
        out_channels=16,
        type='CPM'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            heatmap_size=(
                46,
                46,
            ),
            input_size=(
                368,
                368,
            ),
            sigma=2,
            type='MSRAHeatmap'),
        deconv_out_channels=None,
        final_layer=None,
        in_channels=16,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        num_stages=6,
        out_channels=16,
        type='CPMHead'),
    test_cfg=dict(flip_mode='heatmap', flip_test=True, shift_heatmap=True),
    type='TopdownPoseEstimator')
optim_wrapper = dict(optimizer=dict(lr=0.0005, type='Adam'))
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=210,
        gamma=0.1,
        milestones=[
            170,
            200,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=MY_BATCH_SIZE,
    dataset=dict(
        ann_file='annotations/mpii_val.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root=my_data_root,
        headbox_file= my_data_root+ 'annotations/mpii_gt_val.mat',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                368,
                368,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='MpiiDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='MpiiPCKAccuracy')
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10)
train_dataloader = dict(
    batch_size=MY_BATCH_SIZE,
    dataset=dict(
        ann_file='annotations/mpii_train.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root=my_data_root,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(
                rotate_factor=60,
                scale_factor=(
                    0.75,
                    1.25,
                ),
                shift_prob=0,
                type='RandomBBoxTransform'),
            dict(input_size=(
                368,
                368,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(
                    heatmap_size=(
                        46,
                        46,
                    ),
                    input_size=(
                        368,
                        368,
                    ),
                    sigma=2,
                    type='MSRAHeatmap'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='MpiiDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(
        rotate_factor=60,
        scale_factor=(
            0.75,
            1.25,
        ),
        shift_prob=0,
        type='RandomBBoxTransform'),
    dict(input_size=(
        368,
        368,
    ), type='TopdownAffine'),
    dict(
        encoder=dict(
            heatmap_size=(
                46,
                46,
            ),
            input_size=(
                368,
                368,
            ),
            sigma=2,
            type='MSRAHeatmap'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=MY_BATCH_SIZE,
    dataset=dict(
        ann_file='annotations/mpii_val.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root=my_data_root,
        headbox_file=my_data_root+ 'annotations/mpii_gt_val.mat',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                368,
                368,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='MpiiDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='MpiiPCKAccuracy')
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        368,
        368,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])

