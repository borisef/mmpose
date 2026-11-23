_base_ = ['./td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288_try_changes.py']

# --- weights from the pose checkpoint; fresh optimizer, training restarts at epoch 0 ---
load_from = ('/home/borisef/projects/mm/mmpose/tools/atraf/borisef/'
             'work_dirs/hrnet_UDP_w32_try_changes/epoch_10.pth')
resume = False

# --- work dir ---
work_dir = ('/home/borisef/projects/mm/mmpose/tools/atraf/borisef/'
            'work_dirs/hrnet_UDP_w32_try_changes_freeze/')

train_cfg = dict(max_epochs=20, val_interval=1)

# --- freeze keypoint detector from epoch 0, train only classifiers ---
custom_hooks = [
    dict(type='ClassifierLRSchedulerHook'),  # must stay first
    dict(type='KeypointFreezeHook', freeze_epoch=0),
]

model = dict(
    # avoid downloading ImageNet weights; load_from overrides them anyway
    backbone=dict(init_cfg=None),
    head=dict(
        # lists are replaced (not merged) by _base_ inheritance, so restate
        classifiers=[
            dict(num_classes=3, weight=0.5, field_name='gender',
                 labels=['M', 'W', 'S'],
                 loss_cfg=dict(type='CrossEntropyLoss', reduction='none'),
                 num_convs=2, num_fcs=2,
                 conv_out_channels=256, fc_out_channels=256,
                 lr_schedule=[(0, 0.001)]),
            dict(num_classes=2, weight=0.5, field_name='shape',
                 labels=['round', 'rectangular'],
                 loss_cfg=dict(type='FocalLoss', gamma=2.0, reduction='none'),
                 num_convs=1, num_fcs=1,
                 conv_out_channels=128, fc_out_channels=64,
                 lr_schedule=[(0, 0.001)]),
        ]))

# Paths built from work_dir in the base config were fixed at parse time,
# so re-point them here.
default_hooks = dict(
    visualization=dict(out_dir=work_dir + '/vvv'),
    # pose metrics can't improve while frozen -> select on classifier metrics
    checkpoint=dict(
        save_best=['cls/classifier/gender/accuracy',
                   'cls/classifier/shape/accuracy'],
        rule=['greater', 'greater']))

val_evaluator = [
    dict(type='ClassificationMetric', prefix='cls',
         classifiers=[dict(field_name='gender', num_classes=3),
                      dict(field_name='shape', num_classes=2)]),
    dict(type='ClassificationMetricConfusionMatrix', prefix='cm',
         classifiers=[
             dict(field_name='gender', num_classes=3,
                  class_names=['male', 'female', 'unknown']),
             dict(field_name='shape', num_classes=2,
                  class_names=['slim', 'wide'])],
         generate_chart=True, use_ratios=True,
         chart_images_folder=work_dir + '/confusion_matrices'),
    # sanity check: keypoint quality should stay constant (frozen)
    dict(type='AtrafPCKAccuracy', kpt_indexes=[0, 1, 2, 3],
         prefix='pck_no_oob', thr=0.2,
         ignore_gt_out_of_image=True, ignore_gt_out_of_bbox=True),
]
test_evaluator = val_evaluator

