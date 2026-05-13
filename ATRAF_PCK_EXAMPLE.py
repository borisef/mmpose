# Config Example for AtrafPCKAccuracy

# Original config
val_evaluator_before = [
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

# Updated config with keypoint filtering
val_evaluator_after = [
    dict(type='PCKAccuracy', thr=0.2),  # All keypoints
    dict(type='AUC'),
    dict(type='EPE'),
    dict(type='CocoMetric'),

    # NEW: Evaluate specific keypoint groups
    # COCO keypoints: 0-4=head, 5-10=upper, 11-16=lower
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        kpt_indexes=[0, 1, 2, 3, 4],
        prefix='head'
    ),
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        kpt_indexes=[5, 6, 7, 8, 9, 10],
        prefix='upper'
    ),
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        kpt_indexes=[11, 12, 13, 14, 15, 16],
        prefix='lower'
    ),

    dict(
        type='ClassificationMetric',
        classifiers=[
            dict(field_name='gender', num_classes=3),
            dict(field_name='shape', num_classes=2),
        ]
    )
]

