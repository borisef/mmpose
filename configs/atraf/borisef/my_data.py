data_root = '/home/borisef/data/coco/'
dataset_type = 'CocoDataset'


train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='images/val2017/')
    ))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='images/val2017/')
    ))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')