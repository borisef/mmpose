import json
import os

import numpy as np

from mmpose.apis import visualize

from copy import deepcopy
from typing import Union

import mmcv, cv2
import numpy as np
from mmengine.structures import InstanceData

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer

def copy_kpts_from_raw(raw_kpts,keypoints):
    num_kpts = keypoints.shape[1]
    visibility = np.ones((1,num_kpts))
    for i in range(num_kpts):
        keypoints[0,i,0] = raw_kpts[3*i + 0]
        keypoints[0, i, 1] = raw_kpts[3 * i + 1]
        visibility[0,i] = raw_kpts[3 * i + 2]

    return (keypoints, visibility)




#INPUTS
anno_path = "/home/borisef/projects/mm/mmpose/data/somepose/annotations/mmpose_crowdpose_test.json"
images_path = "/home/borisef/projects/mm/mmpose/data/somepose/images"
metainfo = '/home/borisef/projects/mm/mmpose/configs/_base_/datasets/crowdpose.py'
output_folder = "/home/borisef/tmp/out_skeletons"

make_video_fps = 1
video_shape = [800,800]
my_show = False
max_out_images = 100
count = 0


if( not os.path.exists(output_folder)):
    os.makedirs(output_folder)

keypoint_scores = keypoint_score  = None
target_category_id = 1

skeleton_style = 'mmpose'
visualizer = PoseLocalVisualizer()
metainfo = parse_pose_metainfo(dict(from_file=metainfo))
visualizer.set_dataset_meta(metainfo, skeleton_style=skeleton_style)
visualizer.backend = 'matplotlib'



keypoints = np.random.random((1,14,2))*200 #np.ones((1,14,2))


with open(anno_path, 'r') as f:
    data = json.load(f)

images = data['images']
annos = data['annotations']

map_im_ids = {}
for im in images:
    map_im_ids[im['id']] = im['file_name']

if (make_video_fps is not None):
    im = mmcv.imread(os.path.join(images_path, images[0]['file_name']), channel_order='rgb')
    im = mmcv.imresize(im,video_shape)
    out_video = cv2.VideoWriter(os.path.join(output_folder, 'video.avi'), cv2.VideoWriter_fourcc(*'DIVX'),
                                make_video_fps,
                                (im.shape[0], im.shape[1]))

for an in annos:
    if(an['category_id'] == target_category_id):
        raw_kpts = an['keypoints']
        keypoints, visibility = copy_kpts_from_raw(raw_kpts,keypoints)
        keypoint_score = visibility
        img_name = map_im_ids[an['image_id']]
        img_path = os.path.join(images_path,img_name)
        bbox = an['bbox']

#img_path = "/home/borisef/projects/mm/mmpose/data/somepose/images/100000.jpg"
        img = img_path

        if isinstance(img, str):
            img = mmcv.imread(img, channel_order='rgb')
        elif isinstance(img, np.ndarray):
            img = mmcv.bgr2rgb(img)

        if keypoint_score is None:
            keypoint_score = np.ones(keypoints.shape[1])

        tmp_instances = InstanceData()
        tmp_instances.keypoints = keypoints
        tmp_instances.keypoint_score = keypoint_score

        tmp_datasample = PoseDataSample()
        tmp_datasample.pred_instances = tmp_instances



        visualizer.add_datasample(
            'visualization',
            img,
            tmp_datasample,
            show_kpt_idx=False,
            skeleton_style=skeleton_style,
            show=my_show,
            wait_time=0,
            kpt_thr=0.3)

        im_out = visualizer.get_image()
        out_img_path = os.path.join(output_folder, img_name)
        mmcv.imwrite(im_out[:,:,::-1], out_img_path)

        if (make_video_fps is not None):
            im = mmcv.imresize(im_out,video_shape)
            out_video.write(im[:,:,::-1])

        count = count + 1
        print('Count:' + str(count))
        if(count>max_out_images):
            break
# visualize(
#     img_path,
#     keypoints,
#     keypoint_scores,
#     metainfo=metainfo,
#     show=True)

if(make_video_fps is not None):
    out_video.release()
print("OK")