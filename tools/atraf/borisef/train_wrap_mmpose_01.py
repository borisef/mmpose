import copy
import os.path
import os.path as osp
import sys

import mmcv
import numpy as np
import mmengine
from mmengine.config import Config

import cv2
import mmcv
from mmcv.transforms import Compose

import mmpose
import tools.train as train


#config_file = "/home/borisef/projects/mm/mmpose/configs/body_2d_keypoint/rtmpose/crowdpose/rtmpose-m_8xb64-210e_crowdpose-256x192.py"
config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/crowdpose_rtmpose-m_8xb64-210e_crowdpose-256x192.py" #WORKS
config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/topdown_heatmap_mpii_td-hm_cpm_8xb64-210e_mpii-368x368.py"#WORKS
sys.argv.append(config_file)

train.main()
