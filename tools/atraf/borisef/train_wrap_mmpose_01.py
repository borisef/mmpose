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
#config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/crowdpose_rtmpose-m_8xb64-210e_crowdpose-256x192.py" #WORKS
#config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/crowdpose_rtmpose-m_8xb64-210e_crowdpose-256x192_try.py" #WORKS
#config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/topdown_heatmap_mpii_td-hm_cpm_8xb64-210e_mpii-368x368.py"#WORKS

#config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/td-hm_ViTPose-small_8xb64-210e_coco-256x192_coco_vit_small.py"   # works slow
#config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/td-hm_hrnet-w32_dark-8xb64-210e_coco-256x192_try.py" #works very slow
config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288_try.py" # works

#config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/td-hm_3xrsn50_8xb32-210e_coco-256x192_try.py" # NOT GOOD , ???
#config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/td-hm_rsn50_8xb32-210e_coco-256x192_try.py" #NOT GOOD, bug
#config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/simcc_res50_8xb32-140e_coco-384x288_try.py" # works
#config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/rtmo-s_8xb32-600e_coco-640x640_try.py" #NOT GOOD runs generates crap
#config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/edpose_res50_8xb2-50e_coco-800x1333_trydump.py" #NOT GOOD

#config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/td-hm_litehrnet-30_8xb32-210e_coco-384x288_clean.py"  # works + data in base
#config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/yoloxpose_s_8xb32-300e_coco-640_try.py"# works very slow convergence
#config_file = "/home/borisef/projects/mm/mmpose/configs/atraf/borisef/cid_hrnet-w32_8xb20-140e_coco-512x512_try.py"# NOT GOOD
sys.argv.append(config_file)

train.main()
