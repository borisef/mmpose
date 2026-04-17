# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.models.heads.heatmap_heads import *
import copy

#BE
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple, Union

from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)


@MODELS.register_module()
class HeatmapHeadWithClassifiers(HeatmapHead):
    def __init__(self,
                 in_channels,
                 out_channels,
                 classifiers,
                 **kwargs):
        self.num_classifiers = len(classifiers),
        self.classifiers = copy.deepcopy(classifiers)

        super().__init__(in_channels, out_channels, **kwargs)
        self.cl_mod_list = nn.ModuleList()
        for classifier in self.classifiers:
        # Classification head
            num_classes = classifier['num_classes']
            classifier['avg_pool'] = nn.AdaptiveAvgPool2d((1, 1))
            self.cl_mod_list.append(classifier['avg_pool'])
            classifier['fc1'] = nn.Linear(in_channels, 256, device = 'cuda:0')
            self.cl_mod_list.append(classifier['fc1'])
            classifier['fc2'] = nn.Linear(256, num_classes, device = 'cuda:0')
            self.cl_mod_list.append(classifier['fc2'])

    def forward(self, x):
        """Forward function."""
        heatmaps = super().forward(x)  # Pose estimation heatmaps

        # Classification branch
        classification_features = x[0]
        #classification_features = self.avg_pool(x[0])  # Assuming HRNet output is a list
        classification_features = torch.flatten(classification_features, 1)
        classification_features = F.relu(self.fc1(classification_features))
        classification_logits = self.fc2(classification_features)
        classification_probs = F.softmax(classification_logits, dim=1)

        return heatmaps, classification_probs

    #def loss(self, heatmaps, heatmaps_targets, mask, classification_probs, classification_labels):
    def loss(self,
                 feats: Tuple[Tensor],
                 batch_data_samples: OptSampleList,
                 train_cfg: ConfigType = {}) -> dict:

        pose_loss = HeatmapHead.loss(self, feats, batch_data_samples, train_cfg)
        classification_loss = HeatmapHeadWithClassifiers.loss_only_class(self,feats, batch_data_samples, train_cfg)
        # #F.cross_entropy(classification_probs, classification_labels)

        losses = pose_loss
        #losses['classification_loss'] = classification_loss
        losses.update(classification_loss)

        return losses

    def forward_only_class(self, x, classifier_dict):
        """Forward function."""
        # heatmaps = super().forward(x)  # Pose estimation heatmaps

        # Classification branch
        # classification_features = self.avg_pool(x[0])  # Assuming HRNet output is a list
        # classification_features = torch.flatten(classification_features, 1)
        # classification_features = F.relu(self.fc1(classification_features))
        # classification_logits = self.fc2(classification_features)
        # classification_probs = F.softmax(classification_logits, dim=1)

        classification_features = classifier_dict['avg_pool'](x[0])  # Assuming HRNet output is a list
        classification_features = torch.flatten(classification_features, 1)
        classification_features = F.relu(classifier_dict['fc1'](classification_features))
        classification_logits = classifier_dict['fc2'](classification_features)
        classification_probs = F.softmax(classification_logits, dim=1)

        return classification_probs

    def loss_only_class(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """

        # calculate losses
        losses = dict()
        # loss = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)
        for classifier in self.classifiers:
            classification_probs = HeatmapHeadWithClassifiers.forward_only_class(self,feats, classifier)
            classification_labels = [] #TODO: gender should be packed before hand
            field_name = classifier['field_name']
            for bds in batch_data_samples:
                temp_dict = bds.to_dict()
                classification_labels.append(torch.tensor(temp_dict['raw_ann_info'][field_name], device='cuda:0')) #TODO: device from features
            gt_labels = torch.stack(classification_labels)
            classification_loss = F.cross_entropy(classification_probs, gt_labels)
            weight = classifier['weight']
            classifier_name = 'loss_' + field_name
            class_dict = {classifier_name: classification_loss * weight}
            losses.update(class_dict)

            # # calculate accuracy
            # if train_cfg.get('compute_acc', True):
            #     avg_acc = atraf_classification_accuracy(
            #         pred = to_numpy(classification_probs),
            #         gt=to_numpy(gt_labels),
            #         mask=to_numpy(1),
            #         thr=0.1)
            #
            #     acc_class = torch.tensor(avg_acc, device="cuda:0")
            #     acc_str = "acc_" + field_name
            #     losses.update({acc_str: acc_class})



        return losses