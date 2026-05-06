# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.models.heads.heatmap_heads import *
import copy

#BE
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Tuple

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, OptSampleList, Predictions)


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
            #classifier['fc1'] = nn.Linear(in_channels, 256, device = 'cuda:0')
            classifier['fc1'] = nn.Linear(in_channels, 256)
            self.cl_mod_list.append(classifier['fc1'])
            #classifier['fc2'] = nn.Linear(256, num_classes, device = 'cuda:0')
            classifier['fc2'] = nn.Linear(256, num_classes)
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
                 train_cfg: ConfigType = None) -> dict:

        if train_cfg is None:
            train_cfg = {}
        pose_loss = HeatmapHead.loss(self, feats, batch_data_samples, train_cfg)
        #classification_loss = HeatmapHeadWithClassifiers.loss_only_class(self,feats, batch_data_samples, train_cfg)
        classification_loss = HeatmapHeadWithClassifiers.loss_only_class_weighted(self, feats, batch_data_samples, train_cfg)
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

    def loss_only_class_weighted(self, feats, batch_data_samples, train_cfg=None):
        if train_cfg is None:
            train_cfg = {}
        losses = dict()

        for classifier in self.classifiers:
            classification_probs = self.forward_only_class(feats, classifier)
            classification_labels = []
            field_name = classifier['field_name']
            sample_weights = []  # NEW: collect per-sample weights

            for bds in batch_data_samples:
                temp_dict = bds.to_dict()
                classification_labels.append(
                    torch.tensor(temp_dict['raw_ann_info'][field_name],
                                 device=feats[0].device)
                )
                # NEW: Get per-sample weight for this task
                task_weights = temp_dict['raw_ann_info'].get('task_weights', {})
                sample_weight = task_weights.get(f'{field_name}_weight', 1.0)
                sample_weights.append(sample_weight)

            gt_labels = torch.stack(classification_labels)
            sample_weights = torch.tensor(sample_weights, device=feats[0].device)

            # Apply per-sample weights to loss
            classification_loss = F.cross_entropy(classification_probs, gt_labels,
                                                  reduction='none')
            classification_loss = (classification_loss * sample_weights).mean()

            # Multiply by classifier weight (global weight)
            weight = classifier['weight']
            classifier_name = 'loss_' + field_name
            losses[classifier_name] = classification_loss * weight

        return losses

    def loss_only_class(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = None) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to None

        Returns:
            dict: A dictionary of losses.
        """
        if train_cfg is None:
            train_cfg = {}

        # calculate losses
        losses = dict()
        # loss = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)
        for classifier in self.classifiers:
            classification_probs = HeatmapHeadWithClassifiers.forward_only_class(self,feats, classifier)
            classification_labels = [] #TODO: gender should be packed before hand
            field_name = classifier['field_name']
            for bds in batch_data_samples:
                temp_dict = bds.to_dict()
                classification_labels.append(torch.tensor(temp_dict['raw_ann_info'][field_name], device = feats[0].device))
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

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = None) -> Predictions:
        """Predict results from features, including classifier predictions.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process.
                Defaults to None

        Returns:
            Predictions: The pose predictions from parent class
        """
        if test_cfg is None:
            test_cfg = {}
        # Get predictions from parent HeatmapHead
        preds = HeatmapHead.predict(self, feats, batch_data_samples, test_cfg)

        # Handle flip_test case: feats is a list of feature tuples during TTA
        # Extract the base features for classifier prediction
        if test_cfg.get('flip_test', False) and isinstance(feats, list):
            base_feats = feats[0]
        else:
            base_feats = feats

        # Extract classifier predictions and store them
        for classifier in self.classifiers:
            classification_probs = self.forward_only_class(base_feats, classifier)
            field_name = classifier['field_name']
            labels = classifier.get('labels', None)  # Get optional labels

            # Get predicted class and confidence scores
            pred_classes = torch.argmax(classification_probs, dim=1)
            pred_scores = torch.max(classification_probs, dim=1)[0]

            # Store predictions in each data sample
            for idx, data_sample in enumerate(batch_data_samples):
                # Store classifier predictions in the data sample
                if not hasattr(data_sample, 'pred_classifiers'):
                    data_sample.pred_classifiers = {}

                pred_class_idx = pred_classes[idx].item()
                pred_label = labels[pred_class_idx] if labels else None

                data_sample.pred_classifiers[field_name] = {
                    'pred_class': pred_class_idx,
                    'pred_label': pred_label,
                    'pred_score': pred_scores[idx].item(),
                    'all_probs': classification_probs[idx].detach().cpu().numpy()
                }

                # Also try to get ground truth if available
                if hasattr(data_sample, 'raw_ann_info') and field_name in data_sample.raw_ann_info:
                    gt_class_idx = data_sample.raw_ann_info[field_name]
                    data_sample.pred_classifiers[field_name]['gt_class'] = gt_class_idx
                    # Add ground truth label if available
                    if labels:
                        data_sample.pred_classifiers[field_name]['gt_label'] = labels[gt_class_idx]

        return preds




