# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.models.heads.heatmap_heads import *
import copy

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Tuple, List

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, OptSampleList, Predictions)


class ClassifierHead(nn.Module):
    """A flexible per-task classification head inspired by MMDetection's
    ConvFCBBoxHead pattern.

    Architecture (per head):

        feature map
            -> [num_convs x (Conv2d + BN + ReLU)]   # optional conv tower
            -> AdaptiveAvgPool2d(1,1)                # always present
            -> flatten
            -> [num_fcs x (Linear + ReLU)]           # configurable depth
            -> Linear(fc_out_channels, num_classes)  # output layer
            -> Softmax

    Args:
        in_channels (int): Input feature channels (from backbone/neck).
        num_classes (int): Number of output classes for this task.
        field_name (str): Key used to pull GT labels from raw_ann_info.
        weight (float): Loss weight scalar for this task. Default: 1.0.
        num_convs (int): Number of conv layers before pooling. Default: 0.
        num_fcs (int): Number of hidden FC layers (before the output FC).
                       Must be >= 1. Default: 1.
        conv_out_channels (int): Channel width of each conv layer. Default: 256.
        fc_out_channels (int): Width of each hidden FC layer. Default: 256.
        labels (list[str], optional): Human-readable label names for each
                                      class index. Used at inference time.
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 field_name: str,
                 weight: float = 1.0,
                 num_convs: int = 0,
                 num_fcs: int = 1,
                 conv_out_channels: int = 256,
                 fc_out_channels: int = 256,
                 labels: List[str] = None):
        super().__init__()
        assert num_fcs >= 1, 'num_fcs must be >= 1 (need at least one hidden FC layer)'

        self.field_name = field_name
        self.weight = weight
        self.labels = labels
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels

        # ── Conv tower ────────────────────────────────────────────────────
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else conv_out_channels
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, conv_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(conv_out_channels),
                nn.ReLU(inplace=True),
            ))

        # ── Spatial pooling ───────────────────────────────────────────────
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── FC tower ──────────────────────────────────────────────────────
        # First FC: from (conv_out_channels or in_channels) → fc_out_channels
        fc_in_channels = conv_out_channels if num_convs > 0 else in_channels
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_ch = fc_in_channels if i == 0 else fc_out_channels
            self.fcs.append(nn.Linear(in_ch, fc_out_channels))

        # ── Output layer ──────────────────────────────────────────────────
        self.fc_out = nn.Linear(fc_out_channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for conv_block in self.convs:
            nn.init.kaiming_normal_(conv_block[0].weight,
                                    mode='fan_out', nonlinearity='relu')
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.constant_(fc.bias, 0)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Feature map of shape (N, C, H, W).

        Returns:
            Tensor: Class probabilities of shape (N, num_classes).
        """
        # Conv tower
        for conv in self.convs:
            x = conv(x)

        # Pool + flatten
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        # FC tower
        for fc in self.fcs:
            x = F.relu(fc(x))

        # Output
        return F.softmax(self.fc_out(x), dim=1)


@MODELS.register_module()
class HeatmapHeadWithClassifiers(HeatmapHead):
    """HeatmapHead extended with flexible per-task ClassifierHead branches.

    Each entry in `classifiers` is a dict that maps directly to the
    ClassifierHead constructor kwargs, e.g.:

    .. code-block:: python

        classifiers=[
            dict(
                field_name='gender',
                num_classes=2,
                weight=1.0,
                num_convs=0,
                num_fcs=1,
                fc_out_channels=256,
                labels=['male', 'female'],
            ),
            dict(
                field_name='age_group',
                num_classes=4,
                weight=0.5,
                num_convs=2,        # extra conv tower for this task
                num_fcs=2,          # deeper FC tower
                conv_out_channels=128,
                fc_out_channels=512,
                labels=['child', 'young', 'adult', 'senior'],
            ),
        ]

    Args:
        in_channels (int): Input channels forwarded to HeatmapHead.
        out_channels (int): Output channels forwarded to HeatmapHead.
        classifiers (list[dict]): List of classifier head configs.
        **kwargs: Remaining kwargs passed to HeatmapHead.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 classifiers: List[dict],
                 **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

        # Build one ClassifierHead per task
        self.classifier_heads = nn.ModuleList()
        for cfg in classifiers:
            cfg = copy.deepcopy(cfg)
            self.classifier_heads.append(
                ClassifierHead(in_channels=in_channels, **cfg)
            )

    # ── Forward helpers ───────────────────────────────────────────────────

    def _forward_classifier(self, head: ClassifierHead, feats: Tuple[Tensor]) -> Tensor:
        """Run a single ClassifierHead on the first feature level."""
        return head(feats[0])

    # ── Training ──────────────────────────────────────────────────────────

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = None) -> dict:

        if train_cfg is None:
            train_cfg = {}

        losses = HeatmapHead.loss(self, feats, batch_data_samples, train_cfg)
        losses.update(self._classifier_losses(feats, batch_data_samples))
        return losses

    def _classifier_losses(self,
                           feats: Tuple[Tensor],
                           batch_data_samples: OptSampleList) -> dict:
        losses = {}
        for head in self.classifier_heads:
            probs = self._forward_classifier(head, feats)
            gt_labels, sample_weights = self._extract_labels_and_weights(
                head.field_name, batch_data_samples, feats[0].device)

            per_sample_loss = F.cross_entropy(probs, gt_labels, reduction='none')
            weighted_loss = (per_sample_loss * sample_weights).mean()

            losses[f'loss_{head.field_name}'] = weighted_loss * head.weight
        return losses

    @staticmethod
    def _extract_labels_and_weights(field_name: str,
                                    batch_data_samples: OptSampleList,
                                    device) -> Tuple[Tensor, Tensor]:
        """Pull GT labels and optional per-sample weights from data samples."""
        labels, weights = [], []
        for bds in batch_data_samples:
            ann = bds.to_dict()['raw_ann_info']
            labels.append(torch.tensor(ann[field_name], device=device))
            task_weights = ann.get('task_weights', {})
            weights.append(task_weights.get(f'{field_name}_weight', 1.0))

        return torch.stack(labels), torch.tensor(weights, device=device)

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = None) -> Predictions:

        if test_cfg is None:
            test_cfg = {}

        preds = HeatmapHead.predict(self, feats, batch_data_samples, test_cfg)

        # During TTA feats is a list of tuples — use the base (non-flipped) feats
        base_feats = feats[0] if (test_cfg.get('flip_test', False)
                                  and isinstance(feats, list)) else feats

        for head in self.classifier_heads:
            probs = self._forward_classifier(head, base_feats)
            pred_classes = torch.argmax(probs, dim=1)
            pred_scores = torch.max(probs, dim=1)[0]

            for idx, data_sample in enumerate(batch_data_samples):
                if not hasattr(data_sample, 'pred_classifiers'):
                    data_sample.pred_classifiers = {}

                pred_class_idx = pred_classes[idx].item()
                data_sample.pred_classifiers[head.field_name] = {
                    'pred_class': pred_class_idx,
                    'pred_label': head.labels[pred_class_idx] if head.labels else None,
                    'pred_score': pred_scores[idx].item(),
                    'all_probs': probs[idx].detach().cpu().numpy(),
                }

                # Ground truth (if available)
                if (hasattr(data_sample, 'raw_ann_info')
                        and head.field_name in data_sample.raw_ann_info):
                    gt_idx = data_sample.raw_ann_info[head.field_name]
                    data_sample.pred_classifiers[head.field_name]['gt_class'] = gt_idx
                    if head.labels:
                        data_sample.pred_classifiers[head.field_name]['gt_label'] = \
                            head.labels[gt_idx]

        return preds