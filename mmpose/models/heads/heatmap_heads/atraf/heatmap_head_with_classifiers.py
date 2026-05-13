# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.models.heads.heatmap_heads import *
import copy
import warnings

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, OptSampleList, Predictions)


# ── Loss factory ──────────────────────────────────────────────────────────────

def _build_classifier_loss(loss_cfg: dict) -> nn.Module:
    """Build a classification loss from a plain config dict.

    Supported types (case-insensitive):
        - CrossEntropyLoss  : standard nn.CrossEntropyLoss (expects logits)
        - BCEWithLogitsLoss : nn.BCEWithLogitsLoss for binary / multi-label
        - FocalLoss         : built-in focal variant (no extra dependency)

    All types honour a ``reduction`` key (default: 'none') and
    a ``loss_weight`` key (default: 1.0, stored on the wrapper but
    NOT forwarded to nn.* — weighting is done in compute_loss()).

    Args:
        loss_cfg (dict): Config dict with at least a ``type`` key.

    Returns:
        nn.Module: The constructed loss module.
    """
    cfg = copy.deepcopy(loss_cfg)
    loss_type = cfg.pop('type').lower().replace('_', '')
    reduction  = cfg.pop('reduction', 'none')
    cfg.pop('loss_weight', 1.0)   # consumed by ClassifierHead, not the module

    if loss_type == 'crossentropyloss':
        label_smoothing = cfg.pop('label_smoothing', 0.0)
        return nn.CrossEntropyLoss(reduction=reduction,
                                   label_smoothing=label_smoothing,
                                   **cfg)

    elif loss_type == 'bcewithlogitsloss':
        return nn.BCEWithLogitsLoss(reduction=reduction, **cfg)

    elif loss_type == 'focalloss':
        gamma = cfg.pop('gamma', 2.0)
        alpha = cfg.pop('alpha', None)
        return _FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)

    else:
        raise ValueError(
            f"Unknown classifier loss type '{loss_type}'. "
            f"Supported: CrossEntropyLoss, BCEWithLogitsLoss, FocalLoss."
        )


class _FocalLoss(nn.Module):
    """Focal Loss — no external dependencies required.

    FL(p) = -alpha * (1 - p)^gamma * log(p)

    Args:
        gamma (float): Focusing parameter. Default: 2.0.
        alpha (float | None): Class balancing scalar. Default: None.
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'none'.
    """

    def __init__(self,
                 gamma: float = 2.0,
                 alpha: Optional[float] = None,
                 reduction: str = 'none'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none')   # (N,)
        pt = torch.exp(-ce)                                        # (N,)
        focal = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            focal = self.alpha * focal
        if self.reduction == 'mean':
            return focal.mean()
        if self.reduction == 'sum':
            return focal.sum()
        return focal   # 'none' → (N,)


# ── ClassifierHead ────────────────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """Flexible per-task classification head.

    Architecture:

        feature map
            -> [num_convs x (Conv2d + BN + ReLU)]
            -> AdaptiveAvgPool2d(1, 1)
            -> flatten
            -> [num_fcs x (Linear + ReLU)]
            -> Linear(fc_out_channels, num_classes)   # raw logits
                                                      # softmax only at predict()

    Loss is configured via ``loss_cfg``:

    .. code-block:: python

        # Standard multi-class (default)
        loss_cfg = dict(type='CrossEntropyLoss', reduction='none')

        # With label smoothing
        loss_cfg = dict(type='CrossEntropyLoss', reduction='none',
                        label_smoothing=0.1)

        # Binary task (single logit output, num_classes=1)
        loss_cfg = dict(type='BCEWithLogitsLoss', reduction='none')

        # Imbalanced classes
        loss_cfg = dict(type='FocalLoss', gamma=2.0, reduction='none')

    .. note::
        ``reduction='none'`` is required for per-sample weighting via
        ``task_weights`` in ``raw_ann_info``.  A ``UserWarning`` is raised
        if any other reduction is used.

    Args:
        in_channels (int): Input feature channels.
        num_classes (int): Number of output classes.
        field_name (str): Key used to pull GT labels from raw_ann_info.
        weight (float): Global loss weight scalar. Default: 1.0.
        num_convs (int): Conv layers before pooling. Default: 0.
        num_fcs (int): Hidden FC layers (>= 1). Default: 1.
        conv_out_channels (int): Width of conv layers. Default: 256.
        fc_out_channels (int): Width of hidden FC layers. Default: 256.
        loss_cfg (dict | None): Loss config. Defaults to CrossEntropyLoss
            with reduction='none'.
        labels (list[str] | None): Human-readable class names (inference).
    """

    _default_loss_cfg = dict(type='CrossEntropyLoss', reduction='none')

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 field_name: str,
                 weight: float = 1.0,
                 num_convs: int = 0,
                 num_fcs: int = 1,
                 conv_out_channels: int = 256,
                 fc_out_channels: int = 256,
                 loss_cfg: Optional[dict] = None,
                 labels: Optional[List[str]] = None):
        super().__init__()
        assert num_fcs >= 1, 'num_fcs must be >= 1'

        self.field_name = field_name
        self.weight = weight
        self.labels = labels

        # ── Loss module ───────────────────────────────────────────────────
        resolved_cfg = copy.deepcopy(loss_cfg or self._default_loss_cfg)
        self.loss_weight = resolved_cfg.get('loss_weight', 1.0)
        self.loss_reduction = resolved_cfg.get('reduction', 'none')

        if self.loss_reduction != 'none':
            warnings.warn(
                f"ClassifierHead '{field_name}': loss_cfg has "
                f"reduction='{self.loss_reduction}'. Per-sample weighting "
                f"via task_weights requires reduction='none'. "
                f"Per-sample weights will be ignored.",
                UserWarning,
            )

        self.loss_module = _build_classifier_loss(resolved_cfg)

        # ── Conv tower ────────────────────────────────────────────────────
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else conv_out_channels
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, conv_out_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(conv_out_channels),
                nn.ReLU(inplace=True),
            ))

        # ── Spatial pooling ───────────────────────────────────────────────
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── FC tower ──────────────────────────────────────────────────────
        fc_in_channels = conv_out_channels if num_convs > 0 else in_channels
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_ch = fc_in_channels if i == 0 else fc_out_channels
            self.fcs.append(nn.Linear(in_ch, fc_out_channels))

        # ── Output logits ─────────────────────────────────────────────────
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
        """Returns raw logits (N, num_classes). No softmax."""
        for conv in self.convs:
            x = conv(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        for fc in self.fcs:
            x = F.relu(fc(x))
        return self.fc_out(x)

    def compute_loss(self,
                     logits: Tensor,
                     gt_labels: Tensor,
                     sample_weights: Optional[Tensor] = None) -> Tensor:
        """Weighted scalar loss for this head.

        Args:
            logits (Tensor): Raw logits (N, num_classes).
            gt_labels (Tensor): GT class indices (N,).
            sample_weights (Tensor | None): Per-sample weights (N,).
                Applied only when reduction='none'.

        Returns:
            Tensor: Scalar loss.
        """
        loss = self.loss_module(logits, gt_labels)

        if self.loss_reduction == 'none':
            if sample_weights is not None:
                loss = loss * sample_weights
            loss = loss.mean()

        return loss * self.weight


# ── HeatmapHeadWithClassifiers ────────────────────────────────────────────────

@MODELS.register_module()
class HeatmapHeadWithClassifiers(HeatmapHead):
    """HeatmapHead with flexible per-task ClassifierHead branches.

    Example config:

    .. code-block:: python

        classifiers=[
            dict(
                field_name='gender',
                num_classes=2,
                weight=1.0,
                loss_cfg=dict(type='CrossEntropyLoss', reduction='none'),
                labels=['male', 'female'],
            ),
            dict(
                field_name='age_group',
                num_classes=4,
                weight=0.5,
                num_convs=2,
                num_fcs=2,
                loss_cfg=dict(type='FocalLoss', gamma=2.0, reduction='none'),
                labels=['child', 'young', 'adult', 'senior'],
            ),
            dict(
                field_name='is_occluded',
                num_classes=1,          # single logit for binary BCE
                weight=0.3,
                loss_cfg=dict(type='BCEWithLogitsLoss', reduction='none'),
                labels=['occluded'],
            ),
        ]
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 classifiers: List[dict],
                 **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)

        self.classifier_heads = nn.ModuleList()
        for cfg in classifiers:
            self.classifier_heads.append(
                ClassifierHead(in_channels=in_channels, **copy.deepcopy(cfg))
            )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _forward_classifier(self,
                             head: ClassifierHead,
                             feats: Tuple[Tensor]) -> Tensor:
        return head(feats[0])

    @staticmethod
    def _extract_labels_and_weights(
            field_name: str,
            batch_data_samples: OptSampleList,
            device) -> Tuple[Tensor, Tensor]:
        labels, weights = [], []
        for bds in batch_data_samples:
            ann = bds.to_dict()['raw_ann_info']
            labels.append(torch.tensor(ann[field_name], device=device))
            task_weights = ann.get('task_weights', {})
            weights.append(task_weights.get(f'{field_name}_weight', 1.0))
        return torch.stack(labels), torch.tensor(weights, device=device)

    def _classifier_losses(self,
                           feats: Tuple[Tensor],
                           batch_data_samples: OptSampleList) -> dict:
        losses = {}
        for head in self.classifier_heads:
            logits = self._forward_classifier(head, feats)
            gt_labels, sample_weights = self._extract_labels_and_weights(
                head.field_name, batch_data_samples, feats[0].device)
            losses[f'loss_{head.field_name}'] = head.compute_loss(
                logits, gt_labels, sample_weights)
        return losses

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

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = None) -> Predictions:
        if test_cfg is None:
            test_cfg = {}

        preds = HeatmapHead.predict(self, feats, batch_data_samples, test_cfg)

        base_feats = (feats[0] if (test_cfg.get('flip_test', False)
                                   and isinstance(feats, list)) else feats)

        for head in self.classifier_heads:
            logits = self._forward_classifier(head, base_feats)
            probs = F.softmax(logits, dim=1)   # softmax only at inference
            pred_classes = torch.argmax(probs, dim=1)
            pred_scores = torch.max(probs, dim=1)[0]

            for idx, data_sample in enumerate(batch_data_samples):
                if not hasattr(data_sample, 'pred_classifiers'):
                    data_sample.pred_classifiers = {}

                pred_class_idx = pred_classes[idx].item()
                data_sample.pred_classifiers[head.field_name] = {
                    'pred_class': pred_class_idx,
                    'pred_label': (head.labels[pred_class_idx]
                                   if head.labels else None),
                    'pred_score': pred_scores[idx].item(),
                    'all_probs':  probs[idx].detach().cpu().numpy(),
                }

                if (hasattr(data_sample, 'raw_ann_info')
                        and head.field_name in data_sample.raw_ann_info):
                    gt_idx = data_sample.raw_ann_info[head.field_name]
                    data_sample.pred_classifiers[head.field_name]['gt_class'] = gt_idx
                    if head.labels:
                        data_sample.pred_classifiers[head.field_name]['gt_label'] = \
                            head.labels[gt_idx]

        return preds