# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmengine.hooks import Hook
from mmpose.registry import HOOKS


@HOOKS.register_module()
class KeypointFreezeHook(Hook):
    """Freeze keypoint detector at a given epoch, training only classifiers.

    At ``freeze_epoch``, all parameters except classifier branches on
    ``HeatmapHeadWithClassifiers`` are frozen: ``requires_grad`` is set to
    False, optimizer LR is set to 0, and BatchNorm layers switch to eval
    mode so running statistics stop updating.

    Usage in config::

        custom_hooks = [
            dict(type='ClassifierLRSchedulerHook'),
            dict(type='KeypointFreezeHook', freeze_epoch=50),
        ]

    .. note::
        List this hook **after** ``ClassifierLRSchedulerHook`` so that
        classifier params are already in separate optimizer groups.

    Args:
        freeze_epoch (int): Epoch at which to freeze the keypoint detector.
        freeze_bn (bool): Set BatchNorm layers to eval mode when frozen.
            Default: True.
    """

    priority = 'VERY_LOW'

    def __init__(self, freeze_epoch: int, freeze_bn: bool = True):
        super().__init__()
        self.freeze_epoch = freeze_epoch
        self.freeze_bn = freeze_bn
        self._kp_param_ids = set()
        self._kp_group_indices = []
        self._saved_lrs = {}
        self._frozen = False
        self._bn_modules = []

    def before_run(self, runner):
        # NOTE: Runs in ``before_run`` (not ``before_train``) so that optimizer
        # param-group discovery happens AFTER ``ClassifierLRSchedulerHook`` has
        # split off the classifier groups (same stage, earlier in config order)
        # and BEFORE ``runner.load_or_resume()`` loads the optimizer state dict.
        from mmpose.models.heads.heatmap_heads.atraf \
            .heatmap_head_with_classifiers import (
                HeatmapHeadWithClassifiers,
            )

        model = runner.model
        while hasattr(model, 'module'):
            model = model.module

        head = None
        for m in model.modules():
            if isinstance(m, HeatmapHeadWithClassifiers):
                head = m
                break

        if head is None:
            runner.logger.warning(
                'KeypointFreezeHook: '
                'no HeatmapHeadWithClassifiers found in model')
            return

        classifier_param_ids = set()
        for cls_head in head.classifier_heads:
            for p in cls_head.parameters():
                classifier_param_ids.add(id(p))

        for p in model.parameters():
            if id(p) not in classifier_param_ids:
                self._kp_param_ids.add(id(p))

        optimizer = runner.optim_wrapper.optimizer
        for idx, group in enumerate(optimizer.param_groups):
            group_pids = {id(p) for p in group['params']}
            if group_pids and group_pids.issubset(self._kp_param_ids):
                self._kp_group_indices.append(idx)

        # Collect BN modules in keypoint parts (excluding classifiers).
        classifier_module_ids = set()
        for cls_head in head.classifier_heads:
            for m in cls_head.modules():
                classifier_module_ids.add(id(m))

        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                    nn.SyncBatchNorm)
        for m in model.modules():
            if id(m) not in classifier_module_ids and isinstance(m, bn_types):
                self._bn_modules.append(m)

        runner.logger.info(
            f'KeypointFreezeHook: tracked {len(self._kp_param_ids)} keypoint '
            f'params ({len(self._kp_group_indices)} optimizer groups, '
            f'{len(self._bn_modules)} BN layers), '
            f'will freeze at epoch {self.freeze_epoch}')

    def before_train_epoch(self, runner):
        if not self._kp_param_ids:
            return

        should_freeze = runner.epoch >= self.freeze_epoch

        if should_freeze and not self._frozen:
            self._freeze(runner)
        elif not should_freeze and self._frozen:
            self._unfreeze(runner)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        # model.train() runs after before_train_epoch but before iterations,
        # resetting BN to train mode. Re-apply eval on the first iteration.
        if self._frozen and self.freeze_bn and batch_idx == 0:
            for m in self._bn_modules:
                m.eval()

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None):
        if not self._frozen:
            return
        # Re-enforce LR=0 in case a global LR scheduler overrode it.
        optimizer = runner.optim_wrapper.optimizer
        for idx in self._kp_group_indices:
            optimizer.param_groups[idx]['lr'] = 0.0

    def _freeze(self, runner):
        optimizer = runner.optim_wrapper.optimizer

        for group in optimizer.param_groups:
            for p in group['params']:
                if id(p) in self._kp_param_ids:
                    p.requires_grad_(False)

        for idx in self._kp_group_indices:
            self._saved_lrs[idx] = optimizer.param_groups[idx]['lr']
            optimizer.param_groups[idx]['lr'] = 0.0

        self._frozen = True
        runner.logger.info(
            f'KeypointFreezeHook: froze keypoint detector at epoch '
            f'{runner.epoch} ({len(self._kp_param_ids)} params)')

    def _unfreeze(self, runner):
        optimizer = runner.optim_wrapper.optimizer

        for group in optimizer.param_groups:
            for p in group['params']:
                if id(p) in self._kp_param_ids:
                    p.requires_grad_(True)

        for idx, lr in self._saved_lrs.items():
            optimizer.param_groups[idx]['lr'] = lr
        self._saved_lrs.clear()

        self._frozen = False
        runner.logger.info(
            f'KeypointFreezeHook: unfroze keypoint detector at epoch '
            f'{runner.epoch}')
