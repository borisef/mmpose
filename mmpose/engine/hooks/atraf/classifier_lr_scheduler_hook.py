# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmengine.hooks import Hook
from mmpose.registry import HOOKS


@HOOKS.register_module()
class ClassifierLRSchedulerHook(Hook):
    """Per-classifier learning rate scheduling.

    Reads ``lr_schedule`` from each ClassifierHead and manages dedicated
    optimizer param groups with epoch-based LR steps.  When lr=0,
    parameters are frozen (requires_grad=False) to save compute and
    prevent stale momentum accumulation in Adam-family optimizers.
    BatchNorm layers are also set to eval mode when frozen so that
    running statistics stop updating.

    The hook runs with VERY_LOW priority so it overrides any global
    LR scheduler (ParamSchedulerHook) that may also touch these groups.

    Usage in config::

        custom_hooks = [dict(type='ClassifierLRSchedulerHook')]

    And in the head's classifier list::

        classifiers=[
            dict(
                field_name='gender',
                num_classes=2,
                lr_schedule=[(0, 0.0), (10, 0.0001), (20, 0.001)],
            ),
        ]

    Args:
        reset_optimizer_state (bool): When transitioning from lr=0 to
            lr>0, clear accumulated optimizer state (momentum, etc.)
            so training starts fresh.  Default: True.
    """

    priority = 'VERY_LOW'

    _bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                 nn.SyncBatchNorm)

    def __init__(self, reset_optimizer_state: bool = True):
        super().__init__()
        self.reset_optimizer_state = reset_optimizer_state
        self._group_info = []
        self._current_lrs = {}
        self._frozen_bn_modules = []

    def before_run(self, runner):
        # NOTE: This runs in ``before_run`` (not ``before_train``) so that the
        # extra per-classifier optimizer param groups are added BEFORE
        # ``runner.load_or_resume()`` loads the optimizer state dict.  The saved
        # checkpoint contains these extra groups, so the live optimizer must
        # already have them at resume time or loading fails with "loaded state
        # dict has a different number of parameter groups".
        from mmpose.models.heads.heatmap_heads.atraf.heatmap_head_with_classifiers import (
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
                'ClassifierLRSchedulerHook: '
                'no HeatmapHeadWithClassifiers found in model')
            return

        optimizer = runner.optim_wrapper.optimizer

        for cls_head in head.classifier_heads:
            if cls_head.lr_schedule is None:
                continue

            schedule = cls_head.lr_schedule
            cls_param_ids = {id(p) for p in cls_head.parameters()}

            for group in optimizer.param_groups:
                group['params'] = [
                    p for p in group['params'] if id(p) not in cls_param_ids
                ]

            lr = self._get_lr_for_epoch(schedule, 0)

            optimizer.add_param_group({
                'params': list(cls_head.parameters()),
                'lr': lr,
                'initial_lr': lr,
            })

            idx = len(optimizer.param_groups) - 1

            bn_modules = [m for m in cls_head.modules()
                          if isinstance(m, self._bn_types)]

            self._group_info.append({
                'group_idx': idx,
                'schedule': schedule,
                'field_name': cls_head.field_name,
                'bn_modules': bn_modules,
            })
            self._current_lrs[idx] = lr

            if lr == 0:
                for p in optimizer.param_groups[idx]['params']:
                    p.requires_grad_(False)
                self._frozen_bn_modules.extend(bn_modules)

            runner.logger.info(
                f"ClassifierLRSchedulerHook: '{cls_head.field_name}' "
                f"-> param group {idx}, LR={lr}"
                f"{' (frozen)' if lr == 0 else ''}")

    def before_train_epoch(self, runner):
        if not self._group_info:
            return

        epoch = runner.epoch
        optimizer = runner.optim_wrapper.optimizer

        self._frozen_bn_modules = []

        for info in self._group_info:
            lr = self._get_lr_for_epoch(info['schedule'], epoch)
            idx = info['group_idx']
            prev_lr = self._current_lrs[idx]

            frozen = (lr == 0)
            if frozen:
                self._frozen_bn_modules.extend(info['bn_modules'])

            if lr == prev_lr:
                continue

            group = optimizer.param_groups[idx]
            group['lr'] = lr
            self._current_lrs[idx] = lr

            for p in group['params']:
                p.requires_grad_(not frozen)

            if self.reset_optimizer_state and prev_lr == 0 and lr > 0:
                for p in group['params']:
                    if p in optimizer.state:
                        del optimizer.state[p]

            runner.logger.info(
                f"ClassifierLRSchedulerHook: '{info['field_name']}' "
                f"epoch {epoch} LR: {prev_lr} -> {lr}"
                f"{' (frozen)' if frozen else ''}")

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        # model.train() runs after before_train_epoch, resetting BN to
        # train mode.  Re-apply eval on the first iteration of each epoch.
        if self._frozen_bn_modules and batch_idx == 0:
            for m in self._frozen_bn_modules:
                m.eval()

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None):
        if not self._group_info:
            return
        optimizer = runner.optim_wrapper.optimizer
        for info in self._group_info:
            idx = info['group_idx']
            lr = self._current_lrs[idx]
            optimizer.param_groups[idx]['lr'] = lr
            runner.message_hub.update_scalar(
                f'train/classifier_lr_{info["field_name"]}', lr)

    @staticmethod
    def _get_lr_for_epoch(schedule, epoch):
        lr = schedule[0][1]
        for e, l in schedule:
            if epoch >= e:
                lr = l
            else:
                break
        return lr
