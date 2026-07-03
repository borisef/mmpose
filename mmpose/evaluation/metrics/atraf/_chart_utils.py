# Copyright (c) OpenMMLab. All rights reserved.
"""Shared charting / TensorBoard helpers for ATRAF metrics.

These utilities are used by the recall, success-rate and confusion-matrix
metrics to save PNG charts and (optionally) log them to TensorBoard at a
resume-safe training step.
"""
import numpy as np
from mmengine.logging import MMLogger

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

_TB_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except Exception:
    try:
        from tensorboardX import SummaryWriter
        _TB_AVAILABLE = True
    except Exception:
        _TB_AVAILABLE = False

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    pass


def _get_chart_step(fallback=None):
    """Return a resume-safe TensorBoard step (current training epoch).

    Reads 'epoch' from MMEngine's MessageHub, which the runner restores from the
    checkpoint on resume. Falls back to ``fallback`` (e.g. an internal counter)
    when the info is unavailable (e.g. standalone test runs).
    """
    try:
        from mmengine.logging import MessageHub
        step = MessageHub.get_current_instance().get_info('epoch', None)
        if step is not None:
            return int(step)
    except Exception:
        pass
    return fallback


def _save_image_to_tensorboard(image_path, log_dir, tag, global_step=None):
    """Save image to TensorBoard, converting PNG to proper tensor format."""
    if not _TB_AVAILABLE:
        return False
    try:
        import matplotlib.image as mpimg
        img = mpimg.imread(image_path)  # [H, W, C] in [0, 1] range
        # Remove alpha channel if present
        if img.shape[2] == 4:
            img = img[:, :, :3]
        if _TORCH_AVAILABLE:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            writer = SummaryWriter(log_dir=log_dir)
            if global_step is not None:
                writer.add_image(
                    tag, img_tensor, global_step=global_step,
                    dataformats='CHW')
            else:
                writer.add_image(tag, img_tensor, dataformats='CHW')
            writer.close()
        else:
            img_hwc = img.astype(np.float32)
            if img_hwc.max() <= 1.0:
                img_hwc = (img_hwc * 255).astype(np.uint8)
            writer = SummaryWriter(log_dir=log_dir)
            if global_step is not None:
                writer.add_image(
                    tag, img_hwc, global_step=global_step,
                    dataformats='HWC')
            else:
                writer.add_image(tag, img_hwc, dataformats='HWC')
            writer.close()
        return True
    except Exception as e:
        import traceback
        MMLogger.get_current_instance().warning(
            f'Failed to write image to TensorBoard: {e}\n'
            f'{traceback.format_exc()}')
        return False
