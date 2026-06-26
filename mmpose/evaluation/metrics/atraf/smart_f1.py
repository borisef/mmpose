# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union
import io
import os
import numpy as np
from mmengine.logging import MMLogger
from mmpose.registry import METRICS
from mmpose.evaluation.metrics import PCKAccuracy
from mmpose.evaluation.functional.keypoint_eval import _calc_distances
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
    """Save image to TensorBoard. Handles conversion from PNG to proper tensor format."""
    if not _TB_AVAILABLE:
        return False
    try:
        import matplotlib.image as mpimg
        img = mpimg.imread(image_path)  # [H, W, C] in [0, 1] range
        # Remove alpha channel if present
        if img.shape[2] == 4:
            img = img[:, :, :3]
        # Convert to [C, H, W] format
        if _TORCH_AVAILABLE:
            # Convert to torch tensor [C, H, W]
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # [C, H, W]
            writer = SummaryWriter(log_dir=log_dir)
            if global_step is not None:
                writer.add_image(tag, img_tensor, global_step=global_step, dataformats='CHW')
            else:
                writer.add_image(tag, img_tensor, dataformats='CHW')
            writer.close()
        else:
            # Fallback: use numpy, reshape and convert
            img_hwc = img.astype(np.float32)
            # Make sure it's in correct range for numpy-based tensorboard
            if img_hwc.max() <= 1.0:
                img_hwc = (img_hwc * 255).astype(np.uint8)
            writer = SummaryWriter(log_dir=log_dir)
            if global_step is not None:
                writer.add_image(tag, img_hwc, global_step=global_step, dataformats='HWC')
            else:
                writer.add_image(tag, img_hwc, dataformats='HWC')
            writer.close()
        return True
    except Exception as e:
        import traceback
        MMLogger.get_current_instance().warning(f'Failed to write image to TensorBoard: {e}\n{traceback.format_exc()}')
        return False
@METRICS.register_module()
class Smart_F1(PCKAccuracy):
    """Smart F1 searcher for ATRAF metrics.
    This metric scans score thresholds between 0 and 1 (inclusive) with
    ``num_steps`` samples and selects the threshold that yields the highest
    F1 (harmonic mean of precision and recall). It returns the best F1 and
    the best threshold as metrics.
    Args:
        thr(float): Threshold of PCK calculation. Default: 0.05.
        norm_item (str | Sequence[str]): The item used for normalization.
            Valid items include 'bbox', 'head', 'torso'. Default: ``'bbox'``.
        kpt_indexes (Sequence[int], optional): Indices of keypoints to include
            in the evaluation. If None, all keypoints are included.
            Default: ``None``.
        num_steps (int): Number of thresholds to sample in [0, 1]. Default: 101
            (i.e., step 0.01).
        collect_device (str): Device used for collecting results. Default: 'cpu'.
        prefix (str, optional): Metric prefix. Default: ``None``.
        twin_keypoints (Sequence[Sequence[int]], optional): Twin keypoints for
            symmetric matching. Default: ``None``.
        generate_chart (bool): Whether to generate and save Precision-Recall charts.
            Default: ``False``.
        chart_images_folder (str, optional): Folder to save chart PNGs. If None and
            generate_chart is True, uses './chart_images'. Default: ``None``.
    """
    def __init__(self,
                 thr: float = 0.05,
                 norm_item: Union[str, Sequence[str]] = 'bbox',
                 kpt_indexes: Optional[Sequence[int]] = None,
                 num_steps: int = 101,
                 ignore_gt_out_of_image: bool = False,
                 ignore_gt_out_of_bbox: bool = False,
                 torso_keypoint_indexes: Optional[Sequence[int]] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 twin_keypoints: Optional[Sequence[Sequence[int]]] = None,
                 generate_chart: bool = False,
                 chart_images_folder: Optional[str] = None) -> None:
        super().__init__(thr=thr, norm_item=norm_item, collect_device=collect_device, prefix=prefix)
        self.kpt_indexes = kpt_indexes
        self.num_steps = int(num_steps)
        self.twin_keypoints = twin_keypoints
        self.ignore_gt_out_of_image = ignore_gt_out_of_image
        self.ignore_gt_out_of_bbox = ignore_gt_out_of_bbox
        self.torso_keypoint_indexes = torso_keypoint_indexes or [4, 5]
        self.generate_chart = bool(generate_chart)
        self.chart_images_folder = chart_images_folder
    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]) -> None:
        # Reuse the same processing as PCK-like metrics: store pred/gt/mask/scores
        for data_sample in data_samples:
            pred_coords = data_sample['pred_instances']['keypoints']
            pred_scores = data_sample['pred_instances'].get('keypoint_scores', None)
            gt = data_sample['gt_instances']
            gt_coords = gt['keypoints']
            mask = gt['keypoints_visible'].astype(bool)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.reshape(1, -1)
            if self.kpt_indexes is not None:
                kpt_indexes = np.array(self.kpt_indexes)
                pred_coords = pred_coords[:, kpt_indexes, :]
                gt_coords = gt_coords[:, kpt_indexes, :]
                mask = mask[:, kpt_indexes]
                if pred_scores is not None:
                    pred_scores = pred_scores[:, kpt_indexes]
            if self.ignore_gt_out_of_image:
                img_shape = data_sample.get('img_shape', None)
                if img_shape is not None:
                    h, w = img_shape[0], img_shape[1]
                    gt_xy = gt_coords[0]
                    out_of_image = (
                        (gt_xy[:, 0] < 0) | (gt_xy[:, 1] < 0) |
                        (gt_xy[:, 0] > w) | (gt_xy[:, 1] > h))
                    mask[0, out_of_image] = False
            if self.ignore_gt_out_of_bbox:
                if 'bboxes' in gt:
                    bbox = gt['bboxes'][0]
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    gt_xy = gt_coords[0]
                    out_of_bbox = (
                        (gt_xy[:, 0] < x1) | (gt_xy[:, 1] < y1) |
                        (gt_xy[:, 0] > x2) | (gt_xy[:, 1] > y2))
                    mask[0, out_of_bbox] = False
            if pred_scores is None:
                pred_scores = np.ones((pred_coords.shape[0], pred_coords.shape[1]), dtype=np.float32)
            else:
                pred_scores = np.array(pred_scores)
                if pred_scores.ndim == 1:
                    pred_scores = pred_scores.reshape(1, -1)
            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
                'pred_scores': pred_scores,
            }
            if 'bbox' in self.norm_item:
                assert 'bboxes' in gt, 'The ground truth data info do not have the expected normalized_item ``"bbox"``.'
                bbox_size_ = np.max(gt['bboxes'][0][2:] - gt['bboxes'][0][:2])
                bbox_size = np.array([bbox_size_, bbox_size_]).reshape(-1, 2)
                result['bbox_size'] = bbox_size
            if 'head' in self.norm_item:
                assert 'head_size' in gt, 'The ground truth data info do not have the expected normalized_item ``"head_size"``.'
                head_size_ = gt['head_size']
                head_size = np.array([head_size_, head_size_]).reshape(-1, 2)
                result['head_size'] = head_size
            if 'torso' in self.norm_item:
                tk0, tk1 = self.torso_keypoint_indexes[0], self.torso_keypoint_indexes[1]
                if self.kpt_indexes is not None:
                    kpt_indexes = np.array(self.kpt_indexes)
                    torso_kpt_0_idx = np.where(kpt_indexes == tk0)[0]
                    torso_kpt_1_idx = np.where(kpt_indexes == tk1)[0]
                    if len(torso_kpt_0_idx) > 0 and len(torso_kpt_1_idx) > 0:
                        torso_size_ = np.linalg.norm(gt_coords[0][torso_kpt_0_idx[0]] - gt_coords[0][torso_kpt_1_idx[0]])
                        if torso_size_ < 1:
                            torso_size_ = np.linalg.norm(pred_coords[0][torso_kpt_0_idx[0]] - pred_coords[0][torso_kpt_1_idx[0]])
                    else:
                        torso_size_ = None
                else:
                    torso_size_ = np.linalg.norm(gt_coords[0][tk0] - gt_coords[0][tk1])
                    if torso_size_ < 1:
                        torso_size_ = np.linalg.norm(pred_coords[0][tk0] - pred_coords[0][tk1])
                if torso_size_ is not None:
                    torso_size = np.array([torso_size_, torso_size_]).reshape(-1, 2)
                    result['torso_size'] = torso_size
                else:
                    result['torso_size'] = None
            self.results.append(result)
    def _eval_at_threshold(self, pred_coords, gt_coords, mask, pred_scores, norm_factor, score_threshold):
        # distances: [K, N]
        distances = _calc_distances(pred_coords, gt_coords, mask, norm_factor)
        if self.twin_keypoints is not None:
            distances_orig = distances.copy()
            distances_combined = distances_orig.copy()
            for pair in self.twin_keypoints:
                i, j = pair
                gt_swapped = gt_coords.copy()
                gt_swapped[:, i, :] = gt_coords[:, j, :]
                gt_swapped[:, j, :] = gt_coords[:, i, :]
                mask_swapped = mask.copy()
                mask_swapped[:, i] = mask[:, i] | mask[:, j]
                mask_swapped[:, j] = mask[:, j] | mask[:, i]
                distances_swapped = _calc_distances(pred_coords, gt_swapped, mask_swapped, norm_factor)
                for idx in (i, j):
                    orig = distances_orig[idx]
                    swp = distances_swapped[idx]
                    combined = np.where((orig != -1) & (swp != -1), np.minimum(orig, swp), np.where(orig != -1, orig, swp))
                    distances_combined[idx] = combined
            distances = distances_combined
        valid = distances != -1  # [K, N]
        valid_t = valid.T  # [N, K]
        correct = (distances < self.thr).T & valid_t  # [N, K]
        incorrect = (~correct) & valid_t
        score_mask = pred_scores > score_threshold
        num_correct = int(correct.sum())
        correct_high = int(((correct) & score_mask).sum())
        recall = float(correct_high / num_correct) if num_correct > 0 else 0.0
        num_high_score = int(score_mask.sum())
        incorrect_high = int(((incorrect) & score_mask).sum())
        far = float(incorrect_high / num_high_score) if num_high_score > 0 else 0.0
        precision = float(correct_high / num_high_score) if num_high_score > 0 else 0.0
        if (precision + recall) > 0.0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = 0.0
        total_valid = int(valid_t.sum())
        accuracy = float(correct_high / total_valid) if total_valid > 0 else 0.0
        return recall, far, precision, f1, accuracy
    def _generate_and_log_chart(self, recalls, precisions, thresholds, best_t, best_f1, norm_name):
        """Generate PR chart and optionally log to TensorBoard."""
        if not self.generate_chart or plt is None:
            return
        logger = MMLogger.get_current_instance()
        try:
            out_folder = self.chart_images_folder or os.path.join(os.getcwd(), 'chart_images')
            os.makedirs(out_folder, exist_ok=True)
            name = f'SmartF1_{norm_name}'
            # include step in filename if available
            step = None
            if hasattr(self, '_chart_step'):
                step = int(self._chart_step)
            if step is not None:
                out_path = os.path.join(out_folder, f'{name}_step{step}.png')
            else:
                out_path = os.path.join(out_folder, f'{name}.png')
            # plot Precision vs Recall
            fig, ax = plt.subplots(figsize=(6, 6))
            # omit degenerate (0, 0) points from the plotted curve (high score
            # thresholds where precision is 0/0 and recall is 0)
            rec_arr = np.asarray(recalls)
            prec_arr = np.asarray(precisions)
            keep = ~((rec_arr == 0) & (prec_arr == 0))
            ax.plot(rec_arr[keep], prec_arr[keep], '-o', markersize=3)
            # mark best
            best_idx = int(np.argmin(np.abs(thresholds - best_t)))
            ax.plot(recalls[best_idx], precisions[best_idx], 'ro', markersize=8)
            ax.annotate(f't={best_t:.3f}\nF1={best_f1:.3f}', xy=(recalls[best_idx], precisions[best_idx]),
                        xytext=(5, -15), textcoords='offset points')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Smart F1 ({norm_name})')
            # keep axes anchored at the origin even though the (0, 0) point is
            # omitted from the plotted curve
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)
            logger.info(f'Saved Smart_F1 chart to {out_path}')
            # Try to log to TensorBoard (pass chart step if available)
            if _TB_AVAILABLE:
                gs = int(self._chart_step) if hasattr(self, '_chart_step') else None
                if _save_image_to_tensorboard(out_path, out_folder, f'{self.__class__.__name__}/{name}', global_step=gs):
                    logger.info(f'Logged Smart_F1 chart to TensorBoard (logdir={out_folder}, step={gs})')
                else:
                    logger.warning('Failed to write Smart_F1 chart to TensorBoard.')
        except Exception as e:
            logger.warning(f'Failed to generate Smart_F1 chart: {e}')
    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        pred_coords = np.concatenate([r['pred_coords'] for r in results])
        gt_coords = np.concatenate([r['gt_coords'] for r in results])
        mask = np.concatenate([r['mask'] for r in results])
        pred_scores = np.concatenate([r['pred_scores'] for r in results])
        metrics = dict()
        metric_prefix = ' (filtered by kpt_indexes)' if self.kpt_indexes else ''
        thresholds = np.linspace(0.0, 1.0, self.num_steps)

        # chart step counter: increment per compute_metrics call so TensorBoard
        # images show up at increasing steps. Initialize if not present.
        if not hasattr(self, '_chart_step'):
            self._chart_step = 0
        chart_step = int(self._chart_step)
        # Prefer the resume-safe training epoch as the chart/TensorBoard step;
        # keep the internal counter as a fallback for standalone test runs.
        self._chart_step = _get_chart_step(fallback=chart_step)
        if 'bbox' in self.norm_item:
            norm_size_bbox = np.concatenate([r['bbox_size'] for r in results])
            logger.info(f'Evaluating {self.__class__.__name__} (normalized by ``"bbox_size"``){metric_prefix}...')
            best_f1 = -1.0
            best_t = 0.0
            best_acc = 0.0
            precisions = []
            recalls = []
            for t in thresholds:
                rec, _, prec, f1, acc = self._eval_at_threshold(pred_coords, gt_coords, mask, pred_scores, norm_size_bbox, t)
                precisions.append(prec)
                recalls.append(rec)
                if f1 > best_f1 or (abs(f1 - best_f1) <= 1e-12 and t < best_t):
                    best_f1 = f1
                    best_t = float(t)
                    best_acc = acc
            metrics['SmartF1'] = float(best_f1)
            metrics['SmartThreshold'] = float(best_t)
            metrics['SmartAccuracy'] = float(best_acc)
            # pass chart_step via self attribute; _generate_and_log_chart will use it
            self._generate_and_log_chart(recalls, precisions, thresholds, best_t, best_f1, 'bbox')
        if 'head' in self.norm_item:
            norm_size_head = np.concatenate([r['head_size'] for r in results])
            logger.info(f'Evaluating {self.__class__.__name__} (normalized by ``"head_size"``){metric_prefix}...')
            best_f1 = -1.0
            best_t = 0.0
            best_acc = 0.0
            precisions = []
            recalls = []
            for t in thresholds:
                rec, _, prec, f1, acc = self._eval_at_threshold(pred_coords, gt_coords, mask, pred_scores, norm_size_head, t)
                precisions.append(prec)
                recalls.append(rec)
                if f1 > best_f1 or (abs(f1 - best_f1) <= 1e-12 and t < best_t):
                    best_f1 = f1
                    best_t = float(t)
                    best_acc = acc
            metrics['SmartF1h'] = float(best_f1)
            metrics['SmartThresholdh'] = float(best_t)
            metrics['SmartAccuracyh'] = float(best_acc)
            self._generate_and_log_chart(recalls, precisions, thresholds, best_t, best_f1, 'head')
        if 'torso' in self.norm_item:
            valid_torso_results = [r for r in results if r.get('torso_size') is not None]
            if valid_torso_results:
                norm_size_torso = np.concatenate([r['torso_size'] for r in valid_torso_results])
                valid_indices = [i for i, r in enumerate(results) if r.get('torso_size') is not None]
                valid_pred_coords = pred_coords[valid_indices]
                valid_gt_coords = gt_coords[valid_indices]
                valid_mask = mask[valid_indices]
                valid_pred_scores = pred_scores[valid_indices]
                logger.info(f'Evaluating {self.__class__.__name__} (normalized by ``"torso_size"``){metric_prefix}...')
                best_f1 = -1.0
                best_t = 0.0
                best_acc = 0.0
                precisions = []
                recalls = []
                for t in thresholds:
                    rec, _, prec, f1, acc = self._eval_at_threshold(valid_pred_coords, valid_gt_coords, valid_mask, valid_pred_scores, norm_size_torso, t)
                    precisions.append(prec)
                    recalls.append(rec)
                    if f1 > best_f1 or (abs(f1 - best_f1) <= 1e-12 and t < best_t):
                        best_f1 = f1
                        best_t = float(t)
                        best_acc = acc
                metrics['SmartF1t'] = float(best_f1)
                metrics['SmartThresholdt'] = float(best_t)
                metrics['SmartAccuracyt'] = float(best_acc)
                self._generate_and_log_chart(recalls, precisions, thresholds, best_t, best_f1, 'torso')

        # increment chart step for next invocation
        self._chart_step = chart_step + 1
        return metrics
