# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

import numpy as np
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from mmpose.evaluation.metrics import PCKAccuracy
from mmpose.evaluation.functional.keypoint_eval import _calc_distances


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
    """

    def __init__(self,
                 thr: float = 0.05,
                 norm_item: Union[str, Sequence[str]] = 'bbox',
                 kpt_indexes: Optional[Sequence[int]] = None,
                 num_steps: int = 101,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 twin_keypoints: Optional[Sequence[Sequence[int]]] = None) -> None:
        super().__init__(thr=thr, norm_item=norm_item, collect_device=collect_device, prefix=prefix)
        self.kpt_indexes = kpt_indexes
        self.num_steps = int(num_steps)
        self.twin_keypoints = twin_keypoints

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
                if self.kpt_indexes is not None:
                    kpt_indexes = np.array(self.kpt_indexes)
                    torso_kpt_4_idx = np.where(kpt_indexes == 4)[0]
                    torso_kpt_5_idx = np.where(kpt_indexes == 5)[0]

                    if len(torso_kpt_4_idx) > 0 and len(torso_kpt_5_idx) > 0:
                        torso_size_ = np.linalg.norm(gt_coords[0][torso_kpt_4_idx[0]] - gt_coords[0][torso_kpt_5_idx[0]])
                        if torso_size_ < 1:
                            torso_size_ = np.linalg.norm(pred_coords[0][torso_kpt_4_idx[0]] - pred_coords[0][torso_kpt_5_idx[0]])
                    else:
                        torso_size_ = None
                else:
                    torso_size_ = np.linalg.norm(gt_coords[0][4] - gt_coords[0][5])
                    if torso_size_ < 1:
                        torso_size_ = np.linalg.norm(pred_coords[0][4] - pred_coords[0][5])

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

        return recall, far, precision, f1

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        pred_coords = np.concatenate([r['pred_coords'] for r in results])
        gt_coords = np.concatenate([r['gt_coords'] for r in results])
        mask = np.concatenate([r['mask'] for r in results])
        pred_scores = np.concatenate([r['pred_scores'] for r in results])

        metrics = dict()
        metric_prefix = ' (filtered by kpt_indexes)' if self.kpt_indexes else ''

        thresholds = np.linspace(0.0, 1.0, self.num_steps)

        if 'bbox' in self.norm_item:
            norm_size_bbox = np.concatenate([r['bbox_size'] for r in results])
            logger.info(f'Evaluating {self.__class__.__name__} (normalized by ``"bbox_size"``){metric_prefix}...')
            best_f1 = -1.0
            best_t = 0.0
            for t in thresholds:
                _, _, _, f1 = self._eval_at_threshold(pred_coords, gt_coords, mask, pred_scores, norm_size_bbox, t)
                if f1 > best_f1 or (abs(f1 - best_f1) <= 1e-12 and t < best_t):
                    best_f1 = f1
                    best_t = float(t)

            metrics['SmartF1'] = float(best_f1)
            metrics['SmartThreshold'] = float(best_t)

        if 'head' in self.norm_item:
            norm_size_head = np.concatenate([r['head_size'] for r in results])
            logger.info(f'Evaluating {self.__class__.__name__} (normalized by ``"head_size"``){metric_prefix}...')
            best_f1 = -1.0
            best_t = 0.0
            for t in thresholds:
                _, _, _, f1 = self._eval_at_threshold(pred_coords, gt_coords, mask, pred_scores, norm_size_head, t)
                if f1 > best_f1 or (abs(f1 - best_f1) <= 1e-12 and t < best_t):
                    best_f1 = f1
                    best_t = float(t)
            metrics['SmartF1h'] = float(best_f1)
            metrics['SmartThresholdh'] = float(best_t)

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
                for t in thresholds:
                    _, _, _, f1 = self._eval_at_threshold(valid_pred_coords, valid_gt_coords, valid_mask, valid_pred_scores, norm_size_torso, t)
                    if f1 > best_f1 or (abs(f1 - best_f1) <= 1e-12 and t < best_t):
                        best_f1 = f1
                        best_t = float(t)
                metrics['SmartF1t'] = float(best_f1)
                metrics['SmartThresholdt'] = float(best_t)

        return metrics

