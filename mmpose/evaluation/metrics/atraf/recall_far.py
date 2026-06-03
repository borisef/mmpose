# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

import numpy as np
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from mmpose.evaluation.metrics import PCKAccuracy
from mmpose.evaluation.functional.keypoint_eval import _calc_distances, _distance_acc


@METRICS.register_module()
class Recall_Atraf(PCKAccuracy):
    """ATRAF Recall/FAR metrics.

    Recall: portion of correct keypoints (distance < thr) with
    predicted keypoint score > score_threshold.

    FAR (False Alarm Rate): portion of high-score detections that are
    incorrect (distance >= thr). FAR = (Incorrect & High-score) / Total High-score.

    Args:
        thr(float): Threshold of PCK calculation. Default: 0.05.
        norm_item (str | Sequence[str]): The item used for normalization.
            Valid items include 'bbox', 'head', 'torso'. Default: ``'bbox'``.
        kpt_indexes (Sequence[int], optional): Indices of keypoints to include
            in the evaluation. If None, all keypoints are included.
            Default: ``None``.
        score_threshold (float): Keypoint score threshold to consider a
            prediction a "high-score". Default: 0.5.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Default: ``'cpu'``.
        prefix (str, optional): Metric prefix. Default: ``None``.
    """

    def __init__(self,
                 thr: float = 0.05,
                 norm_item: Union[str, Sequence[str]] = 'bbox',
                 kpt_indexes: Optional[Sequence[int]] = None,
                 score_threshold: float = 0.5,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 twin_keypoints: Optional[Sequence[Sequence[int]]] = None) -> None:
        super().__init__(
            thr=thr,
            norm_item=norm_item,
            collect_device=collect_device,
            prefix=prefix)
        self.kpt_indexes = kpt_indexes
        self.score_threshold = score_threshold
        self.twin_keypoints = twin_keypoints

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples with keypoint filtering and scores."""
        for data_sample in data_samples:
            pred_coords = data_sample['pred_instances']['keypoints']
            pred_scores = data_sample['pred_instances'].get('keypoint_scores', None)

            gt = data_sample['gt_instances']
            gt_coords = gt['keypoints']
            mask = gt['keypoints_visible'].astype(bool)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.reshape(1, -1)

            # filter keypoints by indices if specified
            if self.kpt_indexes is not None:
                kpt_indexes = np.array(self.kpt_indexes)
                pred_coords = pred_coords[:, kpt_indexes, :]
                gt_coords = gt_coords[:, kpt_indexes, :]
                mask = mask[:, kpt_indexes]
                if pred_scores is not None:
                    pred_scores = pred_scores[:, kpt_indexes]

            # ensure pred_scores exists and has batch dim
            if pred_scores is None:
                pred_scores = np.ones((pred_coords.shape[0], pred_coords.shape[1]),
                                      dtype=np.float32)
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
                assert 'bboxes' in gt, 'The ground truth data info do not ' \
                    'have the expected normalized_item ``"bbox"``.'
                bbox_size_ = np.max(gt['bboxes'][0][2:] - gt['bboxes'][0][:2])
                bbox_size = np.array([bbox_size_, bbox_size_]).reshape(-1, 2)
                result['bbox_size'] = bbox_size

            if 'head' in self.norm_item:
                assert 'head_size' in gt, 'The ground truth data info do ' \
                    'not have the expected normalized_item ``"head_size"``.'
                head_size_ = gt['head_size']
                head_size = np.array([head_size_, head_size_]).reshape(-1, 2)
                result['head_size'] = head_size

            if 'torso' in self.norm_item:
                if self.kpt_indexes is not None:
                    kpt_indexes = np.array(self.kpt_indexes)
                    torso_kpt_4_idx = np.where(kpt_indexes == 4)[0]
                    torso_kpt_5_idx = np.where(kpt_indexes == 5)[0]

                    if len(torso_kpt_4_idx) > 0 and len(torso_kpt_5_idx) > 0:
                        torso_size_ = np.linalg.norm(
                            gt_coords[0][torso_kpt_4_idx[0]] -
                            gt_coords[0][torso_kpt_5_idx[0]])
                        if torso_size_ < 1:
                            torso_size_ = np.linalg.norm(
                                pred_coords[0][torso_kpt_4_idx[0]] -
                                pred_coords[0][torso_kpt_5_idx[0]])
                    else:
                        torso_size_ = None
                else:
                    torso_size_ = np.linalg.norm(gt_coords[0][4] - gt_coords[0][5])
                    if torso_size_ < 1:
                        torso_size_ = np.linalg.norm(pred_coords[0][4] -
                                                     pred_coords[0][5])

                if torso_size_ is not None:
                    torso_size = np.array([torso_size_, torso_size_]).reshape(-1, 2)
                    result['torso_size'] = torso_size
                else:
                    result['torso_size'] = None

            self.results.append(result)

    def _compute_from_norm_factor(self, pred_coords, gt_coords, mask, pred_scores, norm_factor):
        """Compute recall and far given normalization factor array [N,2]."""
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
        # valid positions
        valid = distances != -1  # [K, N]
        valid_t = valid.T  # [N, K]
        # correctness per sample/keypoint
        correct = (distances < self.thr).T & valid_t  # [N, K]
        incorrect = (~correct) & valid_t

        # pred_scores: [N, K]
        score_mask = pred_scores > self.score_threshold

        # Recall: portion of correct keypoints with high score
        num_correct = int(correct.sum())
        recall = float(((correct) & score_mask).sum() / num_correct) if num_correct > 0 else 0.0

        # FAR (False Alarm Rate): portion of high-score detections that are incorrect
        # FAR = (Incorrect & High-score) / Total High-score detections
        num_high_score = int(score_mask.sum())
        far = float(((incorrect) & score_mask).sum() / num_high_score) if num_high_score > 0 else 0.0

        return recall, far

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        pred_coords = np.concatenate([r['pred_coords'] for r in results])
        gt_coords = np.concatenate([r['gt_coords'] for r in results])
        mask = np.concatenate([r['mask'] for r in results])
        pred_scores = np.concatenate([r['pred_scores'] for r in results])

        metrics = dict()
        metric_prefix = ' (filtered by kpt_indexes)' if self.kpt_indexes else ''

        if 'bbox' in self.norm_item:
            norm_size_bbox = np.concatenate([r['bbox_size'] for r in results])
            logger.info(f'Evaluating {self.__class__.__name__} '
                        f'(normalized by ``"bbox_size"``){metric_prefix}...')
            recall, far = self._compute_from_norm_factor(pred_coords, gt_coords, mask, pred_scores, norm_size_bbox)
            metrics['Recall'] = recall
            metrics['FAR'] = far

        if 'head' in self.norm_item:
            norm_size_head = np.concatenate([r['head_size'] for r in results])
            logger.info(f'Evaluating {self.__class__.__name__} '
                        f'(normalized by ``"head_size"``){metric_prefix}...')
            recall, far = self._compute_from_norm_factor(pred_coords, gt_coords, mask, pred_scores, norm_size_head)
            metrics['Recallh'] = recall
            metrics['FARh'] = far

        if 'torso' in self.norm_item:
            valid_torso_results = [r for r in results if r.get('torso_size') is not None]
            if valid_torso_results:
                norm_size_torso = np.concatenate([r['torso_size'] for r in valid_torso_results])
                valid_indices = [i for i, r in enumerate(results) if r.get('torso_size') is not None]
                valid_pred_coords = pred_coords[valid_indices]
                valid_gt_coords = gt_coords[valid_indices]
                valid_mask = mask[valid_indices]
                valid_pred_scores = pred_scores[valid_indices]

                logger.info(f'Evaluating {self.__class__.__name__} '
                            f'(normalized by ``"torso_size"``){metric_prefix}...')
                recall, far = self._compute_from_norm_factor(valid_pred_coords, valid_gt_coords, valid_mask, valid_pred_scores, norm_size_torso)
                metrics['Recallt'] = recall
                metrics['FARt'] = far

        return metrics


@METRICS.register_module()
class FAR_atraf(Recall_Atraf):
    """Wrapper that returns only FAR metrics keys."""

    def compute_metrics(self, results: list) -> Dict[str, float]:
        metrics = super().compute_metrics(results)
        far_metrics = {k: v for k, v in metrics.items() if k.startswith('FAR')}
        return far_metrics

