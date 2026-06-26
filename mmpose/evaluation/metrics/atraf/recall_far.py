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
                 ignore_gt_out_of_image: bool = False,
                 ignore_gt_out_of_bbox: bool = False,
                 torso_keypoint_indexes: Optional[Sequence[int]] = None,
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
        self.ignore_gt_out_of_image = ignore_gt_out_of_image
        self.ignore_gt_out_of_bbox = ignore_gt_out_of_bbox
        self.torso_keypoint_indexes = torso_keypoint_indexes or [4, 5]

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
                tk0, tk1 = self.torso_keypoint_indexes[0], self.torso_keypoint_indexes[1]
                if self.kpt_indexes is not None:
                    kpt_indexes = np.array(self.kpt_indexes)
                    torso_kpt_0_idx = np.where(kpt_indexes == tk0)[0]
                    torso_kpt_1_idx = np.where(kpt_indexes == tk1)[0]

                    if len(torso_kpt_0_idx) > 0 and len(torso_kpt_1_idx) > 0:
                        torso_size_ = np.linalg.norm(
                            gt_coords[0][torso_kpt_0_idx[0]] -
                            gt_coords[0][torso_kpt_1_idx[0]])
                        if torso_size_ < 1:
                            torso_size_ = np.linalg.norm(
                                pred_coords[0][torso_kpt_0_idx[0]] -
                                pred_coords[0][torso_kpt_1_idx[0]])
                    else:
                        torso_size_ = None
                else:
                    torso_size_ = np.linalg.norm(gt_coords[0][tk0] - gt_coords[0][tk1])
                    if torso_size_ < 1:
                        torso_size_ = np.linalg.norm(pred_coords[0][tk0] -
                                                     pred_coords[0][tk1])

                if torso_size_ is not None:
                    torso_size = np.array([torso_size_, torso_size_]).reshape(-1, 2)
                    result['torso_size'] = torso_size
                else:
                    result['torso_size'] = None

            self.results.append(result)

    def _compute_from_norm_factor(self, pred_coords, gt_coords, mask, pred_scores, norm_factor):
        """Compute recall, FAR, precision and F1 given normalization factor array [N,2]."""
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
        correct_high = int(((correct) & score_mask).sum())
        recall = float(correct_high / num_correct) if num_correct > 0 else 0.0

        # FAR (False Alarm Rate): portion of high-score detections that are incorrect
        # FAR = (Incorrect & High-score) / Total High-score detections
        num_high_score = int(score_mask.sum())
        incorrect_high = int(((incorrect) & score_mask).sum())
        far = float(incorrect_high / num_high_score) if num_high_score > 0 else 0.0

        # Precision: portion of high-score detections that are correct
        precision = float(correct_high / num_high_score) if num_high_score > 0 else 0.0

        # F1: harmonic mean of precision and recall
        if (precision + recall) > 0.0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = 0.0

        # Accuracy: portion of all valid keypoints that are correct and high-score
        total_valid = int(valid_t.sum())
        accuracy = float(correct_high / total_valid) if total_valid > 0 else 0.0

        return recall, far, precision, f1, accuracy

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
            recall, far, precision, f1, accuracy = self._compute_from_norm_factor(pred_coords, gt_coords, mask, pred_scores, norm_size_bbox)
            metrics['Recall'] = recall
            metrics['FAR'] = far
            metrics['Precision'] = precision
            metrics['F1'] = f1
            metrics['Accuracy'] = accuracy

        if 'head' in self.norm_item:
            norm_size_head = np.concatenate([r['head_size'] for r in results])
            logger.info(f'Evaluating {self.__class__.__name__} '
                        f'(normalized by ``"head_size"``){metric_prefix}...')
            recall, far, precision, f1, accuracy = self._compute_from_norm_factor(pred_coords, gt_coords, mask, pred_scores, norm_size_head)
            metrics['Recallh'] = recall
            metrics['FARh'] = far
            metrics['Precisionh'] = precision
            metrics['F1h'] = f1
            metrics['Accuracyh'] = accuracy

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
                recall, far, precision, f1, accuracy = self._compute_from_norm_factor(valid_pred_coords, valid_gt_coords, valid_mask, valid_pred_scores, norm_size_torso)
                metrics['Recallt'] = recall
                metrics['FARt'] = far
                metrics['Precisiont'] = precision
                metrics['F1t'] = f1
                metrics['Accuracyt'] = accuracy

        return metrics


@METRICS.register_module()
class FAR_atraf(Recall_Atraf):
    """Wrapper that returns only FAR metrics keys."""

    def compute_metrics(self, results: list) -> Dict[str, float]:
        metrics = super().compute_metrics(results)
        far_metrics = {k: v for k, v in metrics.items() if k.startswith('FAR')}
        return far_metrics

