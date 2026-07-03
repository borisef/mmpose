# Copyright (c) OpenMMLab. All rights reserved.
"""Shared base class for ATRAF score-threshold keypoint metrics.

Both :class:`Recall_Atraf` and :class:`Success_Rate` derive their outputs from a
2x2 contingency table (correct/wrong x high-score/low-score) computed at a given
score threshold. This module holds the single source of truth for:

  - ``process()``: extract pred/gt/mask/scores and normalization factors, with
    keypoint filtering, out-of-image / out-of-bbox masking and torso sizing.
  - ``_distances()``: distance computation including twin-keypoint handling.
  - ``_counts_at_threshold()``: the ``(A, B, C, D, N)`` contingency counts.
"""
from typing import Optional, Sequence, Union

import numpy as np

from mmpose.evaluation.functional.keypoint_eval import _calc_distances
from mmpose.evaluation.metrics import PCKAccuracy


class _AtrafScoreMetricBase(PCKAccuracy):
    """Base class providing shared processing and contingency counting.

    Args:
        thr (float): Threshold of PCK calculation. Default: 0.05.
        norm_item (str | Sequence[str]): Item used for normalization. Valid
            items include 'bbox', 'head', 'torso'. Default: ``'bbox'``.
        kpt_indexes (Sequence[int], optional): Indices of keypoints to include.
            If None, all keypoints are included. Default: ``None``.
        ignore_gt_out_of_image (bool): Mask out GT keypoints outside the image.
        ignore_gt_out_of_bbox (bool): Mask out GT keypoints outside the bbox.
        torso_keypoint_indexes (Sequence[int], optional): Keypoint pair used for
            torso normalization. Default: ``[4, 5]``.
        collect_device (str): Device used for collecting results. Default: 'cpu'.
        prefix (str, optional): Metric prefix. Default: ``None``.
        twin_keypoints (Sequence[Sequence[int]], optional): Symmetric keypoint
            pairs for min-distance matching. Default: ``None``.
    """

    def __init__(self,
                 thr: float = 0.05,
                 norm_item: Union[str, Sequence[str]] = 'bbox',
                 kpt_indexes: Optional[Sequence[int]] = None,
                 ignore_gt_out_of_image: bool = False,
                 ignore_gt_out_of_bbox: bool = False,
                 torso_keypoint_indexes: Optional[Sequence[int]] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 twin_keypoints: Optional[Sequence[Sequence[int]]] = None
                 ) -> None:
        super().__init__(
            thr=thr,
            norm_item=norm_item,
            collect_device=collect_device,
            prefix=prefix)
        self.kpt_indexes = kpt_indexes
        self.twin_keypoints = twin_keypoints
        self.ignore_gt_out_of_image = ignore_gt_out_of_image
        self.ignore_gt_out_of_bbox = ignore_gt_out_of_bbox
        self.torso_keypoint_indexes = torso_keypoint_indexes or [4, 5]

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Store pred/gt/mask/scores and normalization factors per sample."""
        for data_sample in data_samples:
            pred_coords = data_sample['pred_instances']['keypoints']
            pred_scores = data_sample['pred_instances'].get(
                'keypoint_scores', None)

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
                pred_scores = np.ones(
                    (pred_coords.shape[0], pred_coords.shape[1]),
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
                tk0, tk1 = self.torso_keypoint_indexes[0], \
                    self.torso_keypoint_indexes[1]
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
                    torso_size_ = np.linalg.norm(
                        gt_coords[0][tk0] - gt_coords[0][tk1])
                    if torso_size_ < 1:
                        torso_size_ = np.linalg.norm(
                            pred_coords[0][tk0] - pred_coords[0][tk1])

                if torso_size_ is not None:
                    torso_size = np.array(
                        [torso_size_, torso_size_]).reshape(-1, 2)
                    result['torso_size'] = torso_size
                else:
                    result['torso_size'] = None

            self.results.append(result)

    def _distances(self, pred_coords, gt_coords, mask, norm_factor):
        """Compute [K, N] distances, applying twin-keypoint min matching."""
        distances = _calc_distances(pred_coords, gt_coords, mask, norm_factor)
        if self.twin_keypoints is None:
            return distances

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
            distances_swapped = _calc_distances(
                pred_coords, gt_swapped, mask_swapped, norm_factor)
            for idx in (i, j):
                orig = distances_orig[idx]
                swp = distances_swapped[idx]
                combined = np.where(
                    (orig != -1) & (swp != -1), np.minimum(orig, swp),
                    np.where(orig != -1, orig, swp))
                distances_combined[idx] = combined
        return distances_combined

    def _counts_at_threshold(self, pred_coords, gt_coords, mask, pred_scores,
                             norm_factor, score_threshold):
        """Return the ``(A, B, C, D, N)`` contingency counts.

        - A: correct  & high-score  (correct_high)
        - B: wrong    & high-score  (incorrect_high)
        - C: correct  & low-score   (correct_low)
        - D: wrong    & low-score   (wrong_low)
        - N: total valid keypoints (A + B + C + D)

        A prediction is "correct" when its normalized distance is < ``thr`` and
        "high-score" when its keypoint score is > ``score_threshold``.
        """
        distances = self._distances(pred_coords, gt_coords, mask, norm_factor)
        valid = distances != -1  # [K, N]
        valid_t = valid.T  # [N, K]
        correct = (distances < self.thr).T & valid_t  # [N, K]
        incorrect = (~correct) & valid_t

        score_mask = pred_scores > score_threshold  # [N, K]
        high = score_mask & valid_t
        low = (~score_mask) & valid_t

        A = int((correct & high).sum())
        B = int((incorrect & high).sum())
        C = int((correct & low).sum())
        D = int((incorrect & low).sum())
        N = int(valid_t.sum())
        return A, B, C, D, N

    @staticmethod
    def _prf(A, B, C):
        """Precision, recall and F1 from contingency counts A, B, C."""
        num_correct = A + C
        num_high = A + B
        recall = float(A / num_correct) if num_correct > 0 else 0.0
        precision = float(A / num_high) if num_high > 0 else 0.0
        if (precision + recall) > 0.0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = 0.0
        return precision, recall, f1

    def _concat_norm_inputs(self, results, norm_key):
        """Concatenate pred/gt/mask/scores and the given normalization factor.

        Returns ``(pred_coords, gt_coords, mask, pred_scores, norm_factor)`` or
        ``None`` when no valid results exist for the requested norm item (e.g.
        torso size unavailable for every sample).
        """
        if norm_key == 'torso_size':
            valid = [r for r in results if r.get('torso_size') is not None]
            if not valid:
                return None
            idx = [i for i, r in enumerate(results)
                   if r.get('torso_size') is not None]
            pred_coords = np.concatenate(
                [r['pred_coords'] for r in results])[idx]
            gt_coords = np.concatenate([r['gt_coords'] for r in results])[idx]
            mask = np.concatenate([r['mask'] for r in results])[idx]
            pred_scores = np.concatenate(
                [r['pred_scores'] for r in results])[idx]
            norm_factor = np.concatenate([r['torso_size'] for r in valid])
            return pred_coords, gt_coords, mask, pred_scores, norm_factor

        pred_coords = np.concatenate([r['pred_coords'] for r in results])
        gt_coords = np.concatenate([r['gt_coords'] for r in results])
        mask = np.concatenate([r['mask'] for r in results])
        pred_scores = np.concatenate([r['pred_scores'] for r in results])
        norm_factor = np.concatenate([r[norm_key] for r in results])
        return pred_coords, gt_coords, mask, pred_scores, norm_factor
