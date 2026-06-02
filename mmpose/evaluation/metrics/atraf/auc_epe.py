# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from mmpose.evaluation.metrics import AUC, EPE
from mmpose.evaluation.functional import keypoint_auc, keypoint_epe
from mmpose.evaluation.functional.keypoint_eval import _calc_distances, _distance_acc


@METRICS.register_module()
class AtrafAUC(AUC):
    """AUC evaluation metric with keypoint filtering support.

    This extends AUC to support filtering keypoints by index, allowing
    evaluation on only specific keypoints of interest.

    Args:
        norm_factor (float): AUC normalization factor. Default: 30 (pixels).
        num_thrs (int): number of thresholds to calculate auc. Default: 20.
        kpt_indexes (Sequence[int], optional): Indices of keypoints to include
            in the evaluation. If None, all keypoints are included.
            Default: ``None``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Default: ``'cpu'``.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Default: ``None``.

    Examples:

        >>> from mmpose.evaluation.metrics.atraf import AtrafAUC
        >>> import numpy as np
        >>> from mmengine.structures import InstanceData
        >>> num_keypoints = 17
        >>> keypoints = np.random.random((1, num_keypoints, 2)) * 10
        >>> gt_instances = InstanceData()
        >>> gt_instances.keypoints = keypoints
        >>> gt_instances.keypoints_visible = np.ones(
        ...     (1, num_keypoints, 1)).astype(bool)
        >>> pred_instances = InstanceData()
        >>> pred_instances.keypoints = keypoints
        >>> data_sample = {
        ...     'gt_instances': gt_instances.to_dict(),
        ...     'pred_instances': pred_instances.to_dict(),
        ... }
        >>> data_samples = [data_sample]
        >>> data_batch = [{'inputs': None}]
        >>> # Evaluate only keypoints 0-4
        >>> auc_metric = AtrafAUC(norm_factor=30, num_thrs=20,
        ...                        kpt_indexes=[0, 1, 2, 3, 4])
        >>> auc_metric.process(data_batch, data_samples)
        >>> auc_metric.evaluate(1)
    """

    def __init__(self,
                 norm_factor: float = 30,
                 num_thrs: int = 20,
                 kpt_indexes: Optional[Sequence[int]] = None,
                 twin_keypoints: Optional[Sequence[Sequence[int]]] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(
            norm_factor=norm_factor,
            num_thrs=num_thrs,
            collect_device=collect_device,
            prefix=prefix)
        self.kpt_indexes = kpt_indexes
        self.twin_keypoints = twin_keypoints

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples with keypoint filtering.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # predicted keypoints coordinates, [1, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['keypoints']
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['keypoints_visible'].astype(bool)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.reshape(1, -1)

            # ATRAF: Filter keypoints by indices if specified
            if self.kpt_indexes is not None:
                kpt_indexes = np.array(self.kpt_indexes)
                pred_coords = pred_coords[:, kpt_indexes, :]
                gt_coords = gt_coords[:, kpt_indexes, :]
                mask = mask[:, kpt_indexes]

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
            }

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate(
            [result['pred_coords'] for result in results])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])

        metric_prefix = ' (filtered by kpt_indexes)' if self.kpt_indexes else ''
        logger.info(f'Evaluating {self.__class__.__name__}{metric_prefix}...')

        metric_prefix = ' (filtered by kpt_indexes)' if self.kpt_indexes else ''
        logger.info(f'Evaluating {self.__class__.__name__}{metric_prefix}...')

        if self.twin_keypoints is None:
            auc = keypoint_auc(pred_coords, gt_coords, mask, self.norm_factor,
                               self.num_thrs)
        else:
            # Compute AUC by computing avg_acc at multiple thresholds using combined distances
            nor = np.tile(np.array([[self.norm_factor, self.norm_factor]]), (pred_coords.shape[0], 1))
            # distances per keypoint: [K, N]
            distances_orig = _calc_distances(pred_coords, gt_coords, mask, nor)
            distances_combined = distances_orig.copy()
            for pair in self.twin_keypoints:
                i, j = pair
                gt_swapped = gt_coords.copy()
                gt_swapped[:, i, :] = gt_coords[:, j, :]
                gt_swapped[:, j, :] = gt_coords[:, i, :]
                mask_swapped = mask.copy()
                mask_swapped[:, i] = mask[:, i] | mask[:, j]
                mask_swapped[:, j] = mask[:, j] | mask[:, i]
                distances_swapped = _calc_distances(pred_coords, gt_swapped, mask_swapped, nor)
                for idx in (i, j):
                    orig = distances_orig[idx]
                    swp = distances_swapped[idx]
                    combined = np.where((orig != -1) & (swp != -1), np.minimum(orig, swp), np.where(orig != -1, orig, swp))
                    distances_combined[idx] = combined

            thrs = [1.0 * i / self.num_thrs for i in range(self.num_thrs)]
            avg_accs = []
            for thr in thrs:
                acc = np.array([_distance_acc(distances_combined[k], thr) for k in range(distances_combined.shape[0])])
                valid_acc = acc[acc >= 0]
                avg_accs.append(valid_acc.mean() if len(valid_acc) > 0 else 0.0)

            auc = sum(avg_accs) / self.num_thrs

        metrics = dict()
        metrics['AUC'] = auc

        return metrics


@METRICS.register_module()
class AtrafEPE(EPE):
    """EPE evaluation metric with keypoint filtering support.

    This extends EPE to support filtering keypoints by index, allowing
    evaluation on only specific keypoints of interest.

    Args:
        kpt_indexes (Sequence[int], optional): Indices of keypoints to include
            in the evaluation. If None, all keypoints are included.
            Default: ``None``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Default: ``'cpu'``.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Default: ``None``.

    Examples:

        >>> from mmpose.evaluation.metrics.atraf import AtrafEPE
        >>> import numpy as np
        >>> from mmengine.structures import InstanceData
        >>> num_keypoints = 17
        >>> keypoints = np.random.random((1, num_keypoints, 2)) * 10
        >>> gt_instances = InstanceData()
        >>> gt_instances.keypoints = keypoints
        >>> gt_instances.keypoints_visible = np.ones(
        ...     (1, num_keypoints, 1)).astype(bool)
        >>> pred_instances = InstanceData()
        >>> pred_instances.keypoints = keypoints
        >>> data_sample = {
        ...     'gt_instances': gt_instances.to_dict(),
        ...     'pred_instances': pred_instances.to_dict(),
        ... }
        >>> data_samples = [data_sample]
        >>> data_batch = [{'inputs': None}]
        >>> # Evaluate only keypoints 0-4
        >>> epe_metric = AtrafEPE(kpt_indexes=[0, 1, 2, 3, 4])
        >>> epe_metric.process(data_batch, data_samples)
        >>> epe_metric.evaluate(1)
    """

    def __init__(self,
                 kpt_indexes: Optional[Sequence[int]] = None,
                 twin_keypoints: Optional[Sequence[Sequence[int]]] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(
            collect_device=collect_device,
            prefix=prefix)
        self.kpt_indexes = kpt_indexes
        self.twin_keypoints = twin_keypoints

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples with keypoint filtering.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # predicted keypoints coordinates, [1, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['keypoints']
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['keypoints_visible'].astype(bool)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.reshape(1, -1)

            # ATRAF: Filter keypoints by indices if specified
            if self.kpt_indexes is not None:
                kpt_indexes = np.array(self.kpt_indexes)
                pred_coords = pred_coords[:, kpt_indexes, :]
                gt_coords = gt_coords[:, kpt_indexes, :]
                mask = mask[:, kpt_indexes]

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
            }

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate(
            [result['pred_coords'] for result in results])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])

        metric_prefix = ' (filtered by kpt_indexes)' if self.kpt_indexes else ''
        logger.info(f'Evaluating {self.__class__.__name__}{metric_prefix}...')

        if self.twin_keypoints is None:
            epe = keypoint_epe(pred_coords, gt_coords, mask)
        else:
            # compute combined distances using normalization factor of ones
            norm_ones = np.ones((pred_coords.shape[0], pred_coords.shape[2]), dtype=np.float32)
            distances_orig = _calc_distances(pred_coords, gt_coords, mask, norm_ones)
            distances_combined = distances_orig.copy()
            for pair in self.twin_keypoints:
                i, j = pair
                gt_swapped = gt_coords.copy()
                gt_swapped[:, i, :] = gt_coords[:, j, :]
                gt_swapped[:, j, :] = gt_coords[:, i, :]
                mask_swapped = mask.copy()
                mask_swapped[:, i] = mask[:, i] | mask[:, j]
                mask_swapped[:, j] = mask[:, j] | mask[:, i]
                distances_swapped = _calc_distances(pred_coords, gt_swapped, mask_swapped, norm_ones)
                for idx in (i, j):
                    orig = distances_orig[idx]
                    swp = distances_swapped[idx]
                    combined = np.where((orig != -1) & (swp != -1), np.minimum(orig, swp), np.where(orig != -1, orig, swp))
                    distances_combined[idx] = combined
            distance_valid = distances_combined[distances_combined != -1]
            epe = distance_valid.sum() / max(1, len(distance_valid))

        metrics = dict()
        metrics['EPE'] = epe

        return metrics

