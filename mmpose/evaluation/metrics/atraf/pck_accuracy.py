# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union

import numpy as np
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from mmpose.evaluation.metrics import PCKAccuracy
from mmpose.evaluation.functional import keypoint_pck_accuracy


@METRICS.register_module()
class AtrafPCKAccuracy(PCKAccuracy):
    """PCK accuracy evaluation metric with keypoint filtering support.

    This extends PCKAccuracy to support filtering keypoints by index, allowing
    evaluation on only specific keypoints of interest.

    Args:
        thr(float): Threshold of PCK calculation. Default: 0.05.
        norm_item (str | Sequence[str]): The item used for normalization.
            Valid items include 'bbox', 'head', 'torso', which correspond
            to 'PCK', 'PCKh' and 'tPCK' respectively. Default: ``'bbox'``.
        kpt_indexes (Sequence[int], optional): Indices of keypoints to include
            in the evaluation. If None, all keypoints are included.
            Example: [0, 1, 2, 3] will include only the first 4 keypoints.
            Default: ``None``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Default: ``'cpu'``.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Default: ``None``.

    Examples:

        >>> from mmpose.evaluation.metrics.atraf import AtrafPCKAccuracy
        >>> import numpy as np
        >>> from mmengine.structures import InstanceData
        >>> num_keypoints = 17
        >>> keypoints = np.random.random((1, num_keypoints, 2)) * 10
        >>> gt_instances = InstanceData()
        >>> gt_instances.keypoints = keypoints
        >>> gt_instances.keypoints_visible = np.ones(
        ...     (1, num_keypoints, 1)).astype(bool)
        >>> gt_instances.bboxes = np.random.random((1, 4)) * 20
        >>> pred_instances = InstanceData()
        >>> pred_instances.keypoints = keypoints
        >>> data_sample = {
        ...     'gt_instances': gt_instances.to_dict(),
        ...     'pred_instances': pred_instances.to_dict(),
        ... }
        >>> data_samples = [data_sample]
        >>> data_batch = [{'inputs': None}]
        >>> # Evaluate only keypoints 0-4
        >>> pck_metric = AtrafPCKAccuracy(thr=0.5, norm_item='bbox',
        ...                                kpt_indexes=[0, 1, 2, 3, 4])
        >>> pck_metric.process(data_batch, data_samples)
        >>> pck_metric.evaluate(1)
    """

    def __init__(self,
                 thr: float = 0.05,
                 norm_item: Union[str, Sequence[str]] = 'bbox',
                 kpt_indexes: Optional[Sequence[int]] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(
            thr=thr,
            norm_item=norm_item,
            collect_device=collect_device,
            prefix=prefix)
        self.kpt_indexes = kpt_indexes

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

            if 'bbox' in self.norm_item:
                assert 'bboxes' in gt, 'The ground truth data info do not ' \
                    'have the expected normalized_item ``"bbox"``.'
                # ground truth bboxes, [1, 4]
                bbox_size_ = np.max(gt['bboxes'][0][2:] - gt['bboxes'][0][:2])
                bbox_size = np.array([bbox_size_, bbox_size_]).reshape(-1, 2)
                result['bbox_size'] = bbox_size

            if 'head' in self.norm_item:
                assert 'head_size' in gt, 'The ground truth data info do ' \
                    'not have the expected normalized_item ``"head_size"``.'
                # ground truth bboxes
                head_size_ = gt['head_size']
                head_size = np.array([head_size_, head_size_]).reshape(-1, 2)
                result['head_size'] = head_size

            if 'torso' in self.norm_item:
                # used in JhmdbDataset
                # ATRAF: When filtering keypoints, need to check if torso
                # keypoints (4 and 5) are in the filtered indices
                if self.kpt_indexes is not None:
                    kpt_indexes = np.array(self.kpt_indexes)
                    # Get indices of torso keypoints if they exist in filtered set
                    torso_kpt_4_idx = np.where(kpt_indexes == 4)[0]
                    torso_kpt_5_idx = np.where(kpt_indexes == 5)[0]

                    if len(torso_kpt_4_idx) > 0 and len(torso_kpt_5_idx) > 0:
                        # Use the filtered indices within the reduced keypoint set
                        torso_size_ = np.linalg.norm(
                            gt_coords[0][torso_kpt_4_idx[0]] -
                            gt_coords[0][torso_kpt_5_idx[0]])
                        if torso_size_ < 1:
                            torso_size_ = np.linalg.norm(
                                pred_coords[0][torso_kpt_4_idx[0]] -
                                pred_coords[0][torso_kpt_5_idx[0]])
                    else:
                        # Torso keypoints not in filtered set, skip torso norm
                        torso_size_ = None
                else:
                    # Original logic when no filtering
                    torso_size_ = np.linalg.norm(gt_coords[0][4] - gt_coords[0][5])
                    if torso_size_ < 1:
                        torso_size_ = np.linalg.norm(pred_coords[0][4] -
                                                     pred_coords[0][5])

                if torso_size_ is not None:
                    torso_size = np.array([torso_size_,
                                           torso_size_]).reshape(-1, 2)
                    result['torso_size'] = torso_size
                else:
                    # Mark torso_size as unavailable for this sample
                    result['torso_size'] = None

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

        metrics = dict()
        if 'bbox' in self.norm_item:
            norm_size_bbox = np.concatenate(
                [result['bbox_size'] for result in results])

            metric_prefix = ' (filtered by kpt_indexes)' if self.kpt_indexes else ''
            logger.info(f'Evaluating {self.__class__.__name__} '
                        f'(normalized by ``"bbox_size"``){metric_prefix}...')

            _, pck, _ = keypoint_pck_accuracy(pred_coords, gt_coords, mask,
                                              self.thr, norm_size_bbox)
            metrics['PCK'] = pck

        if 'head' in self.norm_item:
            norm_size_head = np.concatenate(
                [result['head_size'] for result in results])

            metric_prefix = ' (filtered by kpt_indexes)' if self.kpt_indexes else ''
            logger.info(f'Evaluating {self.__class__.__name__} '
                        f'(normalized by ``"head_size"``){metric_prefix}...')

            _, pckh, _ = keypoint_pck_accuracy(pred_coords, gt_coords, mask,
                                               self.thr, norm_size_head)
            metrics['PCKh'] = pckh

        if 'torso' in self.norm_item:
            # Filter results that have valid torso_size
            valid_torso_results = [r for r in results if r.get('torso_size') is not None]

            if valid_torso_results:
                norm_size_torso = np.concatenate(
                    [result['torso_size'] for result in valid_torso_results])

                # Get correspond pred/gt coords and mask for valid samples
                valid_indices = [i for i, r in enumerate(results) if r.get('torso_size') is not None]

                # Reconstruct pred_coords, gt_coords, mask from valid samples
                valid_pred_coords = pred_coords[valid_indices]
                valid_gt_coords = gt_coords[valid_indices]
                valid_mask = mask[valid_indices]

                metric_prefix = ' (filtered by kpt_indexes)' if self.kpt_indexes else ''
                logger.info(f'Evaluating {self.__class__.__name__} '
                            f'(normalized by ``"torso_size"``){metric_prefix}...')

                _, tpck, _ = keypoint_pck_accuracy(valid_pred_coords, valid_gt_coords,
                                                   valid_mask, self.thr, norm_size_torso)
                metrics['tPCK'] = tpck

        return metrics

