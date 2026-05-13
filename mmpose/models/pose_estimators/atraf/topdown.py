# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from torch import Tensor

from mmpose.registry import MODELS
from mmpose.utils.typing import (InstanceList, PixelDataList, SampleList)
from ..topdown import TopdownPoseEstimator


@MODELS.register_module()
class AtrafTopdownPoseEstimator(TopdownPoseEstimator):
    """Top-down pose estimator with ATRAF custom attribute preservation.

    This estimator extends TopdownPoseEstimator to preserve custom attributes
    (such as pred_classifiers) that may be created by heads during prediction.
    These attributes are saved before calling add_pred_to_datasample() and
    restored after, ensuring they are not lost during data processing.

    Inherits all arguments from TopdownPoseEstimator.
    """

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results with preservation of custom attributes.

        This override wraps the parent's prediction pipeline with save/restore
        logic for custom attributes like pred_classifiers.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W)
            data_samples (List[:obj:`PoseDataSample`]): The batch data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results with
            custom attributes preserved.
        """
        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(inputs.flip(-1))
            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)

        preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)

        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None

        # ATRAF: Save custom attributes created by head.predict()
        # (e.g., pred_classifiers) which may be lost in add_pred_to_datasample
        custom_attrs = {}
        for idx, data_sample in enumerate(data_samples):
            custom_attrs[idx] = {}
            if hasattr(data_sample, 'pred_classifiers'):
                custom_attrs[idx]['pred_classifiers'] = data_sample.pred_classifiers

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)

        # ATRAF: Restore custom attributes after add_pred_to_datasample
        for idx, data_sample in enumerate(results):
            if idx in custom_attrs:
                for attr_name, attr_value in custom_attrs[idx].items():
                    setattr(data_sample, attr_name, attr_value)

        return results

