# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
from torch import Tensor

from mmpose.models.utils.tta import flip_coordinates
from mmpose.registry import MODELS
from mmpose.utils.typing import SampleList
from ..pose_lifter import PoseLifter


@MODELS.register_module()
class AtrafPoseLifter(PoseLifter):
    """Pose lifter with ATRAF custom attribute preservation.

    This estimator extends PoseLifter to preserve custom attributes
    (such as pred_classifiers) that may be created by heads during prediction.
    These attributes are saved before calling add_pred_to_datasample() and
    restored after, ensuring they are not lost during data processing.

    Inherits all arguments from PoseLifter.
    """

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results with preservation of custom attributes.

        This override wraps the parent's prediction pipeline with save/restore
        logic for custom attributes like pred_classifiers.

        Args:
            inputs (Tensor): Inputs with shape like (B, K, C, T).
            data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            list[:obj:`PoseDataSample`]: The pose estimation results with
            custom attributes preserved.
        """
        assert self.with_head, (
            'The model must have head to perform prediction.')

        if self.test_cfg.get('flip_test', False):
            flip_indices = data_samples[0].metainfo['flip_indices']
            _feats = self.extract_feat(inputs)
            _feats_flip = self.extract_feat(
                torch.stack([
                    flip_coordinates(
                        _input,
                        flip_indices=flip_indices,
                        shift_coords=self.test_cfg.get('shift_coords', True),
                        input_size=(1, 1)) for _input in inputs
                ],
                            dim=0))

            feats = [_feats, _feats_flip]
        else:
            feats = self.extract_feat(inputs)

        pose_preds, batch_pred_instances, batch_pred_fields = None, None, None
        traj_preds, batch_traj_instances, batch_traj_fields = None, None, None
        if self.with_traj:
            x, traj_x = feats
            traj_preds = self.traj_head.predict(
                traj_x, data_samples, test_cfg=self.test_cfg)
        else:
            x = feats

        if self.with_head:
            pose_preds = self.head.predict(
                x, data_samples, test_cfg=self.test_cfg)

        if isinstance(pose_preds, tuple):
            batch_pred_instances, batch_pred_fields = pose_preds
        else:
            batch_pred_instances = pose_preds

        if isinstance(traj_preds, tuple):
            batch_traj_instances, batch_traj_fields = traj_preds
        else:
            batch_traj_instances = traj_preds

        # ATRAF: Save custom attributes created by head.predict()
        # (e.g., pred_classifiers) which may be lost in add_pred_to_datasample
        custom_attrs = {}
        for idx, data_sample in enumerate(data_samples):
            custom_attrs[idx] = {}
            if hasattr(data_sample, 'pred_classifiers'):
                custom_attrs[idx]['pred_classifiers'] = data_sample.pred_classifiers

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields,
                                              batch_traj_instances,
                                              batch_traj_fields, data_samples)

        # ATRAF: Restore custom attributes after add_pred_to_datasample
        for idx, data_sample in enumerate(results):
            if idx in custom_attrs:
                for attr_name, attr_value in custom_attrs[idx].items():
                    setattr(data_sample, attr_name, attr_value)

        return results




