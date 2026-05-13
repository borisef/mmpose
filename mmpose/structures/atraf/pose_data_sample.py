# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from mmengine.structures import InstanceData, PixelData

from mmpose.structures import MultilevelPixelData, PoseDataSample as BasePoseDataSample


class AtrafPoseDataSample(BasePoseDataSample):
    """ATRAF-enhanced Pose data sample with custom attribute preservation.

    This class extends PoseDataSample to preserve custom attributes
    (such as pred_classifiers) when converting to dictionary format.
    This is essential for evaluation metrics that may receive the
    result as a dictionary.

    Inherits all functionality from BasePoseDataSample.
    """

    def to_dict(self) -> dict:
        """Convert the data sample to a dictionary.

        This override ensures that custom attributes like pred_classifiers
        are preserved when converting to dict (e.g., in evaluation).

        Returns:
            dict: The converted dictionary representation.
        """
        # Call parent's to_dict() to get standard fields
        result = super().to_dict()

        # ATRAF: Preserve custom attributes like pred_classifiers
        if hasattr(self, 'pred_classifiers'):
            result['pred_classifiers'] = self.pred_classifiers

        return result

