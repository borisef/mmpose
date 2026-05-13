# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine.utils import is_list_of

from mmpose.structures import PoseDataSample
from ..utils import merge_data_samples as base_merge_data_samples


def merge_data_samples(data_samples: List[PoseDataSample]) -> PoseDataSample:
    """Merge the given data samples with ATRAF custom attribute preservation.

    This function extends the base merge_data_samples to preserve custom
    attributes like pred_classifiers that may be present in the data samples.

    Args:
        data_samples (List[:obj:`PoseDataSample`]): The data samples to
            merge

    Returns:
        PoseDataSample: The merged data sample with custom attributes preserved.
    """
    # Call the base merge function
    merged = base_merge_data_samples(data_samples)

    # ATRAF: Preserve custom attributes like pred_classifiers from the first sample
    # These attributes are typically image-level predictions (not instance-level)
    # so we preserve them from the first sample
    if len(data_samples) > 0 and hasattr(data_samples[0], 'pred_classifiers'):
        merged.pred_classifiers = data_samples[0].pred_classifiers

    return merged

