# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log

from mmpose.registry import METRICS
from mmpose.structures import PoseDataSample


@METRICS.register_module()
class ClassificationMetric(BaseMetric):
    """Classification metrics for auxiliary classifiers.

    Computes per-classifier metrics including accuracy, precision, recall, and F1-score.

    Args:
        classifiers (List[Dict]): List of classifier configurations, each containing:
            - field_name (str): Name of the classifier (e.g., 'gender', 'shape')
            - num_classes (int): Number of classes
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be inserted into all metrics
            names when logging the results. E.g. the prefix 'mAP' will result
            in metrics named 'mAP/precision', 'mAP/recall', etc. Defaults to None.
    """

    rule = 'greater'
    greater_keys = ['accuracy', 'precision', 'recall', 'f1']

    def __init__(self,
                 classifiers: List[Dict],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.classifiers = classifiers
        self.classifier_names = [c['field_name'] for c in classifiers]
        # Initialize results storage
        self.classifier_results = {
            c['field_name']: {
                'pred_classes': [],
                'gt_classes': []
            }
            for c in classifiers
        }

    def process(self, data_batch: dict,
                 data_samples: Sequence[PoseDataSample]) -> None:
        """Process one batch of data samples.

        Args:
            data_batch (dict): A batch of data that contains meta information
                of batched data samples.
            data_samples (Sequence[PoseDataSample]): The output of model forward.
        """
        for idx, data_sample in enumerate(data_samples):
            # Handle both dict and PoseDataSample objects
            if isinstance(data_sample, dict):
                has_pred_classifiers = 'pred_classifiers' in data_sample
                pred_classifiers = data_sample.get('pred_classifiers', {})
            else:
                has_pred_classifiers = hasattr(data_sample, 'pred_classifiers')
                pred_classifiers = getattr(data_sample, 'pred_classifiers', {})

            if not has_pred_classifiers:
                continue

            # Extract classifier predictions and GT
            for field_name in self.classifier_names:
                if field_name not in pred_classifiers:
                    continue

                pred_info = pred_classifiers[field_name]

                self.classifier_results[field_name]['pred_classes'].append(
                    pred_info.get('pred_class', -1))

                if 'gt_class' in pred_info:
                    self.classifier_results[field_name]['gt_classes'].append(
                        pred_info['gt_class'])

        # Append to self.results list (required by BaseMetric for distributed training)
        # Each element represents processed data from one sample
        self.results.append({'processed': len(data_samples)})

    def compute_metrics(self, results: dict = None) -> Dict[str, float]:
        """Compute metrics from collected results.

        Args:
            results (dict, optional): Ignored, uses self.classifier_results instead.

        Returns:
            Dict[str, float]: Dictionary of computed metrics.
        """
        metrics = {}

        for field_name in self.classifier_names:
            pred_classes = np.array(self.classifier_results[field_name]['pred_classes'])
            gt_classes = np.array(self.classifier_results[field_name]['gt_classes'])

            if len(gt_classes) == 0:
                print_log(f'No ground truth found for {field_name}',
                         logger='current')
                continue

            # Overall accuracy
            accuracy = np.mean(pred_classes == gt_classes)
            metrics[f'classifier/{field_name}/accuracy'] = float(accuracy)

            # Find classifier config to get number of classes
            num_classes = None
            for clf in self.classifiers:
                if clf['field_name'] == field_name:
                    num_classes = clf['num_classes']
                    break

            if num_classes is None:
                continue

            # Per-class metrics
            precisions = []
            recalls = []
            f1_scores = []

            for class_idx in range(num_classes):
                tp = np.sum((pred_classes == class_idx) &
                           (gt_classes == class_idx))
                fp = np.sum((pred_classes == class_idx) &
                           (gt_classes != class_idx))
                fn = np.sum((pred_classes != class_idx) &
                           (gt_classes == class_idx))

                precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                f1 = float(2 * (precision * recall) / (precision + recall)) \
                    if (precision + recall) > 0 else 0.0

                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

            # Macro averages
            precision_macro = float(np.mean(precisions))
            recall_macro = float(np.mean(recalls))
            f1_macro = float(np.mean(f1_scores))

            metrics[f'classifier/{field_name}/precision_macro'] = precision_macro
            metrics[f'classifier/{field_name}/recall_macro'] = recall_macro
            metrics[f'classifier/{field_name}/f1_macro'] = f1_macro

            # Per-class metrics
            for class_idx in range(num_classes):
                metrics[f'classifier/{field_name}/precision_class_{class_idx}'] = \
                    precisions[class_idx]
                metrics[f'classifier/{field_name}/recall_class_{class_idx}'] = \
                    recalls[class_idx]
                metrics[f'classifier/{field_name}/f1_class_{class_idx}'] = \
                    f1_scores[class_idx]

        # Reset for next evaluation epoch.
        for field_name in self.classifier_names:
            self.classifier_results[field_name] = {
                'pred_classes': [], 'gt_classes': []}

        return metrics


