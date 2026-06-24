# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

import os
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from mmpose.structures import PoseDataSample

# reuse tensorboard helper from smart_f1 to ensure consistent logging
from mmpose.evaluation.metrics.atraf.smart_f1 import _save_image_to_tensorboard, _TB_AVAILABLE, plt, _get_chart_step


@METRICS.register_module()
class ClassificationMetricConfusionMatrix(BaseMetric):
    """Generate confusion matrices for auxiliary classifiers and log them.

    This metric collects predicted classes and ground-truth classes from
    auxiliary classifier outputs (field name and pred/gt entries in
    ``pred_classifiers`` of the model outputs) and produces:
      - per-classifier accuracy / macro precision/recall/F1 metrics (same as
        :class:`ClassificationMetric`)
      - a confusion matrix image saved as PNG to ``chart_images_folder`` and
        logged to TensorBoard (if available) when ``generate_chart`` is True.

    Args:
        classifiers (List[Dict]): List of classifier configs with keys
            ``field_name`` and ``num_classes``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Defaults to 'cpu'.
        prefix (str, optional): Metric prefix. Defaults to None.
        generate_chart (bool): Whether to save confusion matrix PNG and log to
            TensorBoard. Default: False.
        chart_images_folder (str, optional): Folder to save confusion PNGs.
            If None and generate_chart True, uses './chart_images'.
    """

    rule = 'greater'
    greater_keys = ['accuracy']

    def __init__(self,
                 classifiers: List[Dict],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 generate_chart: bool = False,
                 chart_images_folder: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.classifiers = classifiers
        self.classifier_names = [c['field_name'] for c in classifiers]
        # storage for predictions / gts
        self.classifier_results = {
            c['field_name']: {'pred_classes': [], 'gt_classes': []}
            for c in classifiers
        }
        self.generate_chart = bool(generate_chart)
        self.chart_images_folder = chart_images_folder

    def process(self, data_batch: dict, data_samples: Sequence[PoseDataSample]) -> None:
        for data_sample in data_samples:
            if isinstance(data_sample, dict):
                has_pred_classifiers = 'pred_classifiers' in data_sample
                pred_classifiers = data_sample.get('pred_classifiers', {})
            else:
                has_pred_classifiers = hasattr(data_sample, 'pred_classifiers')
                pred_classifiers = getattr(data_sample, 'pred_classifiers', {})

            if not has_pred_classifiers:
                continue

            for field_name in self.classifier_names:
                if field_name not in pred_classifiers:
                    continue
                pred_info = pred_classifiers[field_name]
                self.classifier_results[field_name]['pred_classes'].append(
                    pred_info.get('pred_class', -1))
                if 'gt_class' in pred_info:
                    self.classifier_results[field_name]['gt_classes'].append(
                        pred_info['gt_class'])

        # append a lightweight record for distributed aggregation
        self.results.append({'processed': len(data_samples)})

    def _generate_and_log_confusion(self, cm: np.ndarray, field_name: str):
        if not self.generate_chart or plt is None:
            return
        logger = MMLogger.get_current_instance()
        try:
            out_folder = self.chart_images_folder or os.path.join(os.getcwd(), 'chart_images')
            os.makedirs(out_folder, exist_ok=True)
            name = f'Confusion_{field_name}'
            step = int(self._chart_step) if hasattr(self, '_chart_step') else None
            if step is not None:
                out_path = os.path.join(out_folder, f'{name}_step{step}.png')
            else:
                out_path = os.path.join(out_folder, f'{name}.png')

            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix: {field_name}')

            # annotate cells with counts
            fmt = 'd'
            thresh = cm.max() / 2. if cm.max() > 0 else 0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(int(cm[i, j]), fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")

            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)
            logger.info(f'Saved confusion matrix to {out_path}')

            if _TB_AVAILABLE:
                gs = int(self._chart_step) if hasattr(self, '_chart_step') else None
                if _save_image_to_tensorboard(out_path, out_folder, f'{self.__class__.__name__}/{field_name}', global_step=gs):
                    logger.info(f'Logged confusion matrix to TensorBoard (logdir={out_folder}, step={gs})')
                else:
                    logger.warning('Failed to write confusion matrix to TensorBoard.')
        except Exception as e:
            logger.warning(f'Failed to generate confusion matrix: {e}')

    def compute_metrics(self, results: dict = None) -> Dict[str, float]:
        metrics = {}
        # chart step counter
        if not hasattr(self, '_chart_step'):
            self._chart_step = 0
        chart_step = int(self._chart_step)
        # Prefer the resume-safe training epoch as the chart/TensorBoard step;
        # keep the internal counter as a fallback for standalone test runs.
        self._chart_step = _get_chart_step(fallback=chart_step)

        for clf in self.classifiers:
            field_name = clf['field_name']
            num_classes = clf.get('num_classes', None)
            pred = np.array(self.classifier_results[field_name]['pred_classes'])
            gt = np.array(self.classifier_results[field_name]['gt_classes'])
            if len(gt) == 0:
                MMLogger.get_current_instance().warning(f'No ground truth found for classifier {field_name}')
                continue

            # overall accuracy
            accuracy = float(np.mean(pred == gt))
            metrics[f'classifier/{field_name}/accuracy'] = accuracy

            if num_classes is None:
                continue

            # compute confusion matrix: rows=true, cols=pred
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(gt, pred):
                if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
                    cm[int(t), int(p)] += 1

            # per-class precision/recall/f1 (macro)
            precisions = []
            recalls = []
            f1_scores = []
            for cls in range(num_classes):
                tp = int(cm[cls, cls])
                fp = int(cm[:, cls].sum() - tp)
                fn = int(cm[cls, :].sum() - tp)
                precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

            metrics[f'classifier/{field_name}/precision_macro'] = float(np.mean(precisions))
            metrics[f'classifier/{field_name}/recall_macro'] = float(np.mean(recalls))
            metrics[f'classifier/{field_name}/f1_macro'] = float(np.mean(f1_scores))

            # per-class values
            for cls in range(num_classes):
                metrics[f'classifier/{field_name}/precision_class_{cls}'] = precisions[cls]
                metrics[f'classifier/{field_name}/recall_class_{cls}'] = recalls[cls]
                metrics[f'classifier/{field_name}/f1_class_{cls}'] = f1_scores[cls]

            # generate and log confusion matrix image
            # attach chart step attribute used by helper
            self._generate_and_log_confusion(cm, field_name)

        # Reset for next evaluation epoch.
        for field_name in self.classifier_names:
            self.classifier_results[field_name] = {
                'pred_classes': [], 'gt_classes': []}

        # increment chart step for next invocation
        self._chart_step = chart_step + 1
        return metrics

