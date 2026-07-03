# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, Optional, Sequence, Union

import numpy as np
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from mmpose.evaluation.metrics.atraf._chart_utils import (
    _get_chart_step, _save_image_to_tensorboard, _TB_AVAILABLE, plt)
from mmpose.evaluation.metrics.atraf._score_base import _AtrafScoreMetricBase


@METRICS.register_module()
class Success_Rate(_AtrafScoreMetricBase):
    """ATRAF success-rate metrics with a best-combined threshold search.

    Uses the 2x2 contingency table (see :class:`_AtrafScoreMetricBase`) where
    A = correct & high-score, B = wrong & high-score, C = correct & low-score,
    D = wrong & low-score and N = A + B + C + D (total valid keypoints).

    At the fixed ``score_threshold`` it reports:

    - PD_Success_Rate  = A / N
    - FAR_Error_Rate   = B / N
    - Skip_Rate        = (C + D) / N
    - Combined_Success_Rate =
        [ w_FAR*(1 - FAR_Error_Rate) + w_skip*(1 - Skip_Rate)
          + w_PD*PD_Success_Rate ] / W,   W = w_FAR + w_skip + w_PD

    It also scans score thresholds in [0, 1] to report the best combined value
    (``Best_Combined_Success_Rate``) and the threshold that achieves it
    (``BestThreshold``).

    Args:
        thr (float): Threshold of PCK calculation. Default: 0.05.
        norm_item (str | Sequence[str]): Normalization item(s). Default: 'bbox'.
        kpt_indexes (Sequence[int], optional): Keypoint indices to include.
        score_threshold (float): Fixed "high-score" threshold. Default: 0.5.
        weight_FAR (float): Weight of the (1 - FAR_Error_Rate) term. Default: 1.
        weight_skip (float): Weight of the (1 - Skip_Rate) term. Default: 1.
        weight_PD (float): Weight of the PD_Success_Rate term. Default: 1.
        num_steps (int): Number of thresholds sampled in [0, 1] for the best
            combined search. Default: 101.
        generate_chart (bool): Save a Combined-vs-threshold chart. Default: False.
        chart_images_folder (str, optional): Folder for chart PNGs.
        collect_device (str): Collect device. Default: 'cpu'.
        prefix (str, optional): Metric prefix. Default: ``None``.
        twin_keypoints (Sequence[Sequence[int]], optional): Symmetric pairs.
    """

    def __init__(self,
                 thr: float = 0.05,
                 norm_item: Union[str, Sequence[str]] = 'bbox',
                 kpt_indexes: Optional[Sequence[int]] = None,
                 score_threshold: float = 0.5,
                 weight_FAR: float = 1.0,
                 weight_skip: float = 1.0,
                 weight_PD: float = 1.0,
                 num_steps: int = 101,
                 ignore_gt_out_of_image: bool = False,
                 ignore_gt_out_of_bbox: bool = False,
                 torso_keypoint_indexes: Optional[Sequence[int]] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 twin_keypoints: Optional[Sequence[Sequence[int]]] = None,
                 generate_chart: bool = False,
                 chart_images_folder: Optional[str] = None) -> None:
        super().__init__(
            thr=thr,
            norm_item=norm_item,
            kpt_indexes=kpt_indexes,
            ignore_gt_out_of_image=ignore_gt_out_of_image,
            ignore_gt_out_of_bbox=ignore_gt_out_of_bbox,
            torso_keypoint_indexes=torso_keypoint_indexes,
            collect_device=collect_device,
            prefix=prefix,
            twin_keypoints=twin_keypoints)
        self.score_threshold = score_threshold
        self.weight_FAR = float(weight_FAR)
        self.weight_skip = float(weight_skip)
        self.weight_PD = float(weight_PD)
        self.num_steps = int(num_steps)
        self.generate_chart = bool(generate_chart)
        self.chart_images_folder = chart_images_folder

    def _rates(self, A, B, C, D, N):
        """PD / FAR-error / skip rates from contingency counts."""
        if N <= 0:
            return 0.0, 0.0, 0.0
        pd = float(A / N)
        far_err = float(B / N)
        skip = float((C + D) / N)
        return pd, far_err, skip

    def _combined(self, pd, far_err, skip):
        """Weighted, normalized combined success rate."""
        W = self.weight_FAR + self.weight_skip + self.weight_PD
        if W <= 0:
            return 0.0
        return float((self.weight_FAR * (1.0 - far_err) +
                      self.weight_skip * (1.0 - skip) +
                      self.weight_PD * pd) / W)

    def _metrics_for_norm(self, inputs):
        """Compute the six success-rate values for one normalization item."""
        pred_coords, gt_coords, mask, pred_scores, norm_factor = inputs

        # Fixed-threshold rates + combined.
        A, B, C, D, N = self._counts_at_threshold(
            pred_coords, gt_coords, mask, pred_scores, norm_factor,
            self.score_threshold)
        pd, far_err, skip = self._rates(A, B, C, D, N)
        combined = self._combined(pd, far_err, skip)

        # Best combined over the threshold sweep.
        thresholds = np.linspace(0.0, 1.0, self.num_steps)
        best_c, best_t = -1.0, 0.0
        combined_curve = []
        for t in thresholds:
            a, b, c, d, n = self._counts_at_threshold(
                pred_coords, gt_coords, mask, pred_scores, norm_factor, t)
            cpd, cfar, cskip = self._rates(a, b, c, d, n)
            cval = self._combined(cpd, cfar, cskip)
            combined_curve.append(cval)
            if cval > best_c or (abs(cval - best_c) <= 1e-12 and t < best_t):
                best_c = cval
                best_t = float(t)

        values = {
            'PD_Success_Rate': pd,
            'FAR_Error_Rate': far_err,
            'Skip_Rate': skip,
            'Combined_Success_Rate': combined,
            'Best_Combined_Success_Rate': float(best_c),
            'BestThreshold': float(best_t),
        }
        chart = (thresholds, combined_curve, best_t, float(best_c))
        return values, chart

    def _generate_and_log_chart(self, thresholds, combined_curve, best_t,
                                best_c, norm_name):
        """Generate a Combined-vs-threshold chart and optionally log it."""
        if not self.generate_chart or plt is None:
            return
        logger = MMLogger.get_current_instance()
        try:
            out_folder = self.chart_images_folder or os.path.join(
                os.getcwd(), 'chart_images')
            os.makedirs(out_folder, exist_ok=True)
            name = f'CombinedSuccess_{norm_name}'
            step = int(self._chart_step) if hasattr(self, '_chart_step') \
                else None
            if step is not None:
                out_path = os.path.join(out_folder, f'{name}_step{step}.png')
            else:
                out_path = os.path.join(out_folder, f'{name}.png')

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(thresholds, combined_curve, '-o', markersize=3)
            ax.plot(best_t, best_c, 'ro', markersize=8)
            ax.annotate(
                f't={best_t:.3f}\nC={best_c:.3f}', xy=(best_t, best_c),
                xytext=(5, -15), textcoords='offset points')
            ax.set_xlabel('Score threshold')
            ax.set_ylabel('Combined Success Rate')
            ax.set_title(f'Combined Success Rate ({norm_name})')
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)
            logger.info(f'Saved Success_Rate chart to {out_path}')

            if _TB_AVAILABLE:
                gs = int(self._chart_step) if hasattr(self, '_chart_step') \
                    else None
                if _save_image_to_tensorboard(
                        out_path, out_folder,
                        f'{self.__class__.__name__}/{name}', global_step=gs):
                    logger.info(
                        f'Logged Success_Rate chart to TensorBoard '
                        f'(logdir={out_folder}, step={gs})')
                else:
                    logger.warning(
                        'Failed to write Success_Rate chart to TensorBoard.')
        except Exception as e:
            logger.warning(f'Failed to generate Success_Rate chart: {e}')

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        metrics = dict()
        metric_prefix = ' (filtered by kpt_indexes)' if self.kpt_indexes else ''

        if not hasattr(self, '_chart_step'):
            self._chart_step = 0
        chart_step = int(self._chart_step)
        self._chart_step = _get_chart_step(fallback=chart_step)

        norm_specs = [
            ('bbox', '', 'bbox_size', 'bbox'),
            ('head', 'h', 'head_size', 'head'),
            ('torso', 't', 'torso_size', 'torso'),
        ]
        for norm_item, suffix, norm_key, tag in norm_specs:
            if norm_item not in self.norm_item:
                continue
            inputs = self._concat_norm_inputs(results, norm_key)
            if inputs is None:
                continue
            logger.info(
                f'Evaluating {self.__class__.__name__} '
                f'(normalized by ``"{norm_key}"``){metric_prefix}...')
            values, chart = self._metrics_for_norm(inputs)
            for key, val in values.items():
                metrics[f'{key}{suffix}'] = val
            self._generate_and_log_chart(*chart, tag)

        self._chart_step = chart_step + 1
        return metrics
