# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional, Sequence

import cv2
import mmcv
import mmengine
import mmengine.fileio as fileio
from mmengine.runner import Runner

from mmpose.registry import HOOKS
from mmpose.engine.hooks.visualization_hook import PoseVisualizationHook
from mmpose.structures import PoseDataSample, merge_data_samples


@HOOKS.register_module()
class PoseVisualizationHookWithClassifiers(PoseVisualizationHook):
    """Pose Estimation Visualization Hook with Classifier Support.

    This hook extends PoseVisualizationHook to also visualize classifier
    predictions (e.g., gender, shape) on validation and testing images.

    Inherits all parameters from PoseVisualizationHook.
    """

    def __init__(
        self,
        enable: bool = False,
        interval: int = 50,
        kpt_thr: float = 0.3,
        show: bool = False,
        wait_time: float = 0.,
        out_dir: Optional[str] = None,
        backend_args: Optional[dict] = None,
    ):
        super().__init__(
            enable=enable,
            interval=interval,
            kpt_thr=kpt_thr,
            show=show,
            wait_time=wait_time,
            out_dir=out_dir,
            backend_args=backend_args,
        )

    def _draw_text_on_image(self, img, text: str, position=(10, 30),
                            font_scale=0.7, thickness=2, color=(0, 255, 0)):
        """Draw text on image.

        Args:
            img (np.ndarray): Image in RGB format.
            text (str): Text to draw.
            position (tuple): (x, y) position for text.
            font_scale (float): Font scale.
            thickness (int): Text thickness.
            color (tuple): RGB color tuple (B, G, R format for OpenCV).

        Returns:
            np.ndarray: Image with text drawn.
        """
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Convert color from RGB to BGR
        color_bgr = (color[2], color[1], color[0])
        # Draw text
        cv2.putText(img_bgr, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color_bgr, thickness)
        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[PoseDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`PoseDataSample`]): Outputs from model.
        """
        if self.enable is False:
            return

        self._visualizer.set_dataset_meta(runner.val_evaluator.dataset_meta)

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path = data_batch['data_samples'][0].get('img_path')
        img_bytes = fileio.get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        data_sample = outputs[0]

        # revert the heatmap on the original image
        data_sample = merge_data_samples([data_sample])

        if total_curr_iter % self.interval == 0:
            # Draw text on image
            img = self._draw_text_on_image(img, "try text")

            self._visualizer.add_datasample(
                os.path.basename(img_path) if self.show else 'val_img',
                img,
                data_sample=data_sample,
                draw_gt=False,
                draw_bbox=True,
                draw_heatmap=True,
                show=self.show,
                wait_time=self.wait_time,
                kpt_thr=self.kpt_thr,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[PoseDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`PoseDataSample`]): Outputs from model.
        """
        if self.enable is False:
            return

        if self.out_dir is not None:
            self.out_dir = os.path.join(runner.work_dir, runner.timestamp,
                                        self.out_dir)
            mmengine.mkdir_or_exist(self.out_dir)

        self._visualizer.set_dataset_meta(runner.test_evaluator.dataset_meta)

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.get('img_path')
            img_bytes = fileio.get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            data_sample = merge_data_samples([data_sample])

            out_file = None
            if self.out_dir is not None:
                out_file_name, postfix = os.path.basename(img_path).rsplit(
                    '.', 1)
                index = len([
                    fname for fname in os.listdir(self.out_dir)
                    if fname.startswith(out_file_name)
                ])
                out_file = f'{out_file_name}_{index}.{postfix}'
                out_file = os.path.join(self.out_dir, out_file)

            # Draw text on image
            img = self._draw_text_on_image(img, "try text")

            self._visualizer.add_datasample(
                os.path.basename(img_path) if self.show else 'test_img',
                img,
                data_sample=data_sample,
                show=self.show,
                draw_gt=False,
                draw_bbox=True,
                draw_heatmap=True,
                wait_time=self.wait_time,
                kpt_thr=self.kpt_thr,
                out_file=out_file,
                step=self._test_index)

