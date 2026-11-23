# Copyright (c) OpenMMLab. All rights reserved.
"""Wrapper transform that exposes the rf-detr custom albumentations augmentations
inside an mmpose pipeline.

The augmentation classes themselves are NOT copied into mmpose. They are imported
from the rf-detr project (single source of truth) and injected into the
``albumentations`` namespace via ``register_custom_augmentations()``. The stock
mmpose :class:`~mmpose.datasets.transforms.Albumentation` wrapper then resolves
them by name (``getattr(albumentations, type)``), including inside ``OneOf`` /
``SomeOf`` containers.

The one capability the stock wrapper lacks is passing a segmentation ``mask`` to
the augmentations (it only maps ``img`` -> ``image``). Several rf-detr augs
(``AtrafBurningSpot``, ``AtrafBurningSpotWithTrail``, ``AtrafRandomEllipseSpot``)
read a foreground mask from the albumentations data dict. This wrapper loads that
mask from a sibling ``mask/<filename>`` directory using the same convention as
rf-detr and passes it through.
"""
import importlib.util
import os
import os.path as osp
import threading

import numpy as np

from mmpose.registry import TRANSFORMS
from ..common_transforms import Albumentation

# Default location of the rf-detr project; overridable via env var or the
# ``rfdetr_path`` constructor argument.
_DEFAULT_RFDETR_PATH = '/home/borisef/projects/rf-detr'
_CUSTOM_AUG_RELPATH = osp.join('atraf', 'augmentations', 'custom_augmentations.py')

_register_lock = threading.Lock()
_registered = False


def _ensure_rfdetr_registered(rfdetr_path: str = None) -> None:
    """Import rf-detr's ``custom_augmentations`` module and inject its ``Atraf*``
    classes into the ``albumentations`` namespace. Idempotent.

    Args:
        rfdetr_path (str | None): Path to the rf-detr project root. Falls back to
            the ``RFDETR_ATRAF_PATH`` env var, then to ``_DEFAULT_RFDETR_PATH``.
    """
    global _registered
    if _registered:
        return
    with _register_lock:
        if _registered:
            return

        root = (rfdetr_path or os.environ.get('RFDETR_ATRAF_PATH')
                or _DEFAULT_RFDETR_PATH)
        module_path = osp.join(root, _CUSTOM_AUG_RELPATH)
        if not osp.isfile(module_path):
            raise FileNotFoundError(
                'Could not find rf-detr custom augmentations at '
                f'{module_path!r}. Set the rf-detr project root via the '
                '`rfdetr_path` argument of AtrafAlbumentationsWrapper or the '
                '`RFDETR_ATRAF_PATH` environment variable.')

        # Load the module directly by file path. rf-detr has no top-level
        # `atraf/__init__.py`, so a plain package import is not reliable, and
        # the module only depends on numpy / PIL / albumentations (no intra-atraf
        # imports), so file-based loading is safe.
        spec = importlib.util.spec_from_file_location(
            'rfdetr_custom_augmentations', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        module.register_custom_augmentations()
        _registered = True


@TRANSFORMS.register_module()
class AtrafAlbumentationsWrapper(Albumentation):
    """Run rf-detr ``Atraf*`` albumentations augmentations in an mmpose pipeline.

    Extends :class:`~mmpose.datasets.transforms.Albumentation` with:

    1. Automatic registration of the rf-detr custom augmentations (imported from
       the rf-detr project, not copied).
    2. Per-image mask loading following the rf-detr convention: for an image at
       ``<dir>/<name>.<ext>`` the mask is looked up at
       ``<dir>/<mask_subdir>/<name>.<ext>``. When found it is passed as ``mask``
       to the augmentations; when absent each augmentation falls back to its own
       no-mask behaviour (mask-dependent augs become no-ops).

    Because the mask must match the image resolution, place this transform
    **before** ``TopdownAffine`` (i.e. while the image is still at its original
    size).

    Required Keys:

    - img
    - img_path (optional; required for mask lookup)

    Modified Keys:

    - img

    Args:
        transforms (List[dict]): Albumentations transform configs, e.g.
            ``[dict(type='AtrafClipDynamicRange', p=0.3), ...]``. ``OneOf`` /
            ``SomeOf`` containers are supported.
        keymap (dict | None): key mapping from pipeline keys to albumentations
            keys. Defaults to ``{'img': 'image'}``.
        mask_subdir (str): Sibling directory holding per-image masks.
            Defaults to ``'mask'``.
        bgr2rgb (bool): Convert the mmpose BGR image to RGB before augmenting and
            back afterwards (rf-detr augs assume RGB). Defaults to True.
        rfdetr_path (str | None): rf-detr project root. See
            :func:`_ensure_rfdetr_registered`.
    """

    def __init__(self,
                 transforms,
                 keymap=None,
                 mask_subdir='mask',
                 bgr2rgb=True,
                 rfdetr_path=None):
        # Must register before super().__init__ builds the albumentations Compose,
        # so that the Atraf* class names resolve.
        _ensure_rfdetr_registered(rfdetr_path)
        super().__init__(transforms, keymap)
        self.mask_subdir = mask_subdir
        self.bgr2rgb = bgr2rgb

    def _load_mask(self, img_path, hw):
        """Load the sibling mask for ``img_path`` as an HxW uint8 array, or None."""
        if not img_path:
            return None
        directory, filename = osp.split(img_path)
        mask_path = osp.join(directory, self.mask_subdir, filename)
        if not osp.isfile(mask_path):
            return None

        from PIL import Image as PILImage
        mask = np.array(PILImage.open(mask_path).convert('L'))
        if mask.shape[:2] != tuple(hw):
            import cv2
            h, w = hw
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    def transform(self, results: dict) -> dict:
        """Apply the wrapped augmentations, passing a sibling mask when available."""
        img = results['img']
        if self.bgr2rgb:
            img = img[..., ::-1]
        img = np.ascontiguousarray(img)

        mask = self._load_mask(results.get('img_path'), img.shape[:2])

        kwargs = dict(image=img)
        if mask is not None:
            kwargs['mask'] = mask
        out = self.aug(**kwargs)['image']

        if self.bgr2rgb:
            out = out[..., ::-1]
        results['img'] = np.ascontiguousarray(out)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(transforms={self.transforms}, '
                     f'mask_subdir={self.mask_subdir}, bgr2rgb={self.bgr2rgb})')
        return repr_str
