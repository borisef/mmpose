# ATRAF Custom Augmentations in MMPose

This document describes how to use the **rf-detr custom augmentations** inside an MMPose
training pipeline.

## Overview

The augmentations themselves are a set of albumentations `ImageOnlyTransform` subclasses that
live in the **rf-detr** project (`atraf/augmentations/custom_augmentations.py`). They are **not
copied** into MMPose — they are imported from rf-detr at runtime (single source of truth) and
injected into the `albumentations` namespace via rf-detr's `register_custom_augmentations()`.

MMPose drives them through a single wrapper transform:

- **`AtrafAlbumentationsWrapper`** — `mmpose/datasets/transforms/atraf/atraf_albumentations.py`

It subclasses MMPose's stock `Albumentation` transform
(`mmpose/datasets/transforms/common_transforms.py`), so it builds an `albumentations.Compose`
and resolves each transform **by name** (`getattr(albumentations, type)`), including nested
`OneOf` / `SomeOf` containers. On top of that it adds two things the stock wrapper lacks:

1. Automatic import + registration of the rf-detr `Atraf*` augmentations.
2. Per-image **mask** loading (rf-detr convention) so the mask-dependent augmentations work.

## Requirements

- **albumentations ≥ 1.4.x** in the training environment. The mask-dependent augmentations rely
  on the albumentations `get_params_dependent_on_data(params, data)` hook, which only exists from
  the 1.4 line onward. On albumentations 1.3.x the mask is silently never passed and
  `AtrafBurningSpot*` become no-ops even with a mask present.
  - albumentations 2.x requires Python ≥ 3.9. The `mm_python38` env is Python 3.8, so the newest
    installable release there is **1.4.18**, which is sufficient.
  - Optional: set `NO_ALBUMENTATIONS_UPDATE=1` to silence the "a new version is available" banner.
- **Pillow** (`PIL`) — used by several augmentations and by the wrapper's mask loader.
- The **rf-detr project must be present on disk**. The wrapper locates it, in order:
  1. the `rfdetr_path=` constructor argument,
  2. the `RFDETR_ATRAF_PATH` environment variable,
  3. the default `/home/borisef/projects/rf-detr`.

## Wrapper usage and options

`AtrafAlbumentationsWrapper` accepts:

| Argument | Default | Description |
|---|---|---|
| `transforms` | — | List of albumentations transform configs (dicts with `type` + params). `OneOf` / `SomeOf` containers are supported and may nest `Atraf*` transforms. |
| `keymap` | `{'img': 'image'}` | Mapping from pipeline keys to albumentations keys. |
| `mask_subdir` | `'mask'` | Sibling directory holding per-image masks (see below). |
| `bgr2rgb` | `True` | Convert MMPose's BGR image to RGB before augmenting and back afterwards (rf-detr augs assume RGB). Color-aware augs like `AtrafStyleTransfer` need this; intensity augs are unaffected. |
| `rfdetr_path` | `None` | rf-detr project root; see resolution order above. |

**Placement:** put `AtrafAlbumentationsWrapper` **after `LoadImage` / `GetBBoxCenterScale` but
before `TopdownAffine`**. The mask loaded from disk matches the original image resolution, which
only holds while the image has not yet been warped by `TopdownAffine`.

These are image-only augmentations — they do not modify keypoints or bounding boxes — so they are
safe to run before the geometric transforms.

## Mask convention

For a training image at `<dir>/<name>.<ext>`, the wrapper looks for a mask at
`<dir>/<mask_subdir>/<name>.<ext>` (default `<dir>/mask/<name>.<ext>`). The mask is loaded as a
grayscale image; pixels `> 0` are treated as **foreground**.

- If the mask file exists, it is passed to the augmentations as `mask=`.
- If it is missing (or `img_path` is unavailable), each augmentation falls back to its own
  no-mask behaviour:
  - `AtrafBurningSpot` / `AtrafBurningSpotWithTrail` → **no-op** (they require a mask),
  - `AtrafRandomEllipseSpot` → treats **all pixels as foreground**,
  - all other augmentations are image-only and unaffected.

Reference data layout (as used in rf-detr): `/home/borisef/data/large_rf_detr_data/airplane_07/`
with a sibling `mask/` folder containing one grayscale mask per image.

## Available augmentations

All `Atraf*` transforms are defined in rf-detr's `custom_augmentations.py`. Range parameters are
`(min, max)` pairs sampled uniformly per call. `p` is the per-transform apply probability.

| Transform | Effect | Key parameters | Mask |
|---|---|---|---|
| `AtrafLinearBrightnessGradient` | Linear brightness ramp across the image in a random direction (directional lighting). | `strength_range` (pixel-unit offset), `angle_range` (degrees) | unused |
| `AtrafRandomEllipseSpot` | Projector-like spotlight: a tilted Gaussian ellipse brightens the image; background responds weaker than foreground. | `intensity_range`, `sigma_x_range` / `sigma_y_range`, `background_amplification_ratio` | optional |
| `AtrafBurningSpot` | Generative SWIR fire hot-spot (white core + glow) grown by a stochastic fire model, placed at a random mask-foreground pixel. | `size2target`, `shape_params` (`n_seeds`, `fuel_contrast`, `anisotropy`), `color_params` (`core_fraction`, `diffusion_sigma_ratio`, `blend_alpha`) | required |
| `AtrafBurningSpotWithTrail` | `AtrafBurningSpot` plus a smoke trail with burning embers streaming behind a random flight direction; the fire elongates into the trail. | spot params above, plus `trail_params` (`angle`, `length_ratio`, `width_ratio`, `expansion`, `smoke_intensity`, `decay_length_ratio`, `turbulence`, `meander`, `n_particles`, `particle_size_ratio`, `particle_brightness`, `trail_p`) | required |
| `AtrafClipDynamicRange` | Clips the intensity histogram at random percentiles and stretches to [0, 255] per channel (exposure variation). | `low_percentile_range`, `high_percentile_range` | unused |
| `AtrafMaskedImageAugmentation` | Pastes/blends regions from a library of source images (with optional companion masks) onto the training image. | `images_dir`, `use_as` (`foreground` / `background` / `none`), `blend_alpha`, `scale_range` | unused* |
| `AtrafStyleTransfer` | Domain-adaptation style transfer: picks one random reference image from `reference_images_dir` and applies one randomly-selected method. | `reference_images_dir`, `methods` (subset of method names; `None` = all) | unused |
| `AtrafDebugImageSaver` | Debug only: saves the final augmented image to disk with an overlay of which augmentations fired. Put it last with `p=1.0`. | `save_dir`, `max_images` (round-robin slots) | unused |

\* `AtrafMaskedImageAugmentation`'s `use_as` flag controls its **internal source-image library**
(how maskless *library* images are treated), not the per-training-image mask handled by the
wrapper.

### `AtrafStyleTransfer` — available methods

| Method | Effect | Tunable defaults |
|---|---|---|
| `"FDA"` | Fourier Domain Adaptation — swaps the low-frequency spectrum from the reference image, preserving content structure. | `beta_limit=(0.0, 0.3)` |
| `"HistogramMatching"` | Matches the per-channel histogram of the input to the reference image, blended by a random ratio. | `blend_ratio=(0.7, 1.0)` |
| `"PixelDistributionAdaptation"` | Aligns the pixel distribution via PCA whitening to the reference image, then blends. | `blend_ratio=(0.1, 0.9)`, `transform_type="pca"` |

Use `methods=["FDA"]` to restrict to a single method, or any subset list. The reference folder is
loaded once at construction; a missing/empty folder becomes a no-op with a warning.

## Full config example

Taken from
`configs/atraf/borisef/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288_try_rfdetr_augmentations.py`:

```python
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    # rf-detr custom augmentations (imported from the rf-detr project, not copied).
    # Must run before TopdownAffine so that a sibling mask/<filename> (if present)
    # still matches the original-resolution image. Mask-dependent augs
    # (AtrafBurningSpot*, AtrafRandomEllipseSpot) are no-ops when no mask exists.
    dict(
        type='AtrafAlbumentationsWrapper',
        # rfdetr_path='/home/borisef/projects/rf-detr',  # or RFDETR_ATRAF_PATH env
        transforms=[
            dict(type='AtrafClipDynamicRange', p=0.3),
            dict(
                type='AtrafLinearBrightnessGradient',
                strength_range=[30.0, 80.0],
                angle_range=[0.0, 360.0],
                p=0.3),
            dict(
                type='SomeOf',
                transforms=[
                    dict(type='AtrafRandomEllipseSpot', p=1.0),
                    dict(type='AtrafBurningSpotWithTrail', p=1.0),
                ],
                n=1,
                p=0.3),
            # dict(type='AtrafStyleTransfer',
            #      reference_images_dir='<dir>', p=0.2),
        ]),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
```

To run this config, point `config_file` in `tools/atraf/borisef/train_wrap_mmpose_01.py` at the
`..._try_rfdetr_augmentations.py` file.

## Caveats

- The standard COCO `images/val2017/` directory has **no sibling `mask/` folder**, so
  `AtrafBurningSpot` / `AtrafBurningSpotWithTrail` are no-ops on that data; the image-only
  augmentations still apply. Provide per-image masks (as in the rf-detr reference layout) to
  exercise the mask-dependent augmentations.
- `AtrafStyleTransfer` requires a valid `reference_images_dir`; it is left commented out in the
  example config.
- `AtrafDebugImageSaver` relies on rf-detr's module-level augmentation tracking to list which
  transforms fired; without `enable_augmentation_tracking(...)` it still saves images but with an
  empty overlay.
- For the internals of the generative fire/smoke models, see rf-detr's own docs:
  `atraf/docs/burning-spot.md` and `atraf/docs/smoke-trail.md` (and `atraf/docs/augmentations.md`
  for the rf-detr `AUG_CONFIG` form).
