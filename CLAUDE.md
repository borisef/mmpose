# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

### Installation
```bash
pip install -e .                # dev mode
pip install mmengine>=0.6.0,<1.0.0
pip install mmcv>=2.0.0rc4,<2.3.0
```

### Testing
```bash
pytest tests/ -v                                            # all tests
pytest tests/test_models/test_heads/test_heatmap_head.py -v # single file
pytest tests/ -k "test_topdown" -v                          # pattern match
coverage run --branch --source mmpose -m pytest tests/      # with coverage
```

Pytest is configured in `setup.cfg [pytest]` with xdoctest enabled. Excluded dirs: data, docs, .mim, tests/legacy.

### Linting
```bash
pre-commit run --all-files      # all hooks at once
flake8 mmpose/                  # lint (per-file ignores for configs/)
isort mmpose/                   # import sorting
yapf -i -r mmpose/              # code formatting (PEP8 base)
```

Formatting configs are in `setup.cfg` ([yapf], [isort], [flake8]).

## Architecture

MMPose v1.3.1 uses MMEngine's **Registry + Config** pattern. All components are registered with decorators (`@MODELS.register_module()`) and instantiated from config dicts (`dict(type='ResNet', depth=50)`).

### Registries (mmpose/registry.py)
`MODELS`, `DATASETS`, `TRANSFORMS`, `HOOKS`, `METRICS`, `KEYPOINT_CODECS`, `PARAM_SCHEDULERS`, `OPTIMIZERS`, `VISUALIZERS`, `INFERENCERS`, and others. Each is a child of the corresponding MMEngine registry.

### Core modules

| Package | Purpose |
|---|---|
| `models/pose_estimators/` | Top-level models: `TopdownPoseEstimator`, `BottomupPoseEstimator`, `PoseLifter` |
| `models/backbones/` | Feature extractors (ResNet, HRNet, Swin, etc.) |
| `models/necks/` | Feature aggregation (FPN variants) |
| `models/heads/` | Prediction heads: `HeatmapHead`, `RegressionHead`, `CoordClassificationHead` |
| `codecs/` | Encode/decode between annotations and model targets (`HeatmapLabel`, `SimCCLabel`, etc.) |
| `datasets/` | Dataset classes + transform pipelines |
| `evaluation/metrics/` | `CocoMetric`, `AicMetric`, etc. |
| `structures/` | `PoseDataSample` — central data container with `gt_instances`, `pred_instances`, `metainfo` |
| `apis/` | `MMPoseInferencer` for high-level inference |

### Config system
Configs are Python files using `_base_` inheritance. Key sections: `model`, `codec`, `train_pipeline`, `test_pipeline`, `train_dataloader`, `optim_wrapper`, `param_scheduler`, `default_hooks`, `val_evaluator`.

### Training data flow
```
Config → DataLoader(Dataset + Transforms) → Batch
→ model.loss(feats, data_samples) → loss dict → backprop
```

### Inference data flow
```
Image → model.extract_feat (backbone+neck) → feats
→ head.predict(feats, data_samples) → PoseDataSample with pred_instances
```

## ATRAF Extensions

Custom multi-task extensions under `atraf/` subdirectories:

**Head** (`mmpose/models/heads/heatmap_heads/atraf/heatmap_head_with_classifiers.py`):
- `HeatmapHeadWithClassifiers` — extends `HeatmapHead` with per-task classifier branches
- Each `ClassifierHead`: optional conv tower → adaptive avg pool → FC tower → logits
- Supports CrossEntropyLoss, BCEWithLogitsLoss, FocalLoss
- Per-sample weighting via `task_weights` in `raw_ann_info`
- Per-classifier LR scheduling via `lr_schedule` param (list of `(epoch, lr)` tuples)
- Predictions stored in `data_sample.pred_classifiers[field_name]`

**Hooks** (`mmpose/engine/hooks/atraf/`):
- `PoseVisualizationHookWithClassifiers` — renders classifier predictions on images with color-coded correctness
- `ClassifierLRSchedulerHook` — manages per-classifier optimizer param groups with epoch-based LR steps; freezes params when lr=0

**Pose estimators** (`mmpose/models/pose_estimators/atraf/`):
- `AtrafTopdownPoseEstimator`, `AtrafBottomupPoseEstimator`, `AtrafPoseLifter`
- Preserve custom attributes (like `pred_classifiers`) through `add_pred_to_datasample()`

**Structures** (`mmpose/structures/atraf/`):
- Custom `PoseDataSample` handling for ATRAF attributes

## Dependency Constraints
Strict version checks at import time in `mmpose/__init__.py`:
- MMEngine: >= 0.6.0, < 1.0.0
- MMCV: >= 2.0.0rc4, <= 2.3.0
- PyTorch: >= 1.8
