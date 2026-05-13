# ATRAF PCKAccuracy with Keypoint Filtering

## Feature Overview

The `AtrafPCKAccuracy` metric extends the base `PCKAccuracy` to support filtering specific keypoints during evaluation.

## Parameters

- **thr** (float): Threshold of PCK calculation. Default: 0.05
- **norm_item** (str | Sequence[str]): Normalization method ('bbox', 'head', 'torso'). Default: 'bbox'
- **kpt_indexes** (Sequence[int], optional): Indices of keypoints to include. Default: None (all keypoints)
- **collect_device** (str): Device for collecting results ('cpu' or 'gpu'). Default: 'cpu'
- **prefix** (str, optional): Prefix for metric names. Default: None

## Usage Examples

### Example 1: Evaluate all keypoints (default behavior)
```python
val_evaluator = [
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox'
    )
]
```

### Example 2: Evaluate only specific keypoints
```python
# For COCO (17 keypoints), evaluate only upper body (head, shoulders, elbows, wrists)
# COCO keypoints: 0=nose, 1-2=eyes, 3-4=ears, 5-6=shoulders, 7-8=elbows, 9-10=wrists, ...
val_evaluator = [
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        kpt_indexes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Upper body only
    )
]
```

### Example 3: Evaluate multiple subsets with separate metrics
```python
val_evaluator = [
    # All keypoints
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        prefix='all'
    ),
    # Only upper body
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        kpt_indexes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        prefix='upper_body'
    ),
    # Only lower body
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        kpt_indexes=[11, 12, 13, 14, 15, 16],
        prefix='lower_body'
    ),
]
```

### Example 4: Common COCO subsets

#### Limbs
```python
# COCO limbs only (excluding head)
dict(
    type='AtrafPCKAccuracy',
    thr=0.2,
    norm_item='bbox',
    kpt_indexes=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    prefix='limbs'
)
```

#### Upper Body
```python
# Head and upper limbs
dict(
    type='AtrafPCKAccuracy',
    thr=0.2,
    norm_item='bbox',
    kpt_indexes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    prefix='upper'
)
```

#### Lower Body
```python
# Lower limbs only
dict(
    type='AtrafPCKAccuracy',
    thr=0.2,
    norm_item='bbox',
    kpt_indexes=[11, 12, 13, 14, 15, 16],
    prefix='lower'
)
```

## Important Notes

1. **Keypoint Indices**: Use zero-based indexing (0 to num_keypoints-1)

2. **Torso Normalization**: When using `norm_item='torso'` with filtered keypoints:
   - The metric automatically checks if the required torso keypoints (indices 4 and 5) are in the filtered set
   - If they're not present, torso normalization is skipped for that sample
   - This prevents errors when evaluating subset of keypoints

3. **Prefix Usage**: Use the `prefix` parameter to distinguish multiple evaluators:
   ```python
   # Creates metrics like 'upper_PCK', 'lower_PCK', 'all_PCK'
   dict(type='AtrafPCKAccuracy', prefix='upper', kpt_indexes=[...]),
   dict(type='AtrafPCKAccuracy', prefix='lower', kpt_indexes=[...]),
   dict(type='AtrafPCKAccuracy', prefix='all'),
   ```

4. **Performance**: Filtering fewer keypoints slightly improves evaluation speed

## Log Output

When using keypoint filtering, the log output will include:
```
Evaluating AtrafPCKAccuracy (normalized by ``"bbox_size"``)(filtered by kpt_indexes)...
```

## Example Config Integration

Here's how to update your config:

```python
val_evaluator = [
    dict(type='PCKAccuracy', thr=0.2),
    dict(type='AUC'),
    dict(type='EPE'),
    dict(type='CocoMetric'),
    # Original - all keypoints
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        prefix='all'
    ),
    # Upper body only
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        kpt_indexes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        prefix='upper'
    ),
    # Classification metrics
    dict(
        type='ClassificationMetric',
        classifiers=[
            dict(field_name='gender', num_classes=3),
            dict(field_name='shape', num_classes=2),
        ]
    )
]
```

This would produce metrics:
- `all_PCK` - PCK for all 17 keypoints
- `upper_PCK` - PCK for first 11 keypoints (upper body)

