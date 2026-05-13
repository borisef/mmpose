# 🚀 Quick Start: AtrafPCKAccuracy with kpt_indexes

## Installation

The feature is ready to use! Just update your config file.

## Quick Example

Add to your config's `val_evaluator`:

```python
# Evaluate only keypoints 0, 1, 2, 3
dict(
    type='AtrafPCKAccuracy',
    thr=0.2,
    norm_item='bbox',
    kpt_indexes=[0, 1, 2, 3]  # <-- Only these keypoints
)
```

## For COCO Dataset

```python
val_evaluator = [
    # Original (all keypoints)
    dict(type='PCKAccuracy', thr=0.2),
    
    # NEW: Body part specific evaluations
    dict(type='AtrafPCKAccuracy', thr=0.2, norm_item='bbox', 
         kpt_indexes=[0,1,2,3,4], prefix='head'),
    dict(type='AtrafPCKAccuracy', thr=0.2, norm_item='bbox', 
         kpt_indexes=[5,6,7,8,9,10], prefix='upper'),
    dict(type='AtrafPCKAccuracy', thr=0.2, norm_item='bbox', 
         kpt_indexes=[11,12,13,14,15,16], prefix='lower'),
]
```

## Output

```
PCK: 0.85
head_PCK: 0.92
upper_PCK: 0.88
lower_PCK: 0.80
```

## COCO Keypoint Indices

```
Head:        [0, 1, 2, 3, 4]           (nose, eyes, ears)
Upper:       [5, 6, 7, 8, 9, 10]       (shoulders, elbows, wrists)
Lower:       [11, 12, 13, 14, 15, 16]  (hips, knees, ankles)
Left side:   [1, 3, 5, 7, 9, 11, 13, 15]
Right side:  [2, 4, 6, 8, 10, 12, 14, 16]
```

## Done! 

That's it. The `kpt_indexes` parameter filters which keypoints are included in the PCK calculation.

