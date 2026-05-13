# AtrafPCKAccuracy - Keypoint Filtering Feature

## ✅ Implementation Complete

### What Was Added

A new `AtrafPCKAccuracy` metric that extends the base `PCKAccuracy` with keypoint filtering support.

**Location:** `mmpose/evaluation/metrics/atraf/pck_accuracy.py`

---

## 🎯 Feature Overview

The `AtrafPCKAccuracy` metric allows you to evaluate PCK accuracy on only specific keypoints, enabling analysis of different body parts separately.

### Key Parameters

- **kpt_indexes** (Sequence[int], optional): List of keypoint indices to include
  - Example: `[0, 1, 2, 3]` - include only first 4 keypoints
  - Default: `None` (all keypoints)

---

## 📝 Usage Examples

### Example 1: First 4 keypoints only
```python
val_evaluator = [
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        kpt_indexes=[0, 1, 2, 3]
    )
]
```

### Example 2: COCO dataset - separate body parts

For COCO (17 keypoints):
```python
val_evaluator = [
    # Head: nose, eyes, ears
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        kpt_indexes=[0, 1, 2, 3, 4],
        prefix='head'
    ),
    # Upper body: shoulders, elbows, wrists
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        kpt_indexes=[5, 6, 7, 8, 9, 10],
        prefix='upper'
    ),
    # Lower body: hips, knees, ankles
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        kpt_indexes=[11, 12, 13, 14, 15, 16],
        prefix='lower'
    ),
]
```

### Example 3: Update your existing config

In `configs/atraf/borisef/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288_try1.py`:

Change from:
```python
val_evaluator = [
    dict(type='PCKAccuracy', thr=0.2),
    dict(type='AUC'),
    dict(type='EPE'),
    dict(type='CocoMetric'),
    dict(
        type='ClassificationMetric',
        classifiers=[
            dict(field_name='gender', num_classes=3),
            dict(field_name='shape', num_classes=2),
        ]
    )
]
```

To:
```python
val_evaluator = [
    dict(type='PCKAccuracy', thr=0.2),
    dict(type='AUC'),
    dict(type='EPE'),
    dict(type='CocoMetric'),
    
    # NEW: Evaluate specific keypoint groups
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        kpt_indexes=[0, 1, 2, 3, 4],
        prefix='head'
    ),
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        kpt_indexes=[5, 6, 7, 8, 9, 10],
        prefix='upper'
    ),
    dict(
        type='AtrafPCKAccuracy',
        thr=0.2,
        norm_item='bbox',
        kpt_indexes=[11, 12, 13, 14, 15, 16],
        prefix='lower'
    ),
    
    dict(
        type='ClassificationMetric',
        classifiers=[
            dict(field_name='gender', num_classes=3),
            dict(field_name='shape', num_classes=2),
        ]
    )
]
```

---

## 📊 Expected Output

With the above config, you'll get metrics like:

```
PCK: 0.85 (all keypoints via base PCKAccuracy)
head_PCK: 0.92 (head keypoints only)
upper_PCK: 0.88 (upper body only)
lower_PCK: 0.80 (lower body only)
AUC: 0.78
EPE: 2.34
coco/AP: 0.72
gender_accuracy: 0.95
shape_accuracy: 0.92
```

---

## 🔍 COCO Keypoint Indices Reference

For COCO dataset (17 keypoints):

```
0: nose
1: left_eye        2: right_eye
3: left_ear        4: right_ear
5: left_shoulder   6: right_shoulder
7: left_elbow      8: right_elbow
9: left_wrist      10: right_wrist
11: left_hip       12: right_hip
13: left_knee      14: right_knee
15: left_ankle     16: right_ankle
```

Common groupings:
- **Head**: [0, 1, 2, 3, 4]
- **Upper body**: [5, 6, 7, 8, 9, 10]
- **Lower body**: [11, 12, 13, 14, 15, 16]
- **Left side**: [1, 3, 5, 7, 9, 11, 13, 15]
- **Right side**: [2, 4, 6, 8, 10, 12, 14, 16]

---

## ⚙️ Implementation Details

### How it Works

1. **process()** method: Filters keypoints based on `kpt_indexes` before storing results
2. **compute_metrics()** method: Calculates metrics only on filtered keypoints
3. **Torso handling**: When using `norm_item='torso'`, automatically skips samples where torso keypoints aren't in the filtered set

### Backward Compatibility

- Base `PCKAccuracy` unchanged (no legacy code modified)
- `AtrafPCKAccuracy` can be used alongside regular `PCKAccuracy`
- Works with all normalization methods: 'bbox', 'head', 'torso'

---

## 🚀 Advanced Usage

### Multiple thresholds with different keypoint subsets

```python
val_evaluator = [
    # Strict eval (high threshold) on important keypoints
    dict(
        type='AtrafPCKAccuracy',
        thr=0.1,
        norm_item='bbox',
        kpt_indexes=[0, 5, 6, 11, 12],  # nose, shoulders, hips
        prefix='strict_important'
    ),
    # Relaxed eval (low threshold) on all keypoints
    dict(
        type='AtrafPCKAccuracy',
        thr=0.3,
        norm_item='bbox',
        prefix='relaxed_all'
    ),
]
```

### Per-limb evaluation

```python
val_evaluator = [
    dict(type='AtrafPCKAccuracy', thr=0.2, kpt_indexes=[5, 7, 9], prefix='left_arm'),
    dict(type='AtrafPCKAccuracy', thr=0.2, kpt_indexes=[6, 8, 10], prefix='right_arm'),
    dict(type='AtrafPCKAccuracy', thr=0.2, kpt_indexes=[11, 13, 15], prefix='left_leg'),
    dict(type='AtrafPCKAccuracy', thr=0.2, kpt_indexes=[12, 14, 16], prefix='right_leg'),
]
```

---

## 📋 Files Modified/Created

### Created:
- ✅ `mmpose/evaluation/metrics/atraf/pck_accuracy.py` - AtrafPCKAccuracy class
- ✅ `mmpose/evaluation/metrics/atraf/__init__.py` - Updated to export AtrafPCKAccuracy

### Unchanged (Legacy):
- ✓ `mmpose/evaluation/metrics/keypoint_2d_metrics.py` - PCKAccuracy (no changes)

---

## 💡 Tips

1. Use `prefix` parameter to distinguish metrics in output
2. Start with full keypoint set, then add filtered versions for specific analysis
3. Keypoint indices are zero-based (0 to num_keypoints-1)
4. All normalization methods ('bbox', 'head', 'torso') are supported
5. Combine with `ClassificationMetric` for joint evaluation of pose and attributes

---

## ✨ Summary

You now have a flexible way to evaluate pose estimation on specific keypoint subsets!
Just add the keypoint indices to filter which keypoints to include in the analysis.

