## Twin Keypoints Feature

All ATRAF metrics now support a `twin_keypoints` parameter that allows symmetric keypoint pairs to be matched during evaluation. For example, in COCO format, left and right shoulders are often at indices 5 and 6.

### Use Case
When a predicted keypoint is close to the twin GT location (e.g., predicted shoulder at left_shoulder GT vs right_shoulder GT), it can count as correct. This is useful for evaluating:
- Symmetric body parts (left/right shoulders, elbows, hips, ankles)
- Cases where left/right ambiguity is acceptable

### Syntax

```python
twin_keypoints=[[i, j], [k, l], ...]
```

Where each pair `[i, j]` means: keypoint i can match either gt[i] or gt[j], and keypoint j can match either gt[j] or gt[i].

### Config Example

From `configs/atraf/borisef/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288_try1.py`:

```python
val_evaluator = [
    # Examples of twin_keypoints feature
    dict(type='AtrafPCKAccuracy', kpt_indexes=[0,1,2,3], prefix='atraf_pck_twin', thr=0.2,
         twin_keypoints=[[0, 1], [2, 3]]),
    dict(type='AtrafAUC', kpt_indexes=[0,1,2,3], prefix='atraf_auc_twin',
         twin_keypoints=[[0, 1], [2, 3]]),
    dict(type='AtrafEPE', kpt_indexes=[0,1,2,3], prefix='atraf_epe_twin',
         twin_keypoints=[[0, 1], [2, 3]]),
    dict(type='Recall_Atraf', kpt_indexes=[0,1,2,3], prefix='atraf_recall_twin', thr=0.2, score_threshold=0.5,
         twin_keypoints=[[0, 1], [2, 3]]),
]
```

### Behavior
- For each pair (i, j), the metric computes distances to both original and twin GT keypoints
- Uses the minimum valid distance (best match) to determine correctness
- If original and twin GT are both invisible, the pair is treated as invalid
- Backward compatible: if `twin_keypoints=None` (default), behavior is unchanged

# ATRAF Recall and FAR metrics

This document describes two new ATRAF metrics added to mmpose:

- `Recall_Atraf`: fraction of correct keypoints (distance < thr) whose
  predicted keypoint score > score_threshold.
- `FAR_atraf`: fraction of incorrect keypoints (distance >= thr) whose
  predicted keypoint score > score_threshold.

Both metrics accept the same parameters as `AtrafPCKAccuracy` plus a
`score_threshold` float (default 0.5). They support `norm_item` modes
(`'bbox'`, `'head'`, `'torso'`) and optional `kpt_indexes` filtering.

Usage example (config):

```
dict(type='Recall_Atraf', kpt_indexes=[0,1,2,3], thr=0.2, score_threshold=0.5, prefix='atraf_recall')
dict(type='FAR_atraf', kpt_indexes=[0,1,2,3], thr=0.2, score_threshold=0.5, prefix='atraf_far')
```

Notes:
- If `pred_instances` lacks `keypoint_scores`, scores default to 1.0.
- `Recall_Atraf` returns both Recall and FAR entries (e.g. `Recall`, `FAR`,
  `Recallh`, `FARh`, `Recallt`, `FARt` depending on `norm_item`).
- `FAR_atraf` is a thin wrapper that returns only FAR-related keys and can
  be used when only FAR logging is desired.

