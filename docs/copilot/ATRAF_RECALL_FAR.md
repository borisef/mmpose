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

- `Recall_Atraf`: Recall = (Correct & High-score) / Total Correct
  Portion of correct keypoints (distance < thr) with predicted keypoint score > score_threshold.
  
- `FAR_atraf` (False Alarm Rate): FAR = (Incorrect & High-score) / Total High-score
  Portion of high-score detections that are incorrect (distance >= thr).

Both metrics accept the same parameters as `AtrafPCKAccuracy` plus a
`score_threshold` float (default 0.5). They support `norm_item` modes
(`'bbox'`, `'head'`, `'torso'`) and optional `kpt_indexes` filtering.

## Mathematical Formulation

For a set of detected keypoints, they are partitioned into 4 groups:

| Group | Correctness | Score | Count |
|-------|-------------|-------|-------|
| A | Correct (dist < thr) | High (> threshold) | n_ch |
| B | Incorrect (dist >= thr) | High (> threshold) | n_ih |
| C | Correct (dist < thr) | Low (<= threshold) | n_cl |
| D | Incorrect (dist >= thr) | Low (<= threshold) | n_il |

**Recall** = n_ch / (n_ch + n_cl) = Correct ∩ High-score / Total Correct keypoints

**FAR** = n_ih / (n_ch + n_ih) = Incorrect ∩ High-score / Total High-score detections

## Example with 110 Keypoints

Given:
- 50: Correct & High-score
- 30: Incorrect & High-score
- 20: Correct & Low-score
- 10: Incorrect & Low-score

**Calculations:**
- Recall = 50 / (50 + 20) = 50/70 ≈ 0.714286 (71.4% of correct predictions are high-confidence)
- FAR = 30 / (50 + 30) = 30/80 = 0.375 (37.5% of high-score detections are false alarms)

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

