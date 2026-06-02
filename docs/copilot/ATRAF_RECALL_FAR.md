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

