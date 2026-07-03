"""Toy-example validation for the restructured ATRAF metrics.

Builds a synthetic 20-keypoint sample encoding the reference contingency table
    A = 10 (correct, high score)
    B =  5 (wrong,   high score)
    C =  3 (correct, low score)
    D =  2 (wrong,   low score)
and checks that ``Recall_Atraf`` and ``Success_Rate`` produce the expected
values (see the plan / toy example).

Run:  python tools/atraf/borisef/toy_metrics_check.py
"""
import numpy as np

from mmpose.evaluation.metrics.atraf.recall_far import Recall_Atraf
from mmpose.evaluation.metrics.atraf.success_rate import Success_Rate

THR = 0.05          # PCK threshold (normalized distance)
SCORE_THR = 0.5     # high-score threshold

# Distances (normalized): correct -> below THR, wrong -> above THR.
CORRECT_D = 0.01
WRONG_D = 0.5
# Scores: high -> above SCORE_THR, low -> below.
HIGH_S = 0.9
LOW_S = 0.2

# 20 keypoints: [A=10 correct/high, B=5 wrong/high, C=3 correct/low, D=2 wrong/low]
specs = (
    [(CORRECT_D, HIGH_S)] * 10 +
    [(WRONG_D, HIGH_S)] * 5 +
    [(CORRECT_D, LOW_S)] * 3 +
    [(WRONG_D, LOW_S)] * 2
)
K = len(specs)

# Place GT on a line and offset the prediction by the desired distance. With a
# bbox_size norm factor of 1.0 the offset equals the normalized distance.
gt = np.zeros((1, K, 2), dtype=np.float32)
pred = np.zeros((1, K, 2), dtype=np.float32)
scores = np.zeros((1, K), dtype=np.float32)
for i, (dist, score) in enumerate(specs):
    gt[0, i] = [float(i) * 10.0, 0.0]
    pred[0, i] = [float(i) * 10.0 + dist, 0.0]
    scores[0, i] = score

visible = np.ones((1, K), dtype=np.float32)
# bbox with max side == 1.0 so norm factor == 1.0 -> distance == raw offset
bbox = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)

data_sample = {
    'pred_instances': {'keypoints': pred, 'keypoint_scores': scores},
    'gt_instances': {
        'keypoints': gt,
        'keypoints_visible': visible,
        'bboxes': bbox,
    },
}


def run(metric):
    metric.process({}, [dict(data_sample)])
    out = metric.compute_metrics(metric.results)
    metric.results.clear()
    return out


def approx(a, b, tol=1e-3):
    return abs(a - b) <= tol


recall = run(Recall_Atraf(thr=THR, score_threshold=SCORE_THR))
success = run(Success_Rate(thr=THR, score_threshold=SCORE_THR))

print('Recall_Atraf:', {k: round(v, 4) for k, v in recall.items()})
print('Success_Rate:', {k: round(v, 4) for k, v in success.items()})

expected_recall = {
    'Recall': 10 / 13,       # 0.769
    'Precision': 10 / 15,    # 0.667
    'F1': 0.7143,
    'F2': (0.7143 + 0.5) / 2,  # 0.6071
    'SmartF1': 0.7879,       # best at t->0 : P=13/20, R=1
    'SmartThreshold': 0.0,
}
expected_success = {
    'PD_Success_Rate': 10 / 20,   # 0.5
    'FAR_Error_Rate': 5 / 20,     # 0.25
    'Skip_Rate': 5 / 20,          # 0.25
    'Combined_Success_Rate': 2.0 / 3.0,           # 0.6667
    'Best_Combined_Success_Rate': 2.3 / 3.0,      # 0.7667 at t->0
    'BestThreshold': 0.0,
}

ok = True
for k, v in expected_recall.items():
    got = recall.get(k)
    good = got is not None and approx(got, v)
    ok = ok and good
    print(f'  [{"OK" if good else "FAIL"}] Recall_Atraf.{k}: got {got}, want {round(v,4)}')
for k, v in expected_success.items():
    got = success.get(k)
    good = got is not None and approx(got, v)
    ok = ok and good
    print(f'  [{"OK" if good else "FAIL"}] Success_Rate.{k}: got {got}, want {round(v,4)}')

print('\nALL PASS' if ok else '\nSOME FAILED')
raise SystemExit(0 if ok else 1)
