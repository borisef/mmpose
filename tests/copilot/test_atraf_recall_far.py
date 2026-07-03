import numpy as np

from mmpose.evaluation.metrics.atraf.recall_far import Recall_Atraf
from mmpose.evaluation.metrics.atraf.success_rate import Success_Rate


def make_sample(pred_kpts, gt_kpts, scores=None, bbox=None):
	if scores is None:
		scores = np.ones((1, pred_kpts.shape[1]), dtype=np.float32)
	if bbox is None:
		bbox = np.array([[0, 0, 100, 100]], dtype=np.float32)
	sample = {
		'pred_instances': {
			'keypoints': pred_kpts,
			'keypoint_scores': scores,
		},
		'gt_instances': {
			'keypoints': gt_kpts,
			'keypoints_visible': np.ones((1, pred_kpts.shape[1], 1), dtype=bool),
			'bboxes': bbox,
		}
	}
	return sample


def pytest_float_equal(a, b, eps=1e-6):
	return abs(a - b) <= eps


def test_recall_all_correct():
	pred = np.array([[[10.0, 10.0], [20.0, 20.0]]], dtype=np.float32)
	gt = pred.copy()
	scores = np.array([[0.9, 0.8]], dtype=np.float32)

	metric = Recall_Atraf(thr=0.05, norm_item='bbox', score_threshold=0.5)
	sample = make_sample(pred, gt, scores)
	metric.process([{}], [sample])
	res = metric.compute_metrics(metric.results)
	# FAR is removed; Recall_Atraf now emits F2/SmartF1/SmartThreshold.
	assert 'FAR' not in res
	assert 'Recall' in res and 'F2' in res and 'SmartF1' in res
	assert pytest_float_equal(res['Recall'], 1.0)
	assert pytest_float_equal(res['Precision'], 1.0)
	assert pytest_float_equal(res['F1'], 1.0)
	# F2 = (F1 + PD_Success_Rate)/2 = (1.0 + 1.0)/2 = 1.0 (all correct & high)
	assert pytest_float_equal(res['F2'], 1.0)


def test_recall_one_wrong_high_score():
	# kp0: correct, high-score (0.9 > 0.5)
	# kp1: incorrect (pred far from gt), high-score (0.8 > 0.5)
	pred = np.array([[[10.0, 10.0], [200.0, 200.0]]], dtype=np.float32)
	gt = np.array([[[10.0, 10.0], [20.0, 20.0]]], dtype=np.float32)
	scores = np.array([[0.9, 0.8]], dtype=np.float32)

	metric = Recall_Atraf(thr=0.05, norm_item='bbox', score_threshold=0.5)
	sample = make_sample(pred, gt, scores)
	metric.process([{}], [sample])
	res = metric.compute_metrics(metric.results)
	assert pytest_float_equal(res['Recall'], 1.0)
	# Precision = correct_high / total_high = 1/2 = 0.5
	assert pytest_float_equal(res['Precision'], 0.5)
	# F1 = 2 * P * R / (P + R) = 2 * 0.5 * 1 / 1.5 = 0.666666...
	f1 = 2.0 * 0.5 * 1.0 / (0.5 + 1.0)
	assert pytest_float_equal(res['F1'], f1)
	# PD_Success_Rate = correct_high / N = 1/2 = 0.5 -> F2 = (f1 + 0.5)/2
	assert pytest_float_equal(res['F2'], (f1 + 0.5) / 2.0)


def test_twin_keypoints_behavior():
	# pred for kp0 is at kp1's gt location. Without twin, kp0 incorrect.
	pred = np.array([[[20.0, 20.0], [20.0, 20.0]]], dtype=np.float32)
	gt = np.array([[[10.0, 10.0], [20.0, 20.0]]], dtype=np.float32)
	scores = np.array([[0.9, 0.1]], dtype=np.float32)

	# Without twin, only kp1 is correct but has low score -> Recall=0.0
	metric_no_twin = Recall_Atraf(thr=0.05, norm_item='bbox', score_threshold=0.5)
	sample = make_sample(pred, gt, scores)
	metric_no_twin.process([{}], [sample])
	res_no_twin = metric_no_twin.compute_metrics(metric_no_twin.results)
	assert pytest_float_equal(res_no_twin['Recall'], 0.0)
	assert pytest_float_equal(res_no_twin['Precision'], 0.0)
	assert pytest_float_equal(res_no_twin['F1'], 0.0)

	# With twin pair [0,1], kp0 is considered correct and has high score -> Recall=0.5
	metric_twin = Recall_Atraf(thr=0.05, norm_item='bbox', score_threshold=0.5, twin_keypoints=[[0, 1]])
	metric_twin.process([{}], [sample])
	res_twin = metric_twin.compute_metrics(metric_twin.results)
	assert pytest_float_equal(res_twin['Recall'], 0.5)
	assert pytest_float_equal(res_twin['Precision'], 1.0)
	# F1 = 2 * 1.0 * 0.5 / (1.5) = 0.666666...
	assert pytest_float_equal(res_twin['F1'], 2.0 * 1.0 * 0.5 / (1.0 + 0.5))


def _make_110_kpt_sample():
	"""110 keypoints: 50 correct/high, 30 wrong/high, 20 correct/low, 10 wrong/low."""
	num_kpts = 110
	pred_coords = np.zeros((1, num_kpts, 2), dtype=np.float32)
	gt_coords = np.zeros((1, num_kpts, 2), dtype=np.float32)
	scores = np.zeros((1, num_kpts), dtype=np.float32)

	for i in range(50):  # Correct & High-score
		pred_coords[0, i] = [10.0 + i * 0.1, 10.0 + i * 0.1]
		gt_coords[0, i] = [10.0 + i * 0.1, 10.0 + i * 0.1]
		scores[0, i] = 0.8
	for i in range(50, 80):  # Wrong & High-score
		pred_coords[0, i] = [10.0 + (i-50) * 0.1, 10.0 + (i-50) * 0.1]
		gt_coords[0, i] = [50.0 + (i-50) * 0.1, 50.0 + (i-50) * 0.1]
		scores[0, i] = 0.8
	for i in range(80, 100):  # Correct & Low-score
		pred_coords[0, i] = [20.0 + (i-80) * 0.1, 20.0 + (i-80) * 0.1]
		gt_coords[0, i] = [20.0 + (i-80) * 0.1, 20.0 + (i-80) * 0.1]
		scores[0, i] = 0.3
	for i in range(100, 110):  # Wrong & Low-score
		pred_coords[0, i] = [20.0 + (i-100) * 0.1, 20.0 + (i-100) * 0.1]
		gt_coords[0, i] = [60.0 + (i-100) * 0.1, 60.0 + (i-100) * 0.1]
		scores[0, i] = 0.3
	return make_sample(pred_coords, gt_coords, scores)


def test_recall_110_keypoints_exact_scenario():
	"""A=50, B=30, C=20, D=10 -> Recall=50/70, Precision=50/80, F1=2/3."""
	metric = Recall_Atraf(thr=0.05, norm_item='bbox', score_threshold=0.5)
	metric.process([{}], [_make_110_kpt_sample()])
	res = metric.compute_metrics(metric.results)

	expected_recall = 50.0 / 70.0
	expected_precision = 50.0 / 80.0
	expected_f1 = 2.0 * expected_precision * expected_recall / (expected_precision + expected_recall)

	assert 'FAR' not in res
	assert pytest_float_equal(res['Recall'], expected_recall, eps=1e-5)
	assert pytest_float_equal(res['Precision'], expected_precision, eps=1e-5)
	assert pytest_float_equal(res['F1'], expected_f1, eps=1e-5)


def test_success_rate_110_keypoints_exact_scenario():
	"""A=50, B=30, C=20, D=10, N=110 with equal weights.

	PD=50/110, FAR_Error=30/110, Skip=30/110,
	Combined = (1/3)[(1-30/110)+(1-30/110)+50/110].
	"""
	metric = Success_Rate(thr=0.05, norm_item='bbox', score_threshold=0.5)
	metric.process([{}], [_make_110_kpt_sample()])
	res = metric.compute_metrics(metric.results)

	pd = 50.0 / 110.0
	far_err = 30.0 / 110.0
	skip = 30.0 / 110.0
	combined = ((1.0 - far_err) + (1.0 - skip) + pd) / 3.0

	assert pytest_float_equal(res['PD_Success_Rate'], pd, eps=1e-5)
	assert pytest_float_equal(res['FAR_Error_Rate'], far_err, eps=1e-5)
	assert pytest_float_equal(res['Skip_Rate'], skip, eps=1e-5)
	assert pytest_float_equal(res['Combined_Success_Rate'], combined, eps=1e-5)
	assert 'Best_Combined_Success_Rate' in res and 'BestThreshold' in res
	# Best over thresholds is at least as good as the fixed-threshold value.
	assert res['Best_Combined_Success_Rate'] >= res['Combined_Success_Rate'] - 1e-9
