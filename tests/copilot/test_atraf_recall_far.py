import numpy as np

from mmpose.evaluation.metrics.atraf.recall_far import Recall_Atraf, FAR_atraf


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


def test_recall_far_all_correct():
	pred = np.array([[[10.0, 10.0], [20.0, 20.0]]], dtype=np.float32)
	gt = pred.copy()
	scores = np.array([[0.9, 0.8]], dtype=np.float32)

	metric = Recall_Atraf(thr=0.05, norm_item='bbox', score_threshold=0.5)
	sample = make_sample(pred, gt, scores)
	metric.process([{}], [sample])
	res = metric.compute_metrics(metric.results)
	assert 'Recall' in res and 'FAR' in res
	assert pytest_float_equal(res['Recall'], 1.0)
	assert pytest_float_equal(res['FAR'], 0.0)


def test_recall_far_one_wrong_high_score():
	# kp0: correct, high-score (0.9 > 0.5)
	# kp1: incorrect (pred far from gt), high-score (0.8 > 0.5)
	# Total correct: 1, total high-score: 2
	# Recall = 1/1 = 1.0
	# FAR = (incorrect & high-score) / total high-score = 1/2 = 0.5
	pred = np.array([[[10.0, 10.0], [200.0, 200.0]]], dtype=np.float32)
	gt = np.array([[[10.0, 10.0], [20.0, 20.0]]], dtype=np.float32)
	scores = np.array([[0.9, 0.8]], dtype=np.float32)

	metric = Recall_Atraf(thr=0.05, norm_item='bbox', score_threshold=0.5)
	sample = make_sample(pred, gt, scores)
	metric.process([{}], [sample])
	res = metric.compute_metrics(metric.results)
	assert pytest_float_equal(res['Recall'], 1.0)
	assert pytest_float_equal(res['FAR'], 0.5)


def test_far_atraf_wrapper():
	pred = np.array([[[10.0, 10.0], [200.0, 200.0]]], dtype=np.float32)
	gt = np.array([[[10.0, 10.0], [20.0, 20.0]]], dtype=np.float32)
	scores = np.array([[0.1, 0.9]], dtype=np.float32)

	metric = FAR_atraf(thr=0.05, norm_item='bbox', score_threshold=0.5)
	sample = make_sample(pred, gt, scores)
	metric.process([{}], [sample])
	res = metric.compute_metrics(metric.results)
	assert any(k.startswith('FAR') for k in res.keys())


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

	# With twin pair [0,1], kp0 is considered correct and has high score -> Recall=0.5
	metric_twin = Recall_Atraf(thr=0.05, norm_item='bbox', score_threshold=0.5, twin_keypoints=[[0, 1]])
	metric_twin.process([{}], [sample])
	res_twin = metric_twin.compute_metrics(metric_twin.results)
	assert pytest_float_equal(res_twin['Recall'], 0.5)


def test_recall_far_110_keypoints_exact_scenario():
	"""
	Test CORRECTED FAR calculation with 110 keypoints distributed as:
	- 50: Correct & High-score
	- 30: Wrong & High-score
	- 20: Correct & Low-score
	- 10: Wrong & Low-score

	Expected:
	- Recall = 50 / (50+20) = 50/70 ≈ 0.714286
	- FAR (False Alarm Rate) = 30 / (30+50) = 30/80 = 0.375
	  (incorrect & high-score) / (total high-score detections)
	"""
	num_kpts = 110

	pred_coords = np.zeros((1, num_kpts, 2), dtype=np.float32)
	gt_coords = np.zeros((1, num_kpts, 2), dtype=np.float32)
	scores = np.zeros((1, num_kpts), dtype=np.float32)

	# Group 1: Correct & High-score (0-49)
	for i in range(50):
		pred_coords[0, i] = [10.0 + i * 0.1, 10.0 + i * 0.1]
		gt_coords[0, i] = [10.0 + i * 0.1, 10.0 + i * 0.1]
		scores[0, i] = 0.8

	# Group 2: Wrong & High-score (50-79)
	for i in range(50, 80):
		pred_coords[0, i] = [10.0 + (i-50) * 0.1, 10.0 + (i-50) * 0.1]
		gt_coords[0, i] = [50.0 + (i-50) * 0.1, 50.0 + (i-50) * 0.1]
		scores[0, i] = 0.8

	# Group 3: Correct & Low-score (80-99)
	for i in range(80, 100):
		pred_coords[0, i] = [20.0 + (i-80) * 0.1, 20.0 + (i-80) * 0.1]
		gt_coords[0, i] = [20.0 + (i-80) * 0.1, 20.0 + (i-80) * 0.1]
		scores[0, i] = 0.3

	# Group 4: Wrong & Low-score (100-109)
	for i in range(100, 110):
		pred_coords[0, i] = [20.0 + (i-100) * 0.1, 20.0 + (i-100) * 0.1]
		gt_coords[0, i] = [60.0 + (i-100) * 0.1, 60.0 + (i-100) * 0.1]
		scores[0, i] = 0.3

	metric = Recall_Atraf(thr=0.05, norm_item='bbox', score_threshold=0.5)
	sample = make_sample(pred_coords, gt_coords, scores)
	metric.process([{}], [sample])
	res = metric.compute_metrics(metric.results)

	expected_recall = 50.0 / 70.0  # 0.714286
	expected_far = 30.0 / 80.0      # 0.375

	assert pytest_float_equal(res['Recall'], expected_recall, eps=1e-5), \
		f"Expected Recall={expected_recall}, got {res['Recall']}"
	assert pytest_float_equal(res['FAR'], expected_far, eps=1e-5), \
		f"Expected FAR={expected_far}, got {res['FAR']}"
