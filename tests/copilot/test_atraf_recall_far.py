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
	pred = np.array([[[10.0, 10.0], [200.0, 200.0]]], dtype=np.float32)
	gt = np.array([[[10.0, 10.0], [20.0, 20.0]]], dtype=np.float32)
	scores = np.array([[0.9, 0.8]], dtype=np.float32)

	metric = Recall_Atraf(thr=0.05, norm_item='bbox', score_threshold=0.5)
	sample = make_sample(pred, gt, scores)
	metric.process([{}], [sample])
	res = metric.compute_metrics(metric.results)
	assert pytest_float_equal(res['Recall'], 1.0)
	assert pytest_float_equal(res['FAR'], 1.0)


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


