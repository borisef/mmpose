import numpy as np

from mmpose.evaluation.metrics.atraf.smart_f1 import Smart_F1


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


def test_smart_f1_basic_two_keypoints():
	# Two keypoints: kp0 correct score 0.6, kp1 wrong score 0.4
	# Best threshold around 0.4-0.6 -> only kp0 counted: precision=1, recall=1 -> F1=1
	pred = np.array([[[10.0, 10.0], [200.0, 200.0]]], dtype=np.float32)
	gt = np.array([[[10.0, 10.0], [20.0, 20.0]]], dtype=np.float32)
	scores = np.array([[0.6, 0.4]], dtype=np.float32)

	metric = Smart_F1(thr=0.05, norm_item='bbox', num_steps=101)
	sample = make_sample(pred, gt, scores)
	metric.process([{}], [sample])
	res = metric.compute_metrics(metric.results)

	assert 'SmartF1' in res and 'SmartThreshold' in res
	assert pytest_float_equal(res['SmartF1'], 1.0)
	# The best threshold should be in [0.4, 0.6]
	assert 0.4 <= res['SmartThreshold'] <= 0.6


