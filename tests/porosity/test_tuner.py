import cv2
import numpy as np

from weld_pipeline.porosity_pipeline import PoreParams, detect_pores, detection_mask, pixel_f1
from weld_pipeline.porosity_tuner import optimize_image, refine_params


def _scene(size=200):
    gray = np.full((size, size), 200, np.uint8)
    gt = np.zeros((size, size), np.uint8)
    for cx, cy, r in [(60, 60, 12), (140, 140, 14)]:
        cv2.circle(gray, (cx, cy), r, 25, -1)
        cv2.circle(gt, (cx, cy), r, 255, -1)
    weld = np.full((size, size), 255, np.uint8)
    return gray, weld, gt


def test_optimize_returns_valid_params_and_f1():
    gray, weld, gt = _scene()
    params, f1 = optimize_image(gray, weld, gt, n_trials=25, seed=0)
    assert isinstance(params, PoreParams)
    assert 0.0 <= f1 <= 1.0
    assert params.block_size % 2 == 1


def test_optimize_beats_a_deliberately_bad_param_set():
    gray, weld, gt = _scene()
    params, f1 = optimize_image(gray, weld, gt, n_trials=40, seed=0)
    bad = PoreParams(11, 1, 21, 30, 0.95, 60)
    bad_f1 = pixel_f1(gt, detection_mask(detect_pores(gray, weld, bad), gray.shape))
    assert f1 >= bad_f1


def test_refine_returns_valid_params():
    gray, weld, _ = _scene()
    warm = PoreParams(51, 10, 3, 0, 0.5, 20)
    refined = refine_params(gray, weld, warm, n_trials=10, seed=0)
    assert isinstance(refined, PoreParams)
    assert refined.block_size % 2 == 1
