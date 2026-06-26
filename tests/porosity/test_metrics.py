import numpy as np

from weld_pipeline.porosity_pipeline import pixel_f1


def test_perfect_overlap_is_one():
    m = np.zeros((10, 10), np.uint8)
    m[2:6, 2:6] = 255
    assert pixel_f1(m, m) == 1.0


def test_no_overlap_is_zero():
    gt = np.zeros((10, 10), np.uint8); gt[0:3, 0:3] = 255
    det = np.zeros((10, 10), np.uint8); det[7:10, 7:10] = 255
    assert pixel_f1(gt, det) == 0.0


def test_both_empty_is_one():
    z = np.zeros((10, 10), np.uint8)
    assert pixel_f1(z, z) == 1.0


def test_det_empty_gt_nonempty_is_zero():
    gt = np.zeros((10, 10), np.uint8); gt[0:3, 0:3] = 255
    det = np.zeros((10, 10), np.uint8)
    assert pixel_f1(gt, det) == 0.0


def test_partial_overlap_between_zero_and_one():
    gt = np.zeros((10, 10), np.uint8); gt[0:4, 0:4] = 255   # 16 px
    det = np.zeros((10, 10), np.uint8); det[0:4, 0:2] = 255  # 8 px, all TP
    f1 = pixel_f1(gt, det)
    assert 0.0 < f1 < 1.0
