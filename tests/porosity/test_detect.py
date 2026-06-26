import cv2
import numpy as np

from weld_pipeline.porosity_pipeline import (
    PoreParams, detect_pores, detection_mask, erode_weld_mask,
)


def _synthetic_pore_image(size=200):
    """Light-gray background with two dark circular 'pores'."""
    gray = np.full((size, size), 200, np.uint8)
    cv2.circle(gray, (60, 60), 12, 30, -1)
    cv2.circle(gray, (140, 140), 14, 25, -1)
    return gray


def _full_weld_mask(size=200):
    return np.full((size, size), 255, np.uint8)


PARAMS = PoreParams(
    block_size=51, adapt_c=10, open_ksize=3,
    erode_iters=0, min_circularity=0.5, darkness_thresh=20,
)


def test_erode_zero_iters_is_identity():
    m = _full_weld_mask()
    assert np.array_equal(erode_weld_mask(m, 0), m)


def test_erode_shrinks_mask():
    m = np.zeros((100, 100), np.uint8)
    m[20:80, 20:80] = 255
    eroded = erode_weld_mask(m, 5)
    assert eroded.sum() < m.sum()


def test_detect_finds_two_pores():
    gray = _synthetic_pore_image()
    dets = detect_pores(gray, _full_weld_mask(), PARAMS)
    assert len(dets) == 2


def test_detection_mask_marks_pixels():
    gray = _synthetic_pore_image()
    dets = detect_pores(gray, _full_weld_mask(), PARAMS)
    mask = detection_mask(dets, gray.shape)
    assert mask.shape == gray.shape
    assert mask.max() == 255
    assert mask[60, 60] == 255  # center of first pore


def test_high_circularity_threshold_keeps_round_pores():
    gray = _synthetic_pore_image()
    strict = PoreParams(51, 10, 3, 0, 0.85, 20)
    dets = detect_pores(gray, _full_weld_mask(), strict)
    assert len(dets) >= 1


def test_darkness_threshold_rejects_faint_pore():
    # Faint pore: interior 150 on a 200 background -> contrast gap ~50.
    size = 200
    gray = np.full((size, size), 200, np.uint8)
    cv2.circle(gray, (100, 100), 14, 150, -1)
    weld = np.full((size, size), 255, np.uint8)
    lenient = PoreParams(51, 10, 3, 0, 0.5, 20)    # gap>=20 keeps it
    strict = PoreParams(51, 10, 3, 0, 0.5, 120)    # gap>=120 (in-bounds) rejects it
    assert len(detect_pores(gray, weld, lenient)) >= 1
    assert len(detect_pores(gray, weld, strict)) == 0
