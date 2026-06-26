import numpy as np

from weld_pipeline.porosity_pipeline import PoreDetection, proxy_score


def _det(circ, contrast):
    return PoreDetection(contour=np.zeros((1, 1, 2), np.int32),
                         circularity=circ, darkness_contrast=contrast)


def test_empty_is_zero():
    assert proxy_score([]) == 0.0


def test_more_circular_and_darker_scores_higher():
    weak = [_det(0.3, 20)]
    strong = [_det(0.9, 120)]
    assert proxy_score(strong) > proxy_score(weak)


def test_negative_contrast_clamped():
    assert proxy_score([_det(0.9, -50)]) == 0.0
