import numpy as np

from weld_pipeline.porosity_features import FEATURE_NAMES, extract_features


def test_feature_vector_length_matches_names():
    gray = np.full((50, 50), 128, np.uint8)
    weld = np.full((50, 50), 255, np.uint8)
    feats = extract_features(gray, weld)
    assert feats.shape == (len(FEATURE_NAMES),)
    assert len(FEATURE_NAMES) == 10


def test_constant_image_has_zero_std_and_contrast():
    gray = np.full((50, 50), 100, np.uint8)
    weld = np.full((50, 50), 255, np.uint8)
    feats = dict(zip(FEATURE_NAMES, extract_features(gray, weld)))
    assert feats["std"] == 0.0
    assert feats["contrast"] == 0.0
    assert feats["mean"] == 100.0


def test_empty_weld_mask_falls_back_to_whole_image():
    gray = np.full((40, 40), 80, np.uint8)
    empty = np.zeros((40, 40), np.uint8)
    feats = dict(zip(FEATURE_NAMES, extract_features(gray, empty)))
    assert feats["mean"] == 80.0


def test_weld_area_frac_is_between_zero_and_one():
    gray = np.full((100, 100), 128, np.uint8)
    weld = np.zeros((100, 100), np.uint8)
    weld[0:50, :] = 255   # half the image
    feats = dict(zip(FEATURE_NAMES, extract_features(gray, weld)))
    assert 0.4 < feats["weld_area_frac"] < 0.6
