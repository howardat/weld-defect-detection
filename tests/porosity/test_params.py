from weld_pipeline.porosity_pipeline import PoreParams, sanitize_params, PARAM_BOUNDS


def test_param_bounds_has_six_keys():
    assert set(PARAM_BOUNDS) == {
        "block_size", "adapt_c", "open_ksize",
        "erode_iters", "min_circularity", "darkness_thresh",
    }


def test_sanitize_forces_block_size_odd_and_in_range():
    p = sanitize_params(PoreParams(50, 10, 4, 5, 0.3, 12))
    assert p.block_size % 2 == 1
    assert 11 <= p.block_size <= 201


def test_sanitize_forces_open_ksize_odd():
    p = sanitize_params(PoreParams(51, 10, 8, 5, 0.3, 12))
    assert p.open_ksize % 2 == 1


def test_sanitize_clips_out_of_range():
    p = sanitize_params(PoreParams(9, 999, 99, 99, 5.0, 999))
    assert p.block_size == 11
    assert p.adapt_c == 350
    assert p.open_ksize == 21
    assert p.erode_iters == 30
    assert p.min_circularity == 0.95
    assert p.darkness_thresh == 60


def test_sanitize_coerces_integer_fields():
    p = sanitize_params(PoreParams(51.0, 10.9, 5.0, 5.4, 0.3, 12.7))
    assert isinstance(p.block_size, int)
    assert isinstance(p.adapt_c, int)
    assert isinstance(p.open_ksize, int)
    assert isinstance(p.erode_iters, int)
    assert isinstance(p.darkness_thresh, int)
