import numpy as np

from weld_pipeline.porosity_pipeline import PoreParams
from weld_pipeline.porosity_predictor import (
    PARAM_ORDER, params_to_vector, vector_to_params,
    train_predictor, predict_params,
)


def test_param_vector_roundtrip():
    p = PoreParams(51, 10, 3, 5, 0.3, 20)
    v = params_to_vector(p)
    assert v.shape == (6,)
    back = vector_to_params(v)
    assert back.block_size == 51
    assert back.darkness_thresh == 20


def test_param_order_is_six():
    assert len(PARAM_ORDER) == 6


def test_train_and_predict_returns_pore_params():
    rng = np.random.default_rng(0)
    X = rng.random((12, 10)).astype(np.float32)
    Y = np.column_stack([
        rng.integers(11, 201, 12),   # block_size
        rng.integers(1, 350, 12),    # adapt_c
        rng.integers(1, 21, 12),     # open_ksize
        rng.integers(0, 30, 12),     # erode_iters
        rng.uniform(0.05, 0.95, 12), # min_circularity
        rng.integers(0, 120, 12),    # darkness_thresh
    ]).astype(np.float32)
    models = train_predictor(X, Y, seed=0)
    assert len(models) == 6
    pred = predict_params(models, X[0])
    assert isinstance(pred, PoreParams)
    assert pred.block_size % 2 == 1
    assert 11 <= pred.block_size <= 201
