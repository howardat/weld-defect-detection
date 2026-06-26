import numpy as np

from weld_pipeline.porosity_pipeline import PoreParams
from experiments.loocv import loocv_indices, loocv_predict


def test_loocv_indices_hold_out_each_once():
    folds = loocv_indices(4)
    assert len(folds) == 4
    held = sorted(test for _, test in folds)
    assert held == [0, 1, 2, 3]
    for train, test in folds:
        assert test not in train
        assert len(train) == 3


def test_loocv_predict_returns_one_param_per_row():
    rng = np.random.default_rng(0)
    X = rng.random((6, 10)).astype(np.float32)
    Y = np.column_stack([
        rng.integers(11, 201, 6), rng.integers(1, 350, 6),
        rng.integers(1, 21, 6), rng.integers(0, 30, 6),
        rng.uniform(0.05, 0.95, 6), rng.integers(0, 60, 6),
    ]).astype(np.float32)
    preds = loocv_predict(X, Y, seed=0)
    assert len(preds) == 6
    assert all(isinstance(p, PoreParams) for p in preds)
