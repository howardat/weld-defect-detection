"""Leave-One-Out Cross-Validation utilities shared by experiments A and C."""
from __future__ import annotations

import numpy as np

from weld_pipeline.porosity_pipeline import PoreParams
from weld_pipeline.porosity_predictor import predict_params, train_predictor


def loocv_indices(n: int) -> list[tuple[list[int], int]]:
    return [([j for j in range(n) if j != i], i) for i in range(n)]


def loocv_predict(X: np.ndarray, Y: np.ndarray, seed: int = 0) -> list[PoreParams]:
    preds: list[PoreParams] = []
    for train_idx, test_idx in loocv_indices(len(X)):
        models = train_predictor(X[train_idx], Y[train_idx], seed=seed)
        preds.append(predict_params(models, X[test_idx]))
    return preds
