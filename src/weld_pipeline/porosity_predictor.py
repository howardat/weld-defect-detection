"""Per-parameter Random Forest regressor: image features -> 6 OpenCV params."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from weld_pipeline.porosity_pipeline import PoreParams, sanitize_params

PARAM_ORDER: list[str] = [
    "block_size", "adapt_c", "open_ksize",
    "erode_iters", "min_circularity", "darkness_thresh",
]


def params_to_vector(p: PoreParams) -> np.ndarray:
    return np.array([getattr(p, name) for name in PARAM_ORDER], dtype=np.float32)


def vector_to_params(v: np.ndarray) -> PoreParams:
    kwargs = {name: float(v[i]) for i, name in enumerate(PARAM_ORDER)}
    return sanitize_params(PoreParams(**kwargs))


def train_predictor(X: np.ndarray, Y: np.ndarray, seed: int = 0) -> list:
    models = []
    for col in range(Y.shape[1]):
        rf = RandomForestRegressor(n_estimators=200, max_depth=5,
                                   random_state=seed, n_jobs=-1)
        rf.fit(X, Y[:, col])
        models.append(rf)
    return models


def predict_params(models: list, x: np.ndarray) -> PoreParams:
    x = np.asarray(x, dtype=np.float32).reshape(1, -1)
    preds = np.array([m.predict(x)[0] for m in models], dtype=np.float32)
    return vector_to_params(preds)
