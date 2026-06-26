"""Per-image features describing the weld region, used to predict OpenCV params."""
from __future__ import annotations

import cv2
import numpy as np

FEATURE_NAMES: list[str] = [
    "mean", "std", "p10", "p50", "p90",
    "contrast", "lap_var", "weld_area_frac", "weld_aspect", "edge_density",
]


def extract_features(gray: np.ndarray, weld_mask: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    has_weld = weld_mask is not None and weld_mask.any()
    region = gray[weld_mask > 0] if has_weld else gray.ravel()

    mean = float(np.mean(region))
    std = float(np.std(region))
    p10, p50, p90 = (float(np.percentile(region, q)) for q in (10, 50, 90))
    contrast = p90 - p10
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if has_weld:
        weld_area_frac = float((weld_mask > 0).sum()) / (h * w)
        xs, ys = np.where(weld_mask > 0)[1], np.where(weld_mask > 0)[0]
        bw = xs.max() - xs.min() + 1
        bh = ys.max() - ys.min() + 1
        weld_aspect = float(min(bw, bh) / max(bw, bh))
    else:
        weld_area_frac = 1.0
        weld_aspect = float(min(h, w) / max(h, w))

    edges = cv2.Canny(gray, 30, 100)
    edge_region = edges[weld_mask > 0] if has_weld else edges.ravel()
    edge_density = float((edge_region > 0).mean())

    return np.array([
        mean, std, p10, p50, p90, contrast,
        lap_var, weld_area_frac, weld_aspect, edge_density,
    ], dtype=np.float32)
