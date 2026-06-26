"""Shared porosity detection pipeline — single source of truth for both the
Optuna optimizer and the deployed detector. No matplotlib, no YOLO at import."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "block_size": (11, 201),
    "adapt_c": (1, 350),
    "open_ksize": (1, 21),
    "erode_iters": (0, 30),
    "min_circularity": (0.05, 0.95),
    "darkness_thresh": (0, 60),
}


@dataclass
class PoreParams:
    block_size: int
    adapt_c: int
    open_ksize: int
    erode_iters: int
    min_circularity: float
    darkness_thresh: float


def _clip(name: str, value: float) -> float:
    lo, hi = PARAM_BOUNDS[name]
    return max(lo, min(hi, value))


def _force_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def sanitize_params(p: PoreParams) -> PoreParams:
    block_size = _force_odd(int(round(_clip("block_size", p.block_size))))
    open_ksize = _force_odd(int(round(_clip("open_ksize", p.open_ksize))))
    return PoreParams(
        block_size=int(min(block_size, int(PARAM_BOUNDS["block_size"][1]))),
        adapt_c=int(round(_clip("adapt_c", p.adapt_c))),
        open_ksize=int(min(open_ksize, int(PARAM_BOUNDS["open_ksize"][1]))),
        erode_iters=int(round(_clip("erode_iters", p.erode_iters))),
        min_circularity=float(_clip("min_circularity", p.min_circularity)),
        darkness_thresh=int(round(_clip("darkness_thresh", p.darkness_thresh))),
    )
