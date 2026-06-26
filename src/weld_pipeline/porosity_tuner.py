"""Optuna search: per-image GT optimization and GT-free proxy refinement."""
from __future__ import annotations

import optuna

from weld_pipeline.porosity_pipeline import (
    PARAM_BOUNDS, PoreParams, detect_pores, detection_mask,
    pixel_f1, proxy_score, sanitize_params,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_params(trial) -> PoreParams:
    return sanitize_params(PoreParams(
        block_size=trial.suggest_int("block_size", 11, 201, step=2),
        adapt_c=trial.suggest_int("adapt_c", 1, 350),
        open_ksize=trial.suggest_int("open_ksize", 1, 21, step=2),
        erode_iters=trial.suggest_int("erode_iters", 0, 30),
        min_circularity=trial.suggest_float("min_circularity", 0.05, 0.95),
        darkness_thresh=trial.suggest_int("darkness_thresh", 0, 120),
    ))


def optimize_image(gray, weld_mask, gt_mask, n_trials: int = 300, seed: int = 0):
    def objective(trial):
        params = suggest_params(trial)
        dets = detect_pores(gray, weld_mask, params)
        return pixel_f1(gt_mask, detection_mask(dets, gray.shape))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    return sanitize_params(PoreParams(**best)), float(study.best_value)


def _neighbor_range(name: str, center: float, neighborhood: float):
    lo, hi = PARAM_BOUNDS[name]
    half = (hi - lo) * neighborhood
    return max(lo, center - half), min(hi, center + half)


def refine_params(gray, weld_mask, warm: PoreParams, n_trials: int = 15,
                  seed: int = 0, neighborhood: float = 0.25) -> PoreParams:
    warm = sanitize_params(warm)

    def objective(trial):
        bs_lo, bs_hi = _neighbor_range("block_size", warm.block_size, neighborhood)
        ac_lo, ac_hi = _neighbor_range("adapt_c", warm.adapt_c, neighborhood)
        ok_lo, ok_hi = _neighbor_range("open_ksize", warm.open_ksize, neighborhood)
        er_lo, er_hi = _neighbor_range("erode_iters", warm.erode_iters, neighborhood)
        mc_lo, mc_hi = _neighbor_range("min_circularity", warm.min_circularity, neighborhood)
        dk_lo, dk_hi = _neighbor_range("darkness_thresh", warm.darkness_thresh, neighborhood)
        params = sanitize_params(PoreParams(
            block_size=trial.suggest_int("block_size", int(bs_lo), int(bs_hi), step=2),
            adapt_c=trial.suggest_int("adapt_c", int(ac_lo), int(ac_hi)),
            open_ksize=trial.suggest_int("open_ksize", int(ok_lo), int(ok_hi), step=2),
            erode_iters=trial.suggest_int("erode_iters", int(er_lo), int(er_hi)),
            min_circularity=trial.suggest_float("min_circularity", mc_lo, mc_hi),
            darkness_thresh=trial.suggest_int("darkness_thresh", int(dk_lo), int(dk_hi)),
        ))
        return proxy_score(detect_pores(gray, weld_mask, params))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    if study.best_value <= 0.0:
        return warm
    return sanitize_params(PoreParams(**study.best_params))
