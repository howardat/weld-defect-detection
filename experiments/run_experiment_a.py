"""Experiment A: predictor-only per-image params, evaluated under LOOCV."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weld_pipeline.porosity_data import load_dataset
from weld_pipeline.porosity_pipeline import detect_pores, detection_mask, pixel_f1
from experiments.build_cache import (
    ANNO_FILE, CACHE_FILE, DATA_DIR, MODEL_PT,
    build_cache, cache_to_matrices, load_cache, save_cache,
)
from experiments.loocv import loocv_predict

RESULTS_FILE = Path(CACHE_FILE).parent / "experiment_a_results.json"


def evaluate_predictions(records, preds) -> list[dict]:
    rows = []
    for rec, params in zip(records, preds):
        dets = detect_pores(rec.gray, rec.weld_mask, params)
        f1 = pixel_f1(rec.gt_mask, detection_mask(dets, rec.gray.shape))
        rows.append({"file_name": rec.file_name, "f1": float(f1)})
    return rows


def main(n_trials: int = 300) -> None:
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PT))
    records = load_dataset(DATA_DIR, ANNO_FILE, model)

    if Path(CACHE_FILE).exists():
        cache = load_cache()
        print(f"Loaded cache: {CACHE_FILE}")
    else:
        cache = build_cache(records, n_trials=n_trials)
        save_cache(cache)

    # Align records to cache order by file_name.
    by_name = {r.file_name: r for r in records}
    records = [by_name[it["file_name"]] for it in cache["items"]]

    X, Y = cache_to_matrices(cache)
    preds = loocv_predict(X, Y, seed=0)
    rows = evaluate_predictions(records, preds)

    mean_f1 = float(np.mean([r["f1"] for r in rows]))
    ceilings = [it["ceiling_f1"] for it in cache["items"]]
    out = {
        "method": "experiment_a_predictor_only_loocv",
        "mean_f1": mean_f1,
        "mean_ceiling_f1": float(np.mean(ceilings)),
        "per_image": rows,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nExperiment A mean LOOCV F1 = {mean_f1:.4f}")
    print(f"(per-image Optuna ceiling   = {np.mean(ceilings):.4f}, non-deployable upper bound)")
    print(f"Results → {RESULTS_FILE}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else 300
    main(n_trials=n)
