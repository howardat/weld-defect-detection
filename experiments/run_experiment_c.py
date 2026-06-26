"""Experiment C: predictor warm-start + GT-free proxy refinement, LOOCV."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weld_pipeline.porosity_data import load_dataset
from weld_pipeline.porosity_pipeline import detect_pores, detection_mask, pixel_f1
from weld_pipeline.porosity_tuner import refine_params
from experiments.build_cache import (
    ANNO_FILE, CACHE_FILE, DATA_DIR, MODEL_PT,
    build_cache, cache_to_matrices, load_cache, save_cache,
)
from experiments.loocv import loocv_predict

RESULTS_FILE = Path(CACHE_FILE).parent / "experiment_c_results.json"


def evaluate_with_refinement(records, warm_preds, refine_trials: int = 15,
                             seed: int = 0) -> list[dict]:
    rows = []
    for rec, warm in zip(records, warm_preds):
        refined = refine_params(rec.gray, rec.weld_mask, warm,
                                n_trials=refine_trials, seed=seed)
        dets = detect_pores(rec.gray, rec.weld_mask, refined)
        f1 = pixel_f1(rec.gt_mask, detection_mask(dets, rec.gray.shape))
        rows.append({"file_name": rec.file_name, "f1": float(f1)})
    return rows


def main(n_trials: int = 300, refine_trials: int = 15) -> None:
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PT))
    records = load_dataset(DATA_DIR, ANNO_FILE, model)

    if Path(CACHE_FILE).exists():
        cache = load_cache()
        print(f"Loaded cache: {CACHE_FILE}")
    else:
        cache = build_cache(records, n_trials=n_trials)
        save_cache(cache)

    by_name = {r.file_name: r for r in records}
    records = [by_name[it["file_name"]] for it in cache["items"]]

    X, Y = cache_to_matrices(cache)
    warm_preds = loocv_predict(X, Y, seed=0)
    rows = evaluate_with_refinement(records, warm_preds, refine_trials=refine_trials)

    mean_f1 = float(np.mean([r["f1"] for r in rows]))
    out = {
        "method": "experiment_c_warmstart_proxy_refine_loocv",
        "mean_f1": mean_f1,
        "refine_trials": refine_trials,
        "per_image": rows,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nExperiment C mean LOOCV F1 = {mean_f1:.4f}")
    print(f"Results → {RESULTS_FILE}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else 300
    rt = int(sys.argv[sys.argv.index("--refine") + 1]) if "--refine" in sys.argv else 15
    main(n_trials=n, refine_trials=rt)
