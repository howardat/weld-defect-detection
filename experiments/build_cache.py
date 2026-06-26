"""Offline Phase 1: per-image Optuna optima + features, cached to JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weld_pipeline.porosity_data import load_dataset
from weld_pipeline.porosity_features import FEATURE_NAMES, extract_features
from weld_pipeline.porosity_predictor import PARAM_ORDER, params_to_vector
from weld_pipeline.porosity_tuner import optimize_image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "porosity_val"
ANNO_FILE = DATA_DIR / "_annotations.coco.json"
MODEL_PT = ROOT / "models" / "wda11s-seg.pt"
CACHE_FILE = ROOT / "data" / "json_output" / "porosity_tuning_cache.json"


def build_cache(records, n_trials: int, seed: int = 0) -> dict:
    items = []
    for r in records:
        params, ceiling_f1 = optimize_image(r.gray, r.weld_mask, r.gt_mask,
                                             n_trials=n_trials, seed=seed)
        feats = extract_features(r.gray, r.weld_mask)
        items.append({
            "file_name": r.file_name,
            "image_id": r.image_id,
            "features": [float(x) for x in feats],
            "params": {name: getattr(params, name) for name in PARAM_ORDER},
            "ceiling_f1": float(ceiling_f1),
        })
        print(f"  {r.file_name[:55]:<55} ceiling_F1={ceiling_f1:.3f}")
    return {"feature_names": FEATURE_NAMES, "param_order": PARAM_ORDER, "items": items}


def cache_to_matrices(cache: dict):
    order = cache["param_order"]
    X = np.array([it["features"] for it in cache["items"]], dtype=np.float32)
    Y = np.array([[it["params"][name] for name in order] for it in cache["items"]],
                 dtype=np.float32)
    return X, Y


def save_cache(cache: dict, path=CACHE_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def load_cache(path=CACHE_FILE) -> dict:
    with open(path) as f:
        return json.load(f)


def main(n_trials: int = 300) -> None:
    from ultralytics import YOLO
    print(f"Loading dataset from {DATA_DIR}")
    model = YOLO(str(MODEL_PT))
    records = load_dataset(DATA_DIR, ANNO_FILE, model)
    print(f"Optimizing {len(records)} images with {n_trials} trials each …")
    cache = build_cache(records, n_trials=n_trials)
    save_cache(cache)
    ceilings = [it["ceiling_f1"] for it in cache["items"]]
    print(f"\nCache → {CACHE_FILE}")
    print(f"Mean per-image ceiling F1 = {np.mean(ceilings):.4f}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else 300
    main(n_trials=n)
