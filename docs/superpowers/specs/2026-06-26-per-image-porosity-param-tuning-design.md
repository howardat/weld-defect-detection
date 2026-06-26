# Per-Image Porosity Parameter Tuning — Design

**Date:** 2026-06-26
**Status:** Approved design, pending implementation plan
**Branch:** `28-porosity-module-validation`

## Problem

The classical porosity detector in [`porosity_light_check.py`](../../../src/weld_pipeline/porosity_light_check.py) works well when its OpenCV parameters are hand-tuned for a single image. The global grid sweep in [`porosity_sweep.py`](../../../porosity_sweep.py) — which searches one shared `(block_size, adapt_c, open_ksize)` for all images — peaks at ~40% mean F1. Different images need different parameters, and no single global setting fits them all.

Two root causes:

1. **No per-image adaptation.** One parameter set is forced onto 18 visually diverse images.
2. **The sweep optimizes the wrong target.** It computes pixel-F1 on the *raw binary mask* with no contour filtering. The hand-tuned good results come from the full filtered contour pipeline (circularity + darkness + weld-erosion). The optimizer and the production detector must use the *same* filtered pipeline.

## Goal

Build a system that selects near-optimal OpenCV parameters **per image**, learns to generalize that selection from the 18 ground-truth (GT) images, and works at deployment on new images **without GT**. Deliver two experiments and compare them against the 40% baseline.

## Non-Goals

- No real-time/latency constraint for now — accuracy is the only target.
- No new training images beyond the 18 in `data/porosity_val/`.
- Not changing the YOLO weld-detection model.
- Not touching the YOLO-based [`porosity_check.py`](../../../src/weld_pipeline/porosity_check.py) metrology path.

## Dataset

- `data/porosity_val/` — 18 images, COCO segmentation GT in `_annotations.coco.json` (548 pore annotations).
- GT pore mask per image is built by filling COCO segmentation polygons (see `build_gt_mask` in `porosity_sweep.py`).

## The Filtering Pipeline (shared by optimizer and detector)

This exact pipeline is the objective the optimizer maximizes **and** the detector run at inference. Both must call the same code.

**Weld region preparation**
1. YOLO detects weld (class 3) → raw weld mask.
2. **Erode** weld mask inward by `erode_iters` → eroded weld mask (trims bead-edge shadows).

**Binarisation & morphology**
3. Grayscale → adaptive Gaussian threshold (`block_size`, `adapt_c`, `THRESH_BINARY_INV`).
4. Morphological close (fixed 3×3 ellipse, 2 iters) → morphological open (`open_ksize`).
5. **AND** the binary with the **eroded** weld mask from step 2.

**Contour extraction & filtering**
6. `findContours` on the masked binary → candidate pore contours.
7. **Circularity filter** (2nd-to-last): `4π·area/perimeter²`; reject if `< min_circularity`.
8. **Darkness filter** (last): mean-intensity gap between a 15px ring around the contour and its interior; reject if gap `< darkness_thresh`.

Surviving contours = detected pores. Their filled mask is compared to GT for pixel-F1.

**Explicitly excluded filters:** min area, aspect ratio, min diameter fraction.

## Tunable Parameters (6)

| Param | Range | Role |
|---|---|---|
| `block_size` | odd int 11–201 | adaptive threshold neighbourhood |
| `adapt_c` | int 1–350 | adaptive threshold constant |
| `open_ksize` | odd int 1–21 | morphological opening kernel |
| `erode_iters` | int 0–30 | weld-mask inward erosion |
| `min_circularity` | float 0.05–0.95 | circularity filter threshold |
| `darkness_thresh` | int 0–60 | absolute ring-vs-interior intensity gap |

`darkness_thresh` is an **absolute** intensity gap, replacing the old `0.4 × std(weld region)` heuristic, which did not work well. Because the regressor's input features include weld-region std/contrast, it can still learn the per-image relationship and generalize.

## Architecture

```
OFFLINE (one-time, uses GT):
  1. Per-image Optuna  →  optimal 6-param vector per image (maximize pixel-F1 vs GT)
  2. Feature extraction →  weld-region statistics per image
  3. Regressor training →  features → 6 params
  4. LOOCV evaluation   →  honest per-image F1, averaged

INFERENCE (no GT):
  Experiment A:  features → regressor predicts 6 params → run filtered pipeline
  Experiment C:  features → regressor predicts 6 params (warm start)
                 → short Optuna refinement guided by proxy score → run pipeline
```

### Components / new files

- `src/weld_pipeline/porosity_pipeline.py` — the shared filtered pipeline (steps 1–8 above) as a single function `detect_pores(gray, weld_mask, params) -> detection_mask`, plus `pixel_f1(gt_mask, det_mask)`. This is the single source of truth both the optimizer and the detector call. Refactors the filtering logic currently embedded in `porosity_light_check.py` so it is reusable without the matplotlib visualization.
- `src/weld_pipeline/porosity_tuner.py` — Optuna per-image optimizer over the 6-param search space, maximizing pixel-F1 against GT.
- `src/weld_pipeline/porosity_features.py` — extract the per-image feature vector from the YOLO weld region.
- `src/weld_pipeline/porosity_predictor.py` — regressor train / predict / LOOCV harness.
- `experiments/run_experiment_a.py` — end-to-end Experiment A (predictor only) under LOOCV.
- `experiments/run_experiment_c.py` — end-to-end Experiment C (predictor warm-start + proxy refinement) under LOOCV.

`porosity_light_check.py` keeps its interactive tuner but delegates detection to `porosity_pipeline.py` so there is one filtering implementation.

## Image Features (~12)

Extracted from the YOLO weld region only:

- Grayscale intensity: mean, std, p10, p50, p90.
- Local contrast: p90 − p10.
- Noise estimate: variance of Laplacian.
- Weld geometry: weld-region pixel area (normalized by image area), weld bounding-box aspect ratio.
- Edge density: mean Canny edge response inside the weld region.

Exact list may be trimmed during implementation; the count is small by design because the training set is only 18 rows.

## Regressor

- One `RandomForestRegressor` per parameter (6 models) — simpler to reason about and tune than a single multi-output model.
- Integer/odd-valued params are rounded and coerced to valid form (odd, in-range) after prediction.
- Small forests (few trees, shallow) to avoid overfitting 17-sample training folds.

## Experiment A — Predictor Only

1. **Phase 1:** Run Optuna per image (~300 trials) → 18 optimal 6-param vectors + their GT F1 (the per-image *ceiling*).
2. **Phase 2:** Extract features for all 18 images.
3. **Phase 3 + 4 (LOOCV):** For each held-out image *i*: train 6 regressors on the other 17 `(features → param)` rows, predict image *i*'s params, run the filtered pipeline, record F1 vs GT.
4. Report mean held-out F1 across 18 images.

## Experiment C — Predictor Warm-Start + Proxy Refinement

Same offline Phase 1–2. At inference, for each held-out image:

1. Predict 6 params from the LOOCV-trained regressors (warm start).
2. Run a short Optuna refinement (10–20 trials) seeded with the predicted params, **without GT**, maximizing a proxy quality score.
3. Run the filtered pipeline with the refined params, record F1 vs GT.

**Proxy score (no GT):** mean over detected pores of `darkness_contrast × circularity` — both already computed inside the filtering pipeline. Higher = more confident, well-formed dark pores. The refinement search is bounded to a neighborhood around the warm-start params so it cannot wander far from the learned prior.

## Evaluation & Comparison

Single results table:

| Method | Mean F1 (held-out) |
|---|---|
| Baseline global sweep | ~0.40 |
| Per-image Optuna ceiling (Phase 1, uses GT — upper bound, not deployable) | report |
| Experiment A (predictor, LOOCV) | report |
| Experiment C (warm-start + proxy, LOOCV) | report |

Also report per-image F1 breakdown and save 6-panel comparison visualizations (reuse the `visualize_best` style in `porosity_sweep.py`).

### Research-validity note (LOOCV)

With only 18 images, training and testing on the same set leaks data. **Leave-One-Out Cross-Validation** is used for every reported A/C number: each image is predicted by a regressor trained on the other 17, so no test image was seen during training. The per-image Optuna ceiling is *not* deployable (it uses GT) and is reported only as an upper bound.

## Dependencies (new)

- `optuna` — Bayesian per-image optimization.
- `scikit-learn` — RandomForestRegressor + LOOCV utilities.

Both added to `pyproject.toml`. `scipy` (already present) is sufficient for stats helpers if needed.

## Build & Comparison Order

1. Refactor shared pipeline (`porosity_pipeline.py`) and confirm it reproduces a known hand-tuned result.
2. Implement features + tuner + predictor.
3. Implement and run Experiment A; record results.
4. Implement and run Experiment C; record results.
5. Compare A vs C vs baseline in the results table.

## Risks

- **18 samples is small.** The regressor may generalize poorly to images unlike the training set. Experiment C exists precisely to mitigate this via test-time adaptation.
- **Proxy score may not track F1.** If C's proxy is poorly correlated with true F1, refinement can hurt. Mitigated by bounding the refinement to a small neighborhood of the warm-start params.
- **Optuna cost.** Phase 1 is 18 × ~300 trials; acceptable as a one-time offline cost.
