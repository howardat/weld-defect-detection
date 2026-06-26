# Per-Image Porosity Parameter Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a system that selects near-optimal OpenCV porosity-detection parameters per image, learns to generalize that selection from 18 GT-annotated images, and runs on new images without GT — delivered as two comparable experiments (A: predictor only; C: predictor warm-start + proxy refinement) evaluated under LOOCV against the ~40% global-sweep baseline.

**Architecture:** A single shared filtering pipeline (`porosity_pipeline.py`) is the one source of truth that both the Optuna optimizer and the deployed detector call. Offline, Optuna finds each image's optimal 6-parameter vector against GT; image features and optimal params are cached. A small per-parameter Random Forest learns `features → params`. Experiments A and C are thin orchestration scripts that run Leave-One-Out Cross-Validation over the 18 images and report mean F1.

**Tech Stack:** Python 3.11/3.12, OpenCV, NumPy, Optuna (Bayesian optimization), scikit-learn (RandomForestRegressor), Ultralytics YOLO (weld mask), pytest (TDD). Poetry-managed venv.

## Global Constraints

- Python `>=3.11,<3.13` (from `pyproject.toml`).
- The filtering pipeline is fixed and shared (spec "The Filtering Pipeline"): YOLO weld mask → **erode by `erode_iters`** → adaptive Gaussian threshold (`block_size`,`adapt_c`,`THRESH_BINARY_INV`) → fixed close (3×3 ellipse, 2 iters) → open (`open_ksize`) → **AND with eroded weld mask** → contours → **circularity filter** (2nd-to-last) → **darkness filter** (last). No min-area, no aspect-ratio, no min-diameter-fraction filters.
- Exactly 6 tunable params with these bounds: `block_size` odd 11–201; `adapt_c` int 1–350; `open_ksize` odd 1–21; `erode_iters` int 0–30; `min_circularity` float 0.05–0.95; `darkness_thresh` int 0–120 (absolute ring-vs-interior intensity gap).
- `darkness_thresh` is an absolute gap, NOT the old `0.4×std` heuristic.
- All A/C reported numbers use LOOCV (train on 17, test on the held-out 1). The per-image Optuna ceiling uses GT and is reported only as a non-deployable upper bound.
- Dataset: `data/porosity_val/` (18 images) + `data/porosity_val/_annotations.coco.json` (COCO polygons, category porosity). GT mask = filled segmentation polygons.
- YOLO weld model: `models/wda11s-seg.pt`, class 3, `weld_conf=0.01`.
- pytest is scoped to `tests/porosity/` only (the legacy `tests/test_porosity.py` runs code at import and must NOT be collected).
- Frequent commits: one per task minimum.

---

## File Structure

**Create:**
- `src/weld_pipeline/porosity_pipeline.py` — shared pipeline: `PoreParams`, `sanitize_params`, `erode_weld_mask`, `detect_pores`, `detection_mask`, `pixel_f1`, `proxy_score`, `build_weld_mask`.
- `src/weld_pipeline/porosity_data.py` — `build_gt_mask`, `ImageRecord`, `load_dataset`.
- `src/weld_pipeline/porosity_features.py` — `FEATURE_NAMES`, `extract_features`.
- `src/weld_pipeline/porosity_tuner.py` — `suggest_params`, `optimize_image`, `refine_params`.
- `src/weld_pipeline/porosity_predictor.py` — `PARAM_ORDER`, `params_to_vector`, `vector_to_params`, `train_predictor`, `predict_params`.
- `experiments/__init__.py`, `experiments/run_experiment_a.py`, `experiments/run_experiment_c.py`.
- `tests/porosity/__init__.py` and `tests/porosity/test_*.py` per task.

**Modify:**
- `pyproject.toml` — add `optuna`, `scikit-learn`, dev `pytest`; add `[tool.pytest.ini_options]`.

---

### Task 1: Dependencies, scaffolding, and pytest scoping

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/porosity/__init__.py`
- Create: `tests/porosity/test_smoke.py`

**Interfaces:**
- Produces: an importable `weld_pipeline` package with `optuna`, `sklearn`, `pytest` available; pytest restricted to `tests/porosity/`.

- [ ] **Step 1: Add runtime deps to `pyproject.toml`**

In the `[project]` `dependencies = [ ... ]` list, add these two lines (after `"opencv-python>=4.10.0",`):

```toml
    "optuna>=3.6.0",
    "scikit-learn>=1.4.0",
```

- [ ] **Step 2: Add pytest config + dev group to `pyproject.toml`**

Append to the end of the file:

```toml
[tool.pytest.ini_options]
testpaths = ["tests/porosity"]
addopts = "-q"

[tool.poetry.group.dev.dependencies]
pytest = ">=8.0.0"
```

- [ ] **Step 3: Install the new packages into the venv**

Run: `poetry run pip install "optuna>=3.6.0" "scikit-learn>=1.4.0" "pytest>=8.0.0"`
Expected: ends with `Successfully installed ... optuna-... scikit-learn-... pytest-...` (torch/ultralytics untouched).

- [ ] **Step 4: Create the scoped test package**

Create `tests/porosity/__init__.py` (empty file).

Create `tests/porosity/test_smoke.py`:

```python
def test_dependencies_import():
    import optuna  # noqa: F401
    import sklearn  # noqa: F401
    import cv2  # noqa: F401
    import numpy  # noqa: F401
```

- [ ] **Step 5: Run the smoke test — verify pytest collects ONLY tests/porosity**

Run: `poetry run pytest tests/porosity -v`
Expected: PASS, 1 test, and the legacy `tests/test_porosity.py` is NOT collected (no matplotlib window, no SystemExit).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/porosity/__init__.py tests/porosity/test_smoke.py
git commit -m "chore: add optuna/sklearn/pytest and scope pytest to tests/porosity"
```

---

### Task 2: PoreParams and sanitize_params

**Files:**
- Create: `src/weld_pipeline/porosity_pipeline.py`
- Test: `tests/porosity/test_params.py`

**Interfaces:**
- Produces:
  - `PARAM_BOUNDS: dict[str, tuple[float, float]]` keyed by the 6 param names.
  - `@dataclass PoreParams(block_size:int, adapt_c:int, open_ksize:int, erode_iters:int, min_circularity:float, darkness_thresh:float)`.
  - `sanitize_params(p: PoreParams) -> PoreParams` — clips each field to `PARAM_BOUNDS`; forces `block_size` and `open_ksize` odd; rounds the four integer fields to `int`.

- [ ] **Step 1: Write the failing test**

Create `tests/porosity/test_params.py`:

```python
from weld_pipeline.porosity_pipeline import PoreParams, sanitize_params, PARAM_BOUNDS


def test_param_bounds_has_six_keys():
    assert set(PARAM_BOUNDS) == {
        "block_size", "adapt_c", "open_ksize",
        "erode_iters", "min_circularity", "darkness_thresh",
    }


def test_sanitize_forces_block_size_odd_and_in_range():
    p = sanitize_params(PoreParams(50, 10, 4, 5, 0.3, 12))
    assert p.block_size % 2 == 1
    assert 11 <= p.block_size <= 201


def test_sanitize_forces_open_ksize_odd():
    p = sanitize_params(PoreParams(51, 10, 8, 5, 0.3, 12))
    assert p.open_ksize % 2 == 1


def test_sanitize_clips_out_of_range():
    p = sanitize_params(PoreParams(9, 999, 99, 99, 5.0, 999))
    assert p.block_size == 11
    assert p.adapt_c == 350
    assert p.open_ksize == 21
    assert p.erode_iters == 30
    assert p.min_circularity == 0.95
    assert p.darkness_thresh == 120


def test_sanitize_coerces_integer_fields():
    p = sanitize_params(PoreParams(51.0, 10.9, 5.0, 5.4, 0.3, 12.7))
    assert isinstance(p.block_size, int)
    assert isinstance(p.adapt_c, int)
    assert isinstance(p.open_ksize, int)
    assert isinstance(p.erode_iters, int)
    assert isinstance(p.darkness_thresh, int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/porosity/test_params.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'weld_pipeline.porosity_pipeline'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/weld_pipeline/porosity_pipeline.py`:

```python
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
    "darkness_thresh": (0, 120),
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/porosity/test_params.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/weld_pipeline/porosity_pipeline.py tests/porosity/test_params.py
git commit -m "feat: add PoreParams and sanitize_params to porosity_pipeline"
```

---

### Task 3: pixel_f1 metric

**Files:**
- Modify: `src/weld_pipeline/porosity_pipeline.py`
- Test: `tests/porosity/test_metrics.py`

**Interfaces:**
- Produces: `pixel_f1(gt_mask: np.ndarray, det_mask: np.ndarray) -> float` — pixel-level F1 over boolean masks (`>0` is positive). Both empty → returns `1.0`; det empty but gt non-empty → `0.0`.

- [ ] **Step 1: Write the failing test**

Create `tests/porosity/test_metrics.py`:

```python
import numpy as np

from weld_pipeline.porosity_pipeline import pixel_f1


def test_perfect_overlap_is_one():
    m = np.zeros((10, 10), np.uint8)
    m[2:6, 2:6] = 255
    assert pixel_f1(m, m) == 1.0


def test_no_overlap_is_zero():
    gt = np.zeros((10, 10), np.uint8); gt[0:3, 0:3] = 255
    det = np.zeros((10, 10), np.uint8); det[7:10, 7:10] = 255
    assert pixel_f1(gt, det) == 0.0


def test_both_empty_is_one():
    z = np.zeros((10, 10), np.uint8)
    assert pixel_f1(z, z) == 1.0


def test_det_empty_gt_nonempty_is_zero():
    gt = np.zeros((10, 10), np.uint8); gt[0:3, 0:3] = 255
    det = np.zeros((10, 10), np.uint8)
    assert pixel_f1(gt, det) == 0.0


def test_partial_overlap_between_zero_and_one():
    gt = np.zeros((10, 10), np.uint8); gt[0:4, 0:4] = 255   # 16 px
    det = np.zeros((10, 10), np.uint8); det[0:4, 0:2] = 255  # 8 px, all TP
    f1 = pixel_f1(gt, det)
    assert 0.0 < f1 < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/porosity/test_metrics.py -v`
Expected: FAIL with `ImportError: cannot import name 'pixel_f1'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/weld_pipeline/porosity_pipeline.py`:

```python
def pixel_f1(gt_mask: np.ndarray, det_mask: np.ndarray) -> float:
    gt_b = gt_mask > 0
    det_b = det_mask > 0
    tp = int((gt_b & det_b).sum())
    fp = int((det_b & ~gt_b).sum())
    fn = int((gt_b & ~det_b).sum())
    if tp + fp == 0 and tp + fn == 0:
        return 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/porosity/test_metrics.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/weld_pipeline/porosity_pipeline.py tests/porosity/test_metrics.py
git commit -m "feat: add pixel_f1 metric"
```

---

### Task 4: erode_weld_mask, detect_pores, detection_mask

**Files:**
- Modify: `src/weld_pipeline/porosity_pipeline.py`
- Test: `tests/porosity/test_detect.py`

**Interfaces:**
- Consumes: `PoreParams`, `sanitize_params`.
- Produces:
  - `@dataclass PoreDetection(contour: np.ndarray, circularity: float, darkness_contrast: float)`.
  - `erode_weld_mask(weld_mask: np.ndarray, erode_iters: int) -> np.ndarray` — ellipse 5×5, `erode_iters` iterations; `erode_iters<=0` returns the mask unchanged.
  - `detect_pores(gray: np.ndarray, weld_mask: np.ndarray, params: PoreParams) -> list[PoreDetection]` — runs the full shared pipeline (erode → adaptive threshold → close → open → AND eroded mask → contours → circularity filter → darkness filter). A `weld_mask` that is all zeros is treated as "no restriction" (skip the AND).
  - `detection_mask(detections: list[PoreDetection], shape: tuple[int, int]) -> np.ndarray` — uint8 mask with all detection contours filled to 255.

- [ ] **Step 1: Write the failing test**

Create `tests/porosity/test_detect.py`:

```python
import cv2
import numpy as np

from weld_pipeline.porosity_pipeline import (
    PoreParams, detect_pores, detection_mask, erode_weld_mask,
)


def _synthetic_pore_image(size=200):
    """Light-gray background with two dark circular 'pores'."""
    gray = np.full((size, size), 200, np.uint8)
    cv2.circle(gray, (60, 60), 12, 30, -1)
    cv2.circle(gray, (140, 140), 14, 25, -1)
    return gray


def _full_weld_mask(size=200):
    return np.full((size, size), 255, np.uint8)


PARAMS = PoreParams(
    block_size=51, adapt_c=10, open_ksize=3,
    erode_iters=0, min_circularity=0.5, darkness_thresh=20,
)


def test_erode_zero_iters_is_identity():
    m = _full_weld_mask()
    assert np.array_equal(erode_weld_mask(m, 0), m)


def test_erode_shrinks_mask():
    m = np.zeros((100, 100), np.uint8)
    m[20:80, 20:80] = 255
    eroded = erode_weld_mask(m, 5)
    assert eroded.sum() < m.sum()


def test_detect_finds_two_pores():
    gray = _synthetic_pore_image()
    dets = detect_pores(gray, _full_weld_mask(), PARAMS)
    assert len(dets) == 2


def test_detection_mask_marks_pixels():
    gray = _synthetic_pore_image()
    dets = detect_pores(gray, _full_weld_mask(), PARAMS)
    mask = detection_mask(dets, gray.shape)
    assert mask.shape == gray.shape
    assert mask.max() == 255
    assert mask[60, 60] == 255  # center of first pore


def test_high_circularity_threshold_keeps_round_pores():
    gray = _synthetic_pore_image()
    strict = PoreParams(51, 10, 3, 0, 0.85, 20)
    dets = detect_pores(gray, _full_weld_mask(), strict)
    assert len(dets) >= 1


def test_darkness_threshold_rejects_faint_pore():
    # Faint pore: interior 150 on a 200 background → contrast gap ~50.
    size = 200
    gray = np.full((size, size), 200, np.uint8)
    cv2.circle(gray, (100, 100), 14, 150, -1)
    weld = np.full((size, size), 255, np.uint8)
    lenient = PoreParams(51, 10, 3, 0, 0.5, 20)    # gap>=20 keeps it
    strict = PoreParams(51, 10, 3, 0, 0.5, 120)    # gap>=120 (in-bounds) rejects it
    assert len(detect_pores(gray, weld, lenient)) >= 1
    assert len(detect_pores(gray, weld, strict)) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/porosity/test_detect.py -v`
Expected: FAIL with `ImportError: cannot import name 'detect_pores'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/weld_pipeline/porosity_pipeline.py`:

```python
_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_RING_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
_RING_PAD = 15


@dataclass
class PoreDetection:
    contour: np.ndarray
    circularity: float
    darkness_contrast: float


def erode_weld_mask(weld_mask: np.ndarray, erode_iters: int) -> np.ndarray:
    if erode_iters <= 0:
        return weld_mask
    return cv2.erode(weld_mask, _ERODE_KERNEL, iterations=int(erode_iters))


def _binarize(gray: np.ndarray, params: PoreParams) -> np.ndarray:
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, params.block_size, params.adapt_c,
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _CLOSE_KERNEL, iterations=2)
    if params.open_ksize > 1:
        ok = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.open_ksize, params.open_ksize))
        return cv2.morphologyEx(closed, cv2.MORPH_OPEN, ok)
    return closed


def _darkness_contrast(gray: np.ndarray, contour: np.ndarray) -> float:
    bx, by, bw, bh = cv2.boundingRect(contour)
    x0 = max(0, bx - _RING_PAD); y0 = max(0, by - _RING_PAD)
    x1 = min(gray.shape[1], bx + bw + _RING_PAD); y1 = min(gray.shape[0], by + bh + _RING_PAD)
    crop = gray[y0:y1, x0:x1]
    local = contour - np.array([[[x0, y0]]])
    roi = np.zeros(crop.shape, np.uint8)
    cv2.drawContours(roi, [local], -1, 255, cv2.FILLED)
    mean_inside = cv2.mean(crop, mask=roi)[0]
    ring = cv2.subtract(cv2.dilate(roi, _RING_KERNEL), roi)
    if cv2.countNonZero(ring) == 0:
        return 0.0
    return cv2.mean(crop, mask=ring)[0] - mean_inside


def detect_pores(gray: np.ndarray, weld_mask: np.ndarray, params: PoreParams) -> list[PoreDetection]:
    p = sanitize_params(params)
    opened = _binarize(gray, p)
    if weld_mask is not None and weld_mask.any():
        eroded = erode_weld_mask(weld_mask, p.erode_iters)
        if eroded.any():
            opened = cv2.bitwise_and(opened, eroded)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[PoreDetection] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / perim ** 2) if perim > 0 else 0.0
        if circularity < p.min_circularity:        # 2nd-to-last filter
            continue
        contrast = _darkness_contrast(gray, cnt)
        if contrast < p.darkness_thresh:            # last filter
            continue
        detections.append(PoreDetection(cnt, float(circularity), float(contrast)))
    return detections


def detection_mask(detections: list[PoreDetection], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    for d in detections:
        cv2.drawContours(mask, [d.contour], -1, 255, cv2.FILLED)
    return mask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/porosity/test_detect.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add src/weld_pipeline/porosity_pipeline.py tests/porosity/test_detect.py
git commit -m "feat: add detect_pores filtering pipeline and detection_mask"
```

---

### Task 5: proxy_score

**Files:**
- Modify: `src/weld_pipeline/porosity_pipeline.py`
- Test: `tests/porosity/test_proxy.py`

**Interfaces:**
- Consumes: `PoreDetection`.
- Produces: `proxy_score(detections: list[PoreDetection]) -> float` — GT-free quality signal = mean over detections of `circularity * (darkness_contrast / 255)`, clamped non-negative; returns `0.0` for an empty list. Used as Experiment C's refinement objective.

- [ ] **Step 1: Write the failing test**

Create `tests/porosity/test_proxy.py`:

```python
import numpy as np

from weld_pipeline.porosity_pipeline import PoreDetection, proxy_score


def _det(circ, contrast):
    return PoreDetection(contour=np.zeros((1, 1, 2), np.int32),
                         circularity=circ, darkness_contrast=contrast)


def test_empty_is_zero():
    assert proxy_score([]) == 0.0


def test_more_circular_and_darker_scores_higher():
    weak = [_det(0.3, 20)]
    strong = [_det(0.9, 120)]
    assert proxy_score(strong) > proxy_score(weak)


def test_negative_contrast_clamped():
    assert proxy_score([_det(0.9, -50)]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/porosity/test_proxy.py -v`
Expected: FAIL with `ImportError: cannot import name 'proxy_score'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/weld_pipeline/porosity_pipeline.py`:

```python
def proxy_score(detections: list[PoreDetection]) -> float:
    if not detections:
        return 0.0
    scores = [
        d.circularity * max(0.0, d.darkness_contrast) / 255.0
        for d in detections
    ]
    return float(np.mean(scores))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/porosity/test_proxy.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/weld_pipeline/porosity_pipeline.py tests/porosity/test_proxy.py
git commit -m "feat: add GT-free proxy_score for test-time refinement"
```

---

### Task 6: build_weld_mask (YOLO refactor, single source of truth)

**Files:**
- Modify: `src/weld_pipeline/porosity_pipeline.py`
- Test: `tests/porosity/test_weld_mask.py`

**Interfaces:**
- Produces:
  - `postprocess_weld_mask(mask: np.ndarray) -> np.ndarray` — pure: closes intra-weld gaps (kernel sized so bridgeable area ≤ 0.5% of weld area), fills enclosed holes.
  - `build_weld_mask(img_rgb: np.ndarray, seg_model, weld_conf: float = 0.01) -> np.ndarray` — runs YOLO class-3 inference, ORs masks (or boxes), then `postprocess_weld_mask`. `seg_model` is a loaded Ultralytics `YOLO` (or any object with a compatible `.predict`).

- [ ] **Step 1: Write the failing test**

Create `tests/porosity/test_weld_mask.py`:

```python
import numpy as np

from weld_pipeline.porosity_pipeline import build_weld_mask, postprocess_weld_mask


def test_postprocess_fills_enclosed_hole():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 255
    mask[45:55, 45:55] = 0          # hole in the middle
    filled = postprocess_weld_mask(mask)
    assert filled[50, 50] == 255    # hole filled


def test_postprocess_empty_stays_empty():
    mask = np.zeros((50, 50), np.uint8)
    assert postprocess_weld_mask(mask).sum() == 0


class _FakeResult:
    masks = None
    class _Boxes:
        def __init__(self, xyxy): self._xyxy = xyxy
        def __iter__(self):
            for b in self._xyxy:
                yield type("B", (), {"xyxy": [type("T", (), {"cpu": lambda s: type("N", (), {"numpy": lambda s2: np.array(b)})()})()]})()
    def __init__(self, boxes): self.boxes = self._Boxes(boxes)


class _FakeModel:
    def predict(self, img, conf, classes, verbose):
        return [_FakeResult([[10, 10, 40, 40]])]


def test_build_weld_mask_from_boxes():
    img = np.zeros((60, 60, 3), np.uint8)
    mask = build_weld_mask(img, _FakeModel(), weld_conf=0.01)
    assert mask[25, 25] == 255      # inside the box
    assert mask[5, 5] == 0          # outside the box
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/porosity/test_weld_mask.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_weld_mask'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/weld_pipeline/porosity_pipeline.py`:

```python
def postprocess_weld_mask(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    weld_area = sum(cv2.contourArea(c) for c in cnts)
    r = int(np.sqrt(0.005 * weld_area / np.pi)) if weld_area > 0 else 0
    ksize = 2 * r + 1
    if ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
    return filled


def build_weld_mask(img_rgb: np.ndarray, seg_model, weld_conf: float = 0.01) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    results = seg_model.predict(img_rgb, conf=weld_conf, classes=[3], verbose=False)
    r = results[0]
    if getattr(r, "masks", None) is not None:
        for mt in r.masks.data:
            m = cv2.resize(mt.cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
            mask = cv2.bitwise_or(mask, (m > 0.5).astype(np.uint8) * 255)
    elif getattr(r, "boxes", None) is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, cv2.FILLED)
    return postprocess_weld_mask(mask)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/porosity/test_weld_mask.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/weld_pipeline/porosity_pipeline.py tests/porosity/test_weld_mask.py
git commit -m "feat: add build_weld_mask/postprocess_weld_mask to shared pipeline"
```

---

### Task 7: COCO GT masks and dataset loading

**Files:**
- Create: `src/weld_pipeline/porosity_data.py`
- Test: `tests/porosity/test_data.py`

**Interfaces:**
- Consumes: `build_weld_mask`.
- Produces:
  - `build_gt_mask(annotations: list[dict], image_id: int, h: int, w: int) -> np.ndarray` — fills COCO segmentation polygons for the given `image_id` to 255.
  - `@dataclass ImageRecord(image_id:int, file_name:str, gray:np.ndarray, weld_mask:np.ndarray, gt_mask:np.ndarray)`.
  - `load_dataset(data_dir, anno_file, seg_model, weld_conf=0.01) -> list[ImageRecord]` — reads COCO, loads each image (grayscale), computes weld mask via `build_weld_mask`, builds GT mask; skips missing/unreadable files.

- [ ] **Step 1: Write the failing test**

Create `tests/porosity/test_data.py`:

```python
import numpy as np

from weld_pipeline.porosity_data import build_gt_mask


def test_build_gt_mask_fills_polygon():
    anns = [{
        "image_id": 0,
        "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],  # square
    }]
    mask = build_gt_mask(anns, image_id=0, h=50, w=50)
    assert mask[20, 20] == 255
    assert mask[0, 0] == 0


def test_build_gt_mask_ignores_other_images():
    anns = [{
        "image_id": 5,
        "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
    }]
    mask = build_gt_mask(anns, image_id=0, h=50, w=50)
    assert mask.sum() == 0


def test_build_gt_mask_handles_multiple_polygons_per_annotation():
    anns = [{
        "image_id": 0,
        "segmentation": [
            [2, 2, 8, 2, 8, 8, 2, 8],
            [40, 40, 46, 40, 46, 46, 40, 46],
        ],
    }]
    mask = build_gt_mask(anns, image_id=0, h=50, w=50)
    assert mask[5, 5] == 255
    assert mask[43, 43] == 255
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/porosity/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'weld_pipeline.porosity_data'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/weld_pipeline/porosity_data.py`:

```python
"""Load the porosity validation dataset (images + COCO GT + YOLO weld masks)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from weld_pipeline.porosity_pipeline import build_weld_mask


def build_gt_mask(annotations: list[dict], image_id: int, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    for ann in annotations:
        if ann["image_id"] != image_id:
            continue
        for seg in ann["segmentation"]:
            pts = np.array(seg, np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    gray: np.ndarray
    weld_mask: np.ndarray
    gt_mask: np.ndarray


def load_dataset(data_dir, anno_file, seg_model, weld_conf: float = 0.01) -> list[ImageRecord]:
    data_dir = Path(data_dir)
    with open(anno_file) as f:
        coco = json.load(f)
    images_meta = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]

    records: list[ImageRecord] = []
    for img_id, meta in images_meta.items():
        img_path = data_dir / meta["file_name"]
        if not img_path.exists():
            print(f"  [WARN] missing: {img_path.name} — skipping")
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [WARN] unreadable: {img_path.name} — skipping")
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        records.append(ImageRecord(
            image_id=img_id,
            file_name=meta["file_name"],
            gray=gray,
            weld_mask=build_weld_mask(img_rgb, seg_model, weld_conf),
            gt_mask=build_gt_mask(annotations, img_id, h, w),
        ))
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/porosity/test_data.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/weld_pipeline/porosity_data.py tests/porosity/test_data.py
git commit -m "feat: add build_gt_mask and load_dataset"
```

---

### Task 8: Image feature extraction

**Files:**
- Create: `src/weld_pipeline/porosity_features.py`
- Test: `tests/porosity/test_features.py`

**Interfaces:**
- Produces:
  - `FEATURE_NAMES: list[str]` — the 10 feature names in fixed order: `["mean","std","p10","p50","p90","contrast","lap_var","weld_area_frac","weld_aspect","edge_density"]`.
  - `extract_features(gray: np.ndarray, weld_mask: np.ndarray) -> np.ndarray` — shape `(10,)` float32. Stats computed over weld-region pixels; if `weld_mask` is empty, over the whole image.

- [ ] **Step 1: Write the failing test**

Create `tests/porosity/test_features.py`:

```python
import numpy as np

from weld_pipeline.porosity_features import FEATURE_NAMES, extract_features


def test_feature_vector_length_matches_names():
    gray = np.full((50, 50), 128, np.uint8)
    weld = np.full((50, 50), 255, np.uint8)
    feats = extract_features(gray, weld)
    assert feats.shape == (len(FEATURE_NAMES),)
    assert len(FEATURE_NAMES) == 10


def test_constant_image_has_zero_std_and_contrast():
    gray = np.full((50, 50), 100, np.uint8)
    weld = np.full((50, 50), 255, np.uint8)
    feats = dict(zip(FEATURE_NAMES, extract_features(gray, weld)))
    assert feats["std"] == 0.0
    assert feats["contrast"] == 0.0
    assert feats["mean"] == 100.0


def test_empty_weld_mask_falls_back_to_whole_image():
    gray = np.full((40, 40), 80, np.uint8)
    empty = np.zeros((40, 40), np.uint8)
    feats = dict(zip(FEATURE_NAMES, extract_features(gray, empty)))
    assert feats["mean"] == 80.0


def test_weld_area_frac_is_between_zero_and_one():
    gray = np.full((100, 100), 128, np.uint8)
    weld = np.zeros((100, 100), np.uint8)
    weld[0:50, :] = 255   # half the image
    feats = dict(zip(FEATURE_NAMES, extract_features(gray, weld)))
    assert 0.4 < feats["weld_area_frac"] < 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/porosity/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'weld_pipeline.porosity_features'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/weld_pipeline/porosity_features.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/porosity/test_features.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/weld_pipeline/porosity_features.py tests/porosity/test_features.py
git commit -m "feat: add weld-region feature extraction"
```

---

### Task 9: Optuna per-image optimizer and proxy refinement

**Files:**
- Create: `src/weld_pipeline/porosity_tuner.py`
- Test: `tests/porosity/test_tuner.py`

**Interfaces:**
- Consumes: `PoreParams`, `PARAM_BOUNDS`, `sanitize_params`, `detect_pores`, `detection_mask`, `pixel_f1`, `proxy_score`.
- Produces:
  - `suggest_params(trial) -> PoreParams` — samples the 6-param search space from an Optuna trial using `PARAM_BOUNDS`.
  - `optimize_image(gray, weld_mask, gt_mask, n_trials=300, seed=0) -> tuple[PoreParams, float]` — maximizes `pixel_f1` vs GT; returns `(best_params, best_f1)`.
  - `refine_params(gray, weld_mask, warm: PoreParams, n_trials=15, seed=0, neighborhood=0.25) -> PoreParams` — GT-free; maximizes `proxy_score` over a bounded neighborhood (±`neighborhood`×range, clipped to `PARAM_BOUNDS`) around `warm`; returns best params (falls back to `warm` if no detections ever occur).

- [ ] **Step 1: Write the failing test**

Create `tests/porosity/test_tuner.py`:

```python
import cv2
import numpy as np

from weld_pipeline.porosity_pipeline import PoreParams, detect_pores, detection_mask, pixel_f1
from weld_pipeline.porosity_tuner import optimize_image, refine_params


def _scene(size=200):
    gray = np.full((size, size), 200, np.uint8)
    gt = np.zeros((size, size), np.uint8)
    for cx, cy, r in [(60, 60, 12), (140, 140, 14)]:
        cv2.circle(gray, (cx, cy), r, 25, -1)
        cv2.circle(gt, (cx, cy), r, 255, -1)
    weld = np.full((size, size), 255, np.uint8)
    return gray, weld, gt


def test_optimize_returns_valid_params_and_f1():
    gray, weld, gt = _scene()
    params, f1 = optimize_image(gray, weld, gt, n_trials=25, seed=0)
    assert isinstance(params, PoreParams)
    assert 0.0 <= f1 <= 1.0
    assert params.block_size % 2 == 1


def test_optimize_beats_a_deliberately_bad_param_set():
    gray, weld, gt = _scene()
    params, f1 = optimize_image(gray, weld, gt, n_trials=40, seed=0)
    bad = PoreParams(11, 1, 21, 30, 0.95, 60)
    bad_f1 = pixel_f1(gt, detection_mask(detect_pores(gray, weld, bad), gray.shape))
    assert f1 >= bad_f1


def test_refine_returns_valid_params():
    gray, weld, _ = _scene()
    warm = PoreParams(51, 10, 3, 0, 0.5, 20)
    refined = refine_params(gray, weld, warm, n_trials=10, seed=0)
    assert isinstance(refined, PoreParams)
    assert refined.block_size % 2 == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/porosity/test_tuner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'weld_pipeline.porosity_tuner'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/weld_pipeline/porosity_tuner.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/porosity/test_tuner.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/weld_pipeline/porosity_tuner.py tests/porosity/test_tuner.py
git commit -m "feat: add Optuna per-image optimizer and proxy refinement"
```

---

### Task 10: Random Forest parameter predictor

**Files:**
- Create: `src/weld_pipeline/porosity_predictor.py`
- Test: `tests/porosity/test_predictor.py`

**Interfaces:**
- Consumes: `PoreParams`, `sanitize_params`.
- Produces:
  - `PARAM_ORDER: list[str]` — `["block_size","adapt_c","open_ksize","erode_iters","min_circularity","darkness_thresh"]`.
  - `params_to_vector(p: PoreParams) -> np.ndarray` — shape `(6,)` in `PARAM_ORDER`.
  - `vector_to_params(v: np.ndarray) -> PoreParams` — inverse, then `sanitize_params`.
  - `train_predictor(X: np.ndarray, Y: np.ndarray, seed=0) -> list` — one `RandomForestRegressor` per param column of `Y` (shape `(n,6)`).
  - `predict_params(models: list, x: np.ndarray) -> PoreParams` — predicts each param from feature vector `x`, assembles + sanitizes.

- [ ] **Step 1: Write the failing test**

Create `tests/porosity/test_predictor.py`:

```python
import numpy as np

from weld_pipeline.porosity_pipeline import PoreParams
from weld_pipeline.porosity_predictor import (
    PARAM_ORDER, params_to_vector, vector_to_params,
    train_predictor, predict_params,
)


def test_param_vector_roundtrip():
    p = PoreParams(51, 10, 3, 5, 0.3, 20)
    v = params_to_vector(p)
    assert v.shape == (6,)
    back = vector_to_params(v)
    assert back.block_size == 51
    assert back.darkness_thresh == 20


def test_param_order_is_six():
    assert len(PARAM_ORDER) == 6


def test_train_and_predict_returns_pore_params():
    rng = np.random.default_rng(0)
    X = rng.random((12, 10)).astype(np.float32)
    Y = np.column_stack([
        rng.integers(11, 201, 12),   # block_size
        rng.integers(1, 350, 12),    # adapt_c
        rng.integers(1, 21, 12),     # open_ksize
        rng.integers(0, 30, 12),     # erode_iters
        rng.uniform(0.05, 0.95, 12), # min_circularity
        rng.integers(0, 120, 12),    # darkness_thresh
    ]).astype(np.float32)
    models = train_predictor(X, Y, seed=0)
    assert len(models) == 6
    pred = predict_params(models, X[0])
    assert isinstance(pred, PoreParams)
    assert pred.block_size % 2 == 1
    assert 11 <= pred.block_size <= 201
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/porosity/test_predictor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'weld_pipeline.porosity_predictor'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/weld_pipeline/porosity_predictor.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/porosity/test_predictor.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/weld_pipeline/porosity_predictor.py tests/porosity/test_predictor.py
git commit -m "feat: add Random Forest parameter predictor"
```

---

### Task 11: LOOCV harness (shared by both experiments)

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/loocv.py`
- Test: `tests/porosity/test_loocv.py`

**Interfaces:**
- Consumes: `train_predictor`, `predict_params`.
- Produces:
  - `loocv_indices(n: int) -> list[tuple[list[int], int]]` — for each `i`, `(train_indices_without_i, i)`.
  - `loocv_predict(X: np.ndarray, Y: np.ndarray, seed=0) -> list[PoreParams]` — for each held-out row `i`, train on the rest and predict params for `i`; returns a list of `PoreParams` in index order. (Pipeline/F1 evaluation happens in the experiment scripts, which own the image data.)

- [ ] **Step 1: Write the failing test**

Create `tests/porosity/test_loocv.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/porosity/test_loocv.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/__init__.py` (empty file).

Create `experiments/loocv.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/porosity/test_loocv.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add experiments/__init__.py experiments/loocv.py tests/porosity/test_loocv.py
git commit -m "feat: add LOOCV harness"
```

---

### Task 12: Offline Phase-1 cache builder

**Files:**
- Create: `experiments/build_cache.py`
- Test: `tests/porosity/test_cache_shapes.py`

**Interfaces:**
- Consumes: `load_dataset`, `optimize_image`, `extract_features`, `params_to_vector`, `FEATURE_NAMES`, `PARAM_ORDER`.
- Produces:
  - `build_cache(records, n_trials, seed=0) -> dict` — runs per-image Optuna + feature extraction; returns `{"feature_names":[...], "param_order":[...], "items":[{file_name, image_id, features:[...], params:{...}, ceiling_f1:float}, ...]}`.
  - `cache_to_matrices(cache: dict) -> tuple[np.ndarray, np.ndarray]` — returns `(X, Y)` with `X` shape `(n,10)`, `Y` shape `(n,6)`.
  - `save_cache(cache, path)` / `load_cache(path) -> dict`.
  - Constants: `DATA_DIR`, `ANNO_FILE`, `MODEL_PT`, `CACHE_FILE` (resolved from repo root).

- [ ] **Step 1: Write the failing test** (uses a fake cache, no YOLO/images needed)

Create `tests/porosity/test_cache_shapes.py`:

```python
import numpy as np

from experiments.build_cache import cache_to_matrices


def test_cache_to_matrices_shapes():
    cache = {
        "feature_names": ["mean", "std", "p10", "p50", "p90",
                          "contrast", "lap_var", "weld_area_frac",
                          "weld_aspect", "edge_density"],
        "param_order": ["block_size", "adapt_c", "open_ksize",
                        "erode_iters", "min_circularity", "darkness_thresh"],
        "items": [
            {"file_name": "a.jpg", "image_id": 0,
             "features": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             "params": {"block_size": 51, "adapt_c": 10, "open_ksize": 3,
                        "erode_iters": 5, "min_circularity": 0.3, "darkness_thresh": 20},
             "ceiling_f1": 0.8},
            {"file_name": "b.jpg", "image_id": 1,
             "features": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
             "params": {"block_size": 31, "adapt_c": 20, "open_ksize": 5,
                        "erode_iters": 2, "min_circularity": 0.5, "darkness_thresh": 15},
             "ceiling_f1": 0.7},
        ],
    }
    X, Y = cache_to_matrices(cache)
    assert X.shape == (2, 10)
    assert Y.shape == (2, 6)
    assert Y[0, 0] == 51  # block_size of first item
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/porosity/test_cache_shapes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.build_cache'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/build_cache.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/porosity/test_cache_shapes.py -v`
Expected: PASS, 1 test.

- [ ] **Step 5: Commit**

```bash
git add experiments/build_cache.py tests/porosity/test_cache_shapes.py
git commit -m "feat: add offline Phase-1 cache builder"
```

---

### Task 13: Experiment A (predictor only, LOOCV)

**Files:**
- Create: `experiments/run_experiment_a.py`
- Test: `tests/porosity/test_experiment_a.py`

**Interfaces:**
- Consumes: `load_cache`, `cache_to_matrices`, `loocv_predict`, `detect_pores`, `detection_mask`, `pixel_f1`, `load_dataset`.
- Produces:
  - `evaluate_predictions(records, preds) -> list[dict]` — per-record `{file_name, f1}` from running `detect_pores` with the predicted params and comparing to GT. `records` and `preds` are index-aligned.
  - `main(n_trials=300)` — builds/loads cache, runs LOOCV, evaluates, writes `data/json_output/experiment_a_results.json`, prints mean F1.

- [ ] **Step 1: Write the failing test** (pure `evaluate_predictions`, synthetic records)

Create `tests/porosity/test_experiment_a.py`:

```python
import cv2
import numpy as np

from weld_pipeline.porosity_data import ImageRecord
from weld_pipeline.porosity_pipeline import PoreParams
from experiments.run_experiment_a import evaluate_predictions


def _record():
    size = 200
    gray = np.full((size, size), 200, np.uint8)
    gt = np.zeros((size, size), np.uint8)
    cv2.circle(gray, (100, 100), 14, 25, -1)
    cv2.circle(gt, (100, 100), 14, 255, -1)
    weld = np.full((size, size), 255, np.uint8)
    return ImageRecord(0, "x.jpg", gray, weld, gt)


def test_evaluate_predictions_reports_f1_per_record():
    rec = _record()
    good = PoreParams(51, 10, 3, 0, 0.5, 20)
    rows = evaluate_predictions([rec], [good])
    assert len(rows) == 1
    assert rows[0]["file_name"] == "x.jpg"
    assert 0.0 <= rows[0]["f1"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/porosity/test_experiment_a.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.run_experiment_a'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/run_experiment_a.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/porosity/test_experiment_a.py -v`
Expected: PASS, 1 test.

- [ ] **Step 5: Commit**

```bash
git add experiments/run_experiment_a.py tests/porosity/test_experiment_a.py
git commit -m "feat: add Experiment A (predictor-only LOOCV)"
```

---

### Task 14: Experiment C (warm-start + proxy refinement, LOOCV)

**Files:**
- Create: `experiments/run_experiment_c.py`
- Test: `tests/porosity/test_experiment_c.py`

**Interfaces:**
- Consumes: `loocv_predict`, `refine_params`, `detect_pores`, `detection_mask`, `pixel_f1`, cache helpers.
- Produces:
  - `evaluate_with_refinement(records, warm_preds, refine_trials=15, seed=0) -> list[dict]` — for each record: refine the warm-start params via GT-free `refine_params`, run the pipeline, compute F1 vs GT; returns `{file_name, f1}` per record.
  - `main(n_trials=300, refine_trials=15)` — loads cache, LOOCV warm-start predictions, refinement eval, writes `data/json_output/experiment_c_results.json`, prints mean F1.

- [ ] **Step 1: Write the failing test**

Create `tests/porosity/test_experiment_c.py`:

```python
import cv2
import numpy as np

from weld_pipeline.porosity_data import ImageRecord
from weld_pipeline.porosity_pipeline import PoreParams
from experiments.run_experiment_c import evaluate_with_refinement


def _record():
    size = 200
    gray = np.full((size, size), 200, np.uint8)
    gt = np.zeros((size, size), np.uint8)
    cv2.circle(gray, (100, 100), 14, 25, -1)
    cv2.circle(gt, (100, 100), 14, 255, -1)
    weld = np.full((size, size), 255, np.uint8)
    return ImageRecord(0, "x.jpg", gray, weld, gt)


def test_evaluate_with_refinement_reports_f1():
    rec = _record()
    warm = PoreParams(51, 10, 3, 0, 0.5, 20)
    rows = evaluate_with_refinement([rec], [warm], refine_trials=8, seed=0)
    assert len(rows) == 1
    assert 0.0 <= rows[0]["f1"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/porosity/test_experiment_c.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.run_experiment_c'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/run_experiment_c.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/porosity/test_experiment_c.py -v`
Expected: PASS, 1 test.

- [ ] **Step 5: Commit**

```bash
git add experiments/run_experiment_c.py tests/porosity/test_experiment_c.py
git commit -m "feat: add Experiment C (warm-start + proxy refinement LOOCV)"
```

---

### Task 15: Comparison report and full run

**Files:**
- Create: `experiments/compare_results.py`
- Test: `tests/porosity/test_compare.py`

**Interfaces:**
- Consumes: the two results JSON files.
- Produces:
  - `build_comparison(a_results: dict, c_results: dict, baseline_f1: float = 0.40) -> list[dict]` — rows `[{method, mean_f1}]` for: baseline sweep, Optuna ceiling (from A's `mean_ceiling_f1`), Experiment A, Experiment C.
  - `format_table(rows) -> str` — plain-text table.
  - `main()` — loads both results files, prints the table, writes `data/json_output/porosity_comparison.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/porosity/test_compare.py`:

```python
from experiments.compare_results import build_comparison, format_table


def test_build_comparison_has_four_rows():
    a = {"mean_f1": 0.62, "mean_ceiling_f1": 0.81}
    c = {"mean_f1": 0.68}
    rows = build_comparison(a, c, baseline_f1=0.40)
    methods = [r["method"] for r in rows]
    assert len(rows) == 4
    assert any("Baseline" in m for m in methods)
    assert any("Ceiling" in m for m in methods)
    assert any("Experiment A" in m for m in methods)
    assert any("Experiment C" in m for m in methods)


def test_format_table_contains_numbers():
    rows = [{"method": "Experiment C", "mean_f1": 0.68}]
    text = format_table(rows)
    assert "0.68" in text
    assert "Experiment C" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/porosity/test_compare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.compare_results'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/compare_results.py`:

```python
"""Compare baseline, Optuna ceiling, Experiment A, and Experiment C."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "json_output"
A_FILE = OUT_DIR / "experiment_a_results.json"
C_FILE = OUT_DIR / "experiment_c_results.json"
COMPARE_FILE = OUT_DIR / "porosity_comparison.json"


def build_comparison(a_results: dict, c_results: dict, baseline_f1: float = 0.40) -> list[dict]:
    return [
        {"method": "Baseline global sweep", "mean_f1": float(baseline_f1)},
        {"method": "Per-image Optuna Ceiling (GT, non-deployable)",
         "mean_f1": float(a_results["mean_ceiling_f1"])},
        {"method": "Experiment A (predictor, LOOCV)", "mean_f1": float(a_results["mean_f1"])},
        {"method": "Experiment C (warm-start + proxy, LOOCV)", "mean_f1": float(c_results["mean_f1"])},
    ]


def format_table(rows: list[dict]) -> str:
    width = max(len(r["method"]) for r in rows)
    lines = [f"{'Method'.ljust(width)}  Mean F1", f"{'-' * width}  -------"]
    for r in rows:
        lines.append(f"{r['method'].ljust(width)}  {r['mean_f1']:.4f}")
    return "\n".join(lines)


def main() -> None:
    with open(A_FILE) as f:
        a = json.load(f)
    with open(C_FILE) as f:
        c = json.load(f)
    rows = build_comparison(a, c)
    print(format_table(rows))
    with open(COMPARE_FILE, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nComparison → {COMPARE_FILE}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/porosity/test_compare.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Run the full unit suite**

Run: `poetry run pytest tests/porosity -v`
Expected: PASS, all tests across tasks 1–15.

- [ ] **Step 6: End-to-end smoke run (small trial budget, real data + YOLO)**

Run: `poetry run python -m experiments.run_experiment_a --trials 20`
Expected: prints per-image ceiling lines, then `Experiment A mean LOOCV F1 = ...`, writes `data/json_output/porosity_tuning_cache.json` and `experiment_a_results.json`. (Trial budget is low — numbers are a smoke check, not final.)

Run: `poetry run python -m experiments.run_experiment_c --refine 8`
Expected: reuses the cache, prints `Experiment C mean LOOCV F1 = ...`, writes `experiment_c_results.json`.

Run: `poetry run python -m experiments.compare_results`
Expected: prints the 4-row comparison table, writes `porosity_comparison.json`.

- [ ] **Step 7: Full-budget run (final numbers)**

Run: `poetry run python -m experiments.run_experiment_a --trials 300`
Then: `poetry run python -m experiments.run_experiment_c --refine 15`
Then: `poetry run python -m experiments.compare_results`
Expected: final comparison table. Record the four mean-F1 numbers in the PR/notes.

- [ ] **Step 8: Commit**

```bash
git add experiments/compare_results.py tests/porosity/test_compare.py data/json_output/porosity_comparison.json
git commit -m "feat: add comparison report and record final A vs C vs baseline results"
```

---

## Self-Review

**Spec coverage:**
- Problem (per-image adaptation; optimizer must use the filtered pipeline, not raw binary) → Task 4 `detect_pores` is the shared filtered pipeline; Tasks 9/13/14 optimize/evaluate through it. ✓
- Shared pipeline as single source of truth → Task 4/6 build `porosity_pipeline.py`; consumed everywhere. ✓
- Exact filter order (erode → threshold → close → open → AND eroded → contours → circularity 2nd-to-last → darkness last) → Task 4 implementation + comments. ✓
- 6 params + bounds + absolute `darkness_thresh` → Task 2 `PARAM_BOUNDS`, Task 9 search space. ✓
- Excluded filters (min-area, aspect, min-diameter) → not implemented anywhere. ✓
- Features from weld region → Task 8. ✓
- Per-param Random Forest → Task 10. ✓
- Optuna per-image GT optimization + ceiling → Task 9 + Task 12. ✓
- LOOCV for A and C → Task 11 + Tasks 13/14. ✓
- Proxy = circularity × darkness_contrast, GT-free, bounded refinement → Task 5 + Task 9 `refine_params`. ✓
- Experiment A, Experiment C, comparison table incl. non-deployable ceiling row → Tasks 13, 14, 15. ✓
- New deps optuna + scikit-learn → Task 1. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; every test step shows assertions. ✓

**Type consistency:** `PoreParams`, `PoreDetection`, `detect_pores`/`detection_mask`/`pixel_f1`/`proxy_score`, `PARAM_ORDER`/`params_to_vector`/`vector_to_params`, `build_cache`/`cache_to_matrices`/`load_cache`, `loocv_predict`, `evaluate_predictions`/`evaluate_with_refinement` names are used identically across producing and consuming tasks. ✓

---

## Notes / Risks

- **18 samples is small.** Experiment C's test-time refinement exists to mitigate weak generalization; the comparison quantifies whether it helps.
- **Proxy↔F1 correlation.** If C underperforms A, the proxy is poorly correlated with F1 — the bounded neighborhood (`neighborhood=0.25`) keeps refinement from straying far from the learned prior, limiting downside.
- **Optuna cost.** Full run is 18 × 300 trials once (cached); the smoke run (`--trials 20`) validates wiring fast.
- **Determinism.** All Optuna samplers and RandomForests are seeded for reproducible paper numbers.
