"""Render porosity results as images.

For each validation image, runs Experiment A's LOOCV-predicted params through the
shared pipeline and draws a 3-panel comparison:
  [0] Original
  [1] Original + GT pores (green)
  [2] Original + GT (green) + Experiment A detections (red)  — titled with F1

Also writes a single montage of all images sorted by F1 (best first).

Output: data/porosity_results_viz/<stem>_A.jpg  and  _montage_A.jpg
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from weld_pipeline.porosity_data import load_dataset
from weld_pipeline.porosity_pipeline import detect_pores, detection_mask, pixel_f1
from experiments.build_cache import (
    ANNO_FILE, CACHE_FILE, DATA_DIR, MODEL_PT, cache_to_matrices, load_cache,
)
from experiments.loocv import loocv_predict

OUT_DIR = Path(CACHE_FILE).parent.parent / "porosity_results_viz"

GREEN = (40, 200, 70)
RED = (235, 60, 60)
PAD = 26


def _tag(panel: np.ndarray, label: str) -> np.ndarray:
    h, w = panel.shape[:2]
    out = np.zeros((h + PAD, w, 3), np.uint8)
    out[PAD:, :] = panel
    cv2.putText(out, label, (5, PAD - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (235, 235, 235), 1, cv2.LINE_AA)
    return out


def _panels_for_record(rec, params):
    """Return (tagged_panels list, f1) for one image."""
    rgb = cv2.cvtColor(rec.gray, cv2.COLOR_GRAY2RGB)

    gt_cnts, _ = cv2.findContours(rec.gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = detect_pores(rec.gray, rec.weld_mask, params)
    det_mask = detection_mask(dets, rec.gray.shape)
    f1 = pixel_f1(rec.gt_mask, det_mask)
    det_cnts, _ = cv2.findContours(det_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    p0 = rgb.copy()
    p1 = rgb.copy()
    cv2.drawContours(p1, gt_cnts, -1, GREEN, 2)
    p2 = rgb.copy()
    cv2.drawContours(p2, gt_cnts, -1, GREEN, 2)
    cv2.drawContours(p2, det_cnts, -1, RED, 2)

    labels = ["Original",
              f"GT pores: {len(gt_cnts)} (green)",
              f"Detected: {len(dets)} (red)  F1={f1:.3f}"]
    return [_tag(p, l) for p, l in zip((p0, p1, p2), labels)], f1


def main() -> None:
    if not Path(CACHE_FILE).exists():
        print(f"Cache not found: {CACHE_FILE}\nRun experiment A first.")
        return

    cache = load_cache()
    model = YOLO(str(MODEL_PT))
    records = load_dataset(DATA_DIR, ANNO_FILE, model)
    by_name = {r.file_name: r for r in records}
    records = [by_name[it["file_name"]] for it in cache["items"]]

    X, Y = cache_to_matrices(cache)
    preds = loocv_predict(X, Y, seed=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []  # (f1, stacked_row_image) for the montage
    for rec, params in zip(records, preds):
        panels, f1 = _panels_for_record(rec, params)
        # normalize panel heights for the per-image strip
        h = min(p.shape[0] for p in panels)
        panels = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
        strip = np.hstack(panels)
        stem = Path(rec.file_name).stem[:40]
        cv2.imwrite(str(OUT_DIR / f"{stem}_A.jpg"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append((f1, strip))
        print(f"  F1={f1:.3f}  {rec.file_name[:55]}")

    # Montage: all images sorted best-first, padded to a common width.
    rows.sort(key=lambda r: r[0], reverse=True)
    max_w = max(img.shape[1] for _, img in rows)
    padded = []
    for _, img in rows:
        if img.shape[1] < max_w:
            img = cv2.copyMakeBorder(img, 0, 0, 0, max_w - img.shape[1],
                                     cv2.BORDER_CONSTANT, value=(20, 20, 20))
        padded.append(img)
    montage = np.vstack(padded)
    montage_path = OUT_DIR / "_montage_A.jpg"
    cv2.imwrite(str(montage_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])

    mean_f1 = float(np.mean([f for f, _ in rows]))
    print(f"\n{len(rows)} images rendered → {OUT_DIR}")
    print(f"Montage (best-first) → {montage_path}")
    print(f"Mean Experiment A LOOCV F1 = {mean_f1:.4f}")


if __name__ == "__main__":
    main()
