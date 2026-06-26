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


def proxy_score(detections: list[PoreDetection]) -> float:
    if not detections:
        return 0.0
    scores = [
        d.circularity * max(0.0, d.darkness_contrast) / 255.0
        for d in detections
    ]
    return float(np.mean(scores))


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
