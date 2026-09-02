"""
Interactive porosity tuner with live F1 / precision / recall.

Shows two panels only: YOLO weld detection and final pore detections with
ground-truth overlay. Image selector on the left lets you switch between all
test images. F1 / precision / recall update across the full test set after
every slider release.

Ground truth: data/test/_annotations.coco.json  (segmentation polygons)
Model:        models/wda11s-seg.pt

IoU is computed at the pixel-mask level (detected contour vs GT polygon),
not bounding-box level, for accurate matching on irregular pore shapes.

Usage:
    python src/weld_pipeline/porosity_tune_f1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from weld_pipeline.porosity_light_check import (
    _build_weld_mask,
    _compute_dynamic_params,
)

# ── Config ────────────────────────────────────────────────────────────────────
_REPO      = Path(__file__).parents[2]
TEST_DIR   = _REPO / "data" / "test"
ANNO_FILE  = TEST_DIR / "_annotations.coco.json"
MODEL_PATH = str(_REPO / "models" / "wda11s-seg.pt")
WELD_CONF  = 0.01
MEDIAN_K   = 5
IOU_THRESH = 0.3

INIT = dict(
    threshold   = 50,
    darkness    = 0.85,
    diam_frac   = 0.05,
    circularity = 0.55,
    aspect      = 0.5,
    close_k     = 3,
    open_k      = 3,
    erode_iters = 10,
)

_RING_K  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
_ERODE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ── Ground truth ──────────────────────────────────────────────────────────────
def _load_gt() -> tuple[dict[str, list], dict[str, str]]:
    """
    Parse COCO annotations → segmentation contours (not bounding boxes).

    Returns:
        gt          — {filename: [contour, ...]}  each contour is (N,1,2) int32
        short_names — {filename: short display name}
    """
    with open(ANNO_FILE) as f:
        coco = json.load(f)
    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    gt: dict[str, list] = {img["file_name"]: [] for img in coco["images"]}
    short_names = {
        img["file_name"]: Path(img.get("extra", {}).get("name", img["file_name"])).stem
        for img in coco["images"]
    }
    for ann in coco["annotations"]:
        segs = ann.get("segmentation") or []
        if not segs:
            continue
        # Use the largest polygon ring (handles rare multi-ring annotations)
        poly = max(segs, key=len)
        cnt  = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        gt[id_to_name[ann["image_id"]]].append(cnt.reshape(-1, 1, 2))
    return gt, short_names


# ── Mask-level IoU ────────────────────────────────────────────────────────────
def _mask_iou(det_cnt: np.ndarray, gt_cnt: np.ndarray) -> float:
    """
    Pixel-level IoU between a detected contour and a GT polygon contour.
    Both contours use absolute image coordinates.
    Operates in a local bounding box to avoid allocating full-image arrays.
    """
    dx, dy, dw, dh = cv2.boundingRect(det_cnt)
    gx, gy, gw, gh = cv2.boundingRect(gt_cnt)
    x0 = min(dx, gx);      y0 = min(dy, gy)
    x1 = max(dx+dw, gx+gw); y1 = max(dy+dh, gy+gh)
    w = x1 - x0;            h = y1 - y0
    if w <= 0 or h <= 0:
        return 0.0
    offset    = np.array([[[x0, y0]]])
    det_mask  = np.zeros((h, w), dtype=np.uint8)
    gt_mask   = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(det_mask, [det_cnt - offset], -1, 1, cv2.FILLED)
    cv2.drawContours(gt_mask,  [gt_cnt  - offset], -1, 1, cv2.FILLED)
    inter = int(np.count_nonzero(det_mask & gt_mask))
    union = int(np.count_nonzero(det_mask | gt_mask))
    return inter / union if union > 0 else 0.0


def _match(det_cnts: list, gt_cnts: list) -> tuple[int, int, int]:
    """Greedy mask-IoU match → (TP, FP, FN)."""
    if not gt_cnts:
        return 0, len(det_cnts), 0
    if not det_cnts:
        return 0, 0, len(gt_cnts)
    pairs: list[tuple[float, int, int]] = []
    for di, d in enumerate(det_cnts):
        for gi, g in enumerate(gt_cnts):
            iou = _mask_iou(d, g)
            if iou >= IOU_THRESH:
                pairs.append((iou, di, gi))
    pairs.sort(reverse=True)
    used_d: set[int] = set()
    used_g: set[int] = set()
    for _, di, gi in pairs:
        if di not in used_d and gi not in used_g:
            used_d.add(di); used_g.add(gi)
    tp = len(used_d)
    return tp, len(det_cnts) - tp, len(gt_cnts) - tp


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return f, p, r


# ── Image preloading ──────────────────────────────────────────────────────────
def _preload(gt: dict, short_names: dict, seg_model) -> list[dict]:
    entries = []
    for fname, gt_cnts in gt.items():
        path = TEST_DIR / fname
        if not path.exists():
            print(f"  [skip] {fname} not found")
            continue
        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, MEDIAN_K)

        weld_mask = None
        weld_vis  = None
        if seg_model is not None:
            weld_mask = _build_weld_mask(img_rgb, seg_model, WELD_CONF)
            if weld_mask is not None and weld_mask.any():
                ov   = img_rgb.copy().astype(np.float32)
                teal = np.array([0, 220, 180], dtype=np.float32)
                ov[weld_mask > 0] = ov[weld_mask > 0] * 0.5 + teal * 0.5
                ov = ov.astype(np.uint8)
                cnts, _ = cv2.findContours(
                    weld_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(ov, cnts, -1, (0, 255, 180), 2)
                weld_vis = ov

        otsu_val, _, _, _, _, mask_mean, weld_width = _compute_dynamic_params(
            blurred, gray, weld_mask, 0.85, 0.05)

        entries.append(dict(
            fname=fname,
            short_name=short_names.get(fname, fname),
            img_rgb=img_rgb, gray=gray,
            blurred=blurred, weld_mask=weld_mask, weld_vis=weld_vis,
            otsu_val=otsu_val, mask_mean=mask_mean, weld_width=weld_width,
            gt_contours=gt_cnts,
            _cache=dict(
                binary_key=None, binary_mask=None,
                close_key=None,  closed=None, combined=None,
                open_key=None,   opened=None, contours=None,
                erode=None,      eroded=None,
            ),
        ))
        print(f"  loaded {fname}  (gt pores: {len(gt_cnts)})")
    return entries


# ── Per-image detection ───────────────────────────────────────────────────────
def _detect(entry: dict, p: dict) -> list:
    """Run detection pipeline on one entry. Returns list of pore contours."""
    c    = entry["_cache"]
    gray = entry["gray"]
    blr  = entry["blurred"]
    wm   = entry["weld_mask"]

    applied_thresh = int(np.clip(p["threshold"], 0, 255))
    min_darkness   = entry["mask_mean"] * p["darkness"]
    ww             = entry["weld_width"]
    min_area = (
        float(np.pi * ((ww * p["diam_frac"]) / 2) ** 2)
        if ww is not None else 25.0
    )

    if applied_thresh != c["binary_key"]:
        _, bm = cv2.threshold(blr, applied_thresh, 255, cv2.THRESH_BINARY_INV)
        c["binary_key"] = applied_thresh; c["binary_mask"] = bm
    else:
        bm = c["binary_mask"]

    ck = int(p["close_k"]) | 1
    if (applied_thresh, ck) != c["close_key"]:
        k_c    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        closed = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, k_c, iterations=2)
        comb   = cv2.bitwise_and(closed, wm) if wm is not None else closed
        c["close_key"] = (applied_thresh, ck)
        c["closed"] = closed; c["combined"] = comb
    else:
        comb = c["combined"]

    ok = int(p["open_k"]) | 1
    if (applied_thresh, ck, ok) != c["open_key"]:
        k_o    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        opened = cv2.morphologyEx(comb, cv2.MORPH_OPEN, k_o)
        cnts, _ = cv2.findContours(
            opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c["open_key"] = (applied_thresh, ck, ok)
        c["opened"] = opened; c["contours"] = cnts
    else:
        cnts = c["contours"]

    ei = int(p["erode_iters"])
    if ei != c["erode"]:
        eroded = (cv2.erode(wm, _ERODE_K, iterations=ei)
                  if wm is not None and ei > 0 else None)
        c["erode"] = ei; c["eroded"] = eroded
    else:
        eroded = c["eroded"]

    pore_cnts = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perim = cv2.arcLength(cnt, True)
        circ  = (4 * np.pi * area / perim ** 2) if perim > 0 else 0.0
        if circ < p["circularity"]:
            continue
        _, (rw, rh), _ = cv2.minAreaRect(cnt)
        asp = min(rw, rh) / max(rw, rh) if max(rw, rh) > 0 else 0.0
        if asp < p["aspect"]:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        rp = 15
        x0 = max(0, bx - rp);              y0 = max(0, by - rp)
        x1 = min(gray.shape[1], bx+bw+rp); y1 = min(gray.shape[0], by+bh+rp)
        crop  = gray[y0:y1, x0:x1]
        cloc  = cnt - np.array([[[x0, y0]]])
        roi   = np.zeros(crop.shape, dtype=np.uint8)
        cv2.drawContours(roi, [cloc], -1, 255, cv2.FILLED)
        m_in  = cv2.mean(crop, mask=roi)[0]
        ring  = cv2.subtract(cv2.dilate(roi, _RING_K), roi)
        if cv2.countNonZero(ring) > 0:
            if cv2.mean(crop, mask=ring)[0] - m_in < min_darkness:
                continue
        if eroded is not None:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                if eroded[cy, cx] == 0:
                    continue
        pore_cnts.append(cnt)

    return pore_cnts


def _build_result(entry: dict, pore_cnts: list) -> np.ndarray:
    res = entry["img_rgb"].copy()
    for gt_cnt in entry["gt_contours"]:
        cv2.drawContours(res, [gt_cnt], -1, (0, 220, 50), 2)
    for cnt in pore_cnts:
        cv2.drawContours(res, [cnt], -1, (255, 50, 50), 2)
    return res


# ── Main interactive function ─────────────────────────────────────────────────
def interactive_tune_f1() -> None:
    print("Loading YOLO model …")
    from ultralytics import YOLO
    seg_model = YOLO(MODEL_PATH)

    print("Loading ground truth …")
    gt, short_names = _load_gt()

    print("Preloading test images …")
    entries = _preload(gt, short_names, seg_model)
    if not entries:
        raise RuntimeError(f"No test images found in {TEST_DIR}")

    cur_idx = [0]

    def current() -> dict:
        return entries[cur_idx[0]]

    def get_params() -> dict:
        return dict(
            threshold   = sl_thresh.val,
            darkness    = sl_dark.val,
            diam_frac   = sl_diam.val,
            circularity = sl_circ.val,
            aspect      = sl_aspect.val,
            close_k     = sl_close.val,
            open_k      = sl_open.val,
            erode_iters = sl_erode.val,
        )

    e0    = current()
    p0    = INIT.copy()
    pore0 = _detect(e0, p0)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.47, top=0.94)

    ax_weld = fig.add_subplot(1, 2, 1)
    ax_res  = fig.add_subplot(1, 2, 2)
    ax_weld.axis("off"); ax_res.axis("off")

    im_weld = ax_weld.imshow(
        e0["weld_vis"] if e0["weld_vis"] is not None else e0["img_rgb"])
    ax_weld.set_title("YOLO weld detection", fontsize=10)

    res0   = _build_result(e0, pore0)
    im_res = ax_res.imshow(res0)
    ttl    = ax_res.set_title(
        f"Detections: {len(pore0)}  (green = GT  ·  red = detected)", fontsize=10)

    # Image selector
    radio_ax = plt.axes([0.01, 0.47, 0.12, 0.46])
    labels   = [e["short_name"] for e in entries]
    radio    = widgets.RadioButtons(radio_ax, labels, active=0,
                                    label_props={"fontsize": [8] * len(labels)})
    radio_ax.set_title("Image", fontsize=8, pad=3)

    # Stats labels
    def _f1_str(tp: int, fp: int, fn: int) -> str:
        f, p, r = _f1(tp, fp, fn)
        return (f"F1 = {f:.3f}   Precision = {p:.3f}   Recall = {r:.3f}"
                f"   (TP={tp}  FP={fp}  FN={fn}  IoU ≥ {IOU_THRESH})")

    def _dyn_str(e: dict) -> str:
        ww = f"{e['weld_width']:.1f}px" if e["weld_width"] is not None else "N/A"
        return (f"{e['short_name']}  ·  Otsu suggestion: {e['otsu_val']}"
                f"  ·  Weld width: {ww}  ·  Mask mean: {e['mask_mean']:.1f}")

    def _compute_global_stats(params: dict) -> tuple[int, int, int]:
        tp = fp = fn = 0
        for e in entries:
            cnts = _detect(e, params)
            t, f_, n = _match(cnts, e["gt_contours"])
            tp += t; fp += f_; fn += n
        return tp, fp, fn

    tp0, fp0, fn0 = _compute_global_stats(p0)
    lbl_f1  = fig.text(0.5, 0.44, _f1_str(tp0, fp0, fn0),
                       ha="center", fontsize=10, fontweight="bold", color="#111")
    lbl_dyn = fig.text(0.5, 0.955, _dyn_str(e0),
                       ha="center", fontsize=8, color="#555")

    # ── Sliders ───────────────────────────────────────────────────────────────
    sl_y    = [0.04 + i * 0.048 for i in range(8)]
    sl_axes = [plt.axes([0.15, y, 0.72, 0.03]) for y in sl_y]
    sl_thresh = widgets.Slider(sl_axes[7], "Threshold",         0,   255, valinit=INIT["threshold"], valstep=1)
    sl_dark  = widgets.Slider(sl_axes[6], "Darkness fraction", 0.0, 2.0, valinit=INIT["darkness"])
    sl_diam  = widgets.Slider(sl_axes[5], "Min diam fraction", 0.0, 0.5, valinit=INIT["diam_frac"])
    sl_circ  = widgets.Slider(sl_axes[4], "Min circularity",   0.1, 1.0, valinit=INIT["circularity"])
    sl_aspect= widgets.Slider(sl_axes[3], "Min aspect ratio",  0.0, 1.0, valinit=INIT["aspect"])
    sl_close = widgets.Slider(sl_axes[2], "Close kernel",      1,   51,  valinit=INIT["close_k"],     valstep=2)
    sl_open  = widgets.Slider(sl_axes[1], "Open kernel",       1,   51,  valinit=INIT["open_k"],      valstep=2)
    sl_erode = widgets.Slider(sl_axes[0], "Erode mask iters",  0,   30,  valinit=INIT["erode_iters"], valstep=1)

    # ── Update callbacks ──────────────────────────────────────────────────────
    def _refresh(params: dict) -> None:
        e         = current()
        pore_cnts = _detect(e, params)
        tp, fp, fn = _compute_global_stats(params)

        weld_img = e["weld_vis"] if e["weld_vis"] is not None else e["img_rgb"]
        im_weld.set_data(weld_img)
        ax_weld.set_xlim(0, weld_img.shape[1])
        ax_weld.set_ylim(weld_img.shape[0], 0)

        res = _build_result(e, pore_cnts)
        im_res.set_data(res)
        ax_res.set_xlim(0, res.shape[1])
        ax_res.set_ylim(res.shape[0], 0)

        ttl.set_text(
            f"Detections: {len(pore_cnts)}  (green = GT  ·  red = detected)")
        lbl_f1.set_text(_f1_str(tp, fp, fn))
        lbl_dyn.set_text(_dyn_str(e))
        fig.canvas.draw_idle()

    def _on_release(_event) -> None:
        _refresh(get_params())

    def _on_radio(label: str) -> None:
        cur_idx[0] = labels.index(label)
        _refresh(get_params())

    fig.canvas.mpl_connect("button_release_event", _on_release)
    radio.on_clicked(_on_radio)

    plt.show()


if __name__ == "__main__":
    interactive_tune_f1()
