"""
Pore detection using classical image processing, with optional YOLO weld-region filtering.

Dynamic parameters (computed per image from weld mask statistics):
  - Otsu threshold     — computed on blurred pixels inside the weld mask only
  - Canny thresholds   — canny_low = 0.66 * median, canny_high = 1.33 * median (weld pixels)
  - Ring darkness min  — mask_mean * darkness_fraction
  - Min pore area      — derived from weld width via skeletonization

Static parameters (config constants, not computed per image):
  - median_ksize, morphological kernel sizes and iteration counts
  - erode_iters, min_circularity, min_aspect_ratio
  - min_diameter_fraction, darkness_fraction

Pipeline:
  0. (optional) YOLO weld detection   (restrict search to weld region, class 3)
  1. Grayscale → median blur          (noise suppression)
  2. Otsu threshold on weld pixels    (dark-blob binary mask)
  3. Morphological close + weld mask  (close pore boundaries, restrict to weld region)
  4. Morphological open               (clean up noise)
  5. Canny on final binary mask       (pore boundary detection; thresholds from weld median)
  6. Dilate Canny edges + findContours
  7. Filter contours                  (area, circularity, aspect ratio,
                                       darkness = ring brightness in RGB vs pore interior,
                                       eroded centroid)
  8. Visualise all stages
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


def _yolo_raw_mask(img_rgb: np.ndarray, seg_model, weld_conf: float) -> np.ndarray:
    """Run YOLO class-3 inference and return a mask for the highest-confidence detection."""
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    results = seg_model.predict(img_rgb, conf=weld_conf, classes=[3], verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return mask
    best = int(r.boxes.conf.argmax())
    if r.masks is not None:
        m = cv2.resize(r.masks.data[best].cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (m > 0.5).astype(np.uint8) * 255
    else:
        x1, y1, x2, y2 = map(int, r.boxes.xyxy[best].cpu().numpy())
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, cv2.FILLED)
    return mask


def _postprocess_weld_mask(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes so pore voids inside the weld boundary are included."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def _build_weld_mask(img_rgb: np.ndarray, seg_model, weld_conf: float) -> np.ndarray:
    return _postprocess_weld_mask(_yolo_raw_mask(img_rgb, seg_model, weld_conf))


def estimate_weld_width(weld_mask: np.ndarray) -> float | None:
    """
    Estimate weld bead width in pixels via distance transform + skeletonization.
    Returns median(skeleton distance values) * 2, or None if the mask is empty
    or the skeleton has fewer than 10 points.
    """
    if not weld_mask.any():
        return None
    dist     = cv2.distanceTransform(weld_mask, cv2.DIST_L2, 5)
    skeleton = skeletonize(weld_mask > 0)
    skel_pts = np.argwhere(skeleton)
    if len(skel_pts) < 10:
        return None
    return float(np.median(dist[skel_pts[:, 0], skel_pts[:, 1]])) * 2


def _compute_dynamic_params(
    blurred: np.ndarray,
    gray: np.ndarray,
    weld_mask: np.ndarray | None,
    darkness_fraction: float,
    min_diameter_fraction: float,
) -> tuple[int, int, int, float, float, float, float | None]:
    """
    Compute per-image adaptive parameters from weld mask region statistics.

    Returns:
        otsu_val, canny_low, canny_high, min_darkness, min_area_px,
        mask_mean, weld_width

    The last two (mask_mean, weld_width) are the raw image statistics that
    callers can hold onto to cheaply re-derive min_darkness / min_area_px
    when darkness_fraction or min_diameter_fraction change interactively.
    Falls back to full-image statistics when no weld mask is available.
    """
    if weld_mask is not None and weld_mask.any():
        weld_pixels = blurred[weld_mask > 0].reshape(-1, 1)
        otsu_val, _ = cv2.threshold(weld_pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        median_val  = float(np.median(weld_pixels))
        mask_mean   = float(np.mean(gray[weld_mask > 0]))
        weld_width  = estimate_weld_width(weld_mask)
    else:
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        median_val  = float(np.median(blurred))
        mask_mean   = float(np.mean(gray))
        weld_width  = None

    canny_low    = int(np.clip(0.66 * median_val, 0, 255))
    canny_high   = int(np.clip(1.33 * median_val, 0, 255))
    min_darkness = mask_mean * darkness_fraction

    if weld_width is not None:
        min_pore_d  = weld_width * min_diameter_fraction
        min_area_px = float(np.pi * (min_pore_d / 2) ** 2)
    else:
        min_area_px = 25.0

    return int(otsu_val), canny_low, canny_high, min_darkness, min_area_px, mask_mean, weld_width


def porosity_light_check(
    image_path: str,
    # ── static config ───────────────────────────────────────────────────────────
    median_ksize: int            = 5,     # median blur kernel (must be odd)
    min_circularity: float       = 0.55,  # 4π·area/perimeter²  (1.0 = perfect circle)
    min_aspect_ratio: float      = 0.5,   # min(w,h)/max(w,h) of fitted rect
    seg_model                    = '/Users/oljk/Projects/weld-pipeline/data/img/pores2.jpg',
    weld_conf: float             = 0.01,  # YOLO confidence threshold for weld detection
    close_ksize: int             = 3,     # closing kernel size for pore boundary bridging
    open_ksize: int              = 3,     # opening kernel size after binary + weld masking
    erode_iters: int             = 10,    # inward weld mask erosion for centroid exclusion
    min_diameter_fraction: float = 0.05,  # min pore diameter as fraction of weld width
    darkness_fraction: float     = 0.85,  # min_darkness = mask_mean * this value
    # ── display ─────────────────────────────────────────────────────────────────
    visualize: bool              = True,
) -> list[dict]:
    """
    Detect pore-like shapes using classical image processing.
    Dynamic parameters (Otsu threshold, Canny thresholds, darkness threshold,
    min pore area) are computed per-image from statistics of the YOLO weld region.
    Returns a list of dicts: {contour, area, circularity, bbox}.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── 0. YOLO weld region mask (optional) ────────────────────────────────────
    weld_mask = None
    if seg_model is not None:
        if isinstance(seg_model, (str, Path)):
            from ultralytics import YOLO
            seg_model = YOLO(seg_model)
        weld_mask = _build_weld_mask(img_rgb, seg_model, weld_conf)

    # ── 1. Median blur ──────────────────────────────────────────────────────────
    blurred = cv2.medianBlur(gray, median_ksize)

    # ── Dynamic parameter computation ──────────────────────────────────────────
    otsu_val, _, _, min_darkness, min_area_px, _, _ = _compute_dynamic_params(
        blurred, gray, weld_mask, darkness_fraction, min_diameter_fraction)

    # ── 2. Otsu threshold on weld pixels (dark pores → white blobs) ────────────
    _, binary_mask = cv2.threshold(blurred, otsu_val, 255, cv2.THRESH_BINARY_INV)

    # ── 3. Morphological close + weld mask ─────────────────────────────────────
    close_k  = int(close_ksize) | 1
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    closed   = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.bitwise_and(closed, weld_mask) if weld_mask is not None else closed

    # ── 4. Morphological opening ────────────────────────────────────────────────
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    opened      = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_kernel)

    # ── 5. Contour detection & shape filtering ──────────────────────────────────
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    eroded_mask = None
    if weld_mask is not None and erode_iters > 0:
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded_mask = cv2.erode(weld_mask, ek, iterations=erode_iters)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pores = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue

        perimeter   = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
        if circularity < min_circularity:
            continue

        _, (rw, rh), _ = cv2.minAreaRect(cnt)
        aspect = min(rw, rh) / max(rw, rh) if max(rw, rh) > 0 else 0.0
        if aspect < min_aspect_ratio:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        rp = 15
        x0 = max(0, bx - rp);                  y0 = max(0, by - rp)
        x1 = min(gray.shape[1], bx + bw + rp); y1 = min(gray.shape[0], by + bh + rp)
        crop_g  = gray[y0:y1, x0:x1]
        cnt_loc = cnt - np.array([[[x0, y0]]])
        roi_loc = np.zeros(crop_g.shape, dtype=np.uint8)
        cv2.drawContours(roi_loc, [cnt_loc], -1, 255, cv2.FILLED)
        mean_inside = cv2.mean(crop_g, mask=roi_loc)[0]
        ring_loc = cv2.subtract(cv2.dilate(roi_loc, ring_kernel), roi_loc)
        if cv2.countNonZero(ring_loc) > 0:
            if cv2.mean(crop_g, mask=ring_loc)[0] - mean_inside < min_darkness:
                continue

        if eroded_mask is not None:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                if eroded_mask[cy, cx] == 0:
                    continue

        x, y, w, h = cv2.boundingRect(cnt)
        pores.append({
            "contour":     cnt,
            "area":        round(float(area), 2),
            "circularity": round(float(circularity), 3),
            "bbox":        [x, y, x + w, y + h],
        })

    # ── 8. Visualisation ────────────────────────────────────────────────────────
    if visualize:
        result = img_rgb.copy()
        for p in pores:
            cv2.drawContours(result, [p["contour"]], -1, (255, 50, 50), 2)
            x1, y1, x2, y2 = p["bbox"]
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 200, 0), 1)

        panels = [
            ("1: Original",        img_rgb,     None),
            ("2: Median blur",     blurred,     "gray"),
            ("3: Otsu threshold",  binary_mask, "gray"),
            ("4: After closing",   closed,      "gray"),
            ("5: Combined mask",   combined,    "gray"),
            ("6: After opening",   opened,      "gray"),
        ]
        if weld_mask is not None:
            weld_overlay = img_rgb.copy().astype(np.float32)
            teal = np.array([0, 220, 180], dtype=np.float32)
            weld_overlay[weld_mask > 0] = weld_overlay[weld_mask > 0] * 0.5 + teal * 0.5
            weld_overlay = weld_overlay.astype(np.uint8)
            weld_cnts, _ = cv2.findContours(weld_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(weld_overlay, weld_cnts, -1, (0, 255, 180), 2)
            panels.append(("7: YOLO weld", weld_overlay, None))
        panels.append((f"{'8' if weld_mask is not None else '7'}: Pores ({len(pores)})", result, None))

        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
        for ax, (title, im, cmap) in zip(axes, panels):
            ax.imshow(im, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        param_txt = (
            f"Otsu={otsu_val}  darkness≥{min_darkness:.1f}  min_area={min_area_px:.1f}px²"
        )
        fig.text(0.5, 0.01, param_txt, ha="center", fontsize=8, color="gray")
        plt.tight_layout()
        plt.show()
        plt.close("all")

    return pores


def interactive_tune(
    image_path: str,
    median_ksize: int = 5,
    seg_model         = '/Users/oljk/Projects/weld-pipeline/models/wda11s-seg.pt',
    weld_conf: float  = 0.01,
) -> None:
    """
    Interactive matplotlib window with sliders for static config parameters.
    Dynamic parameters (Otsu threshold, Canny thresholds, weld width) are
    computed once from the loaded image and shown as read-only info labels.
    Only the static config values that feed contour filtering are slider-driven.
    """
    import matplotlib.widgets as widgets

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, median_ksize)

    # ── Run YOLO once; build static weld overlay ────────────────────────────────
    cur_weld_mask   = None
    static_weld_vis = None
    has_weld_model  = seg_model is not None
    if has_weld_model:
        if isinstance(seg_model, (str, Path)):
            from ultralytics import YOLO
            seg_model = YOLO(seg_model)
        cur_weld_mask = _build_weld_mask(img_rgb, seg_model, weld_conf)
        if cur_weld_mask.any():
            static_weld_vis = img_rgb.copy().astype(np.float32)
            teal = np.array([0, 220, 180], dtype=np.float32)
            static_weld_vis[cur_weld_mask > 0] = static_weld_vis[cur_weld_mask > 0] * 0.5 + teal * 0.5
            static_weld_vis = static_weld_vis.astype(np.uint8)
            weld_cnts, _ = cv2.findContours(cur_weld_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(static_weld_vis, weld_cnts, -1, (0, 255, 180), 2)

    # ── Compute fixed dynamic params (Otsu, Canny, weld stats) once ────────────
    otsu_val, _, _, _, _, mask_mean, weld_width = _compute_dynamic_params(
        blurred, gray, cur_weld_mask, darkness_fraction=0.85, min_diameter_fraction=0.05)

    ring_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # ── Per-step cache: skip expensive ops when inputs haven't changed ──────────
    _cache = dict(
        binary_key=None, binary_mask=None,
        close_key=None, closed=None, combined=None,
        open_key=None, opened=None, contours=None,
        erode=None, eroded=None,
    )

    def _recompute(otsu_mult, darkness_frac, min_diam_frac, min_circ, min_aspect,
                   close_ksize, open_ksize, erode_iters):
        applied_thresh = int(np.clip(otsu_val * otsu_mult, 0, 255))

        # ── binary mask (recompute when otsu changes) ──
        if applied_thresh != _cache['binary_key']:
            _, bm = cv2.threshold(blurred, applied_thresh, 255, cv2.THRESH_BINARY_INV)
            _cache['binary_key']  = applied_thresh
            _cache['binary_mask'] = bm
        else:
            bm = _cache['binary_mask']

        # ── close + weld mask (recompute when otsu or close_ksize changes) ──
        ck = int(close_ksize) | 1
        if (applied_thresh, ck) != _cache['close_key']:
            ck_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
            closed    = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, ck_kernel, iterations=2)
            comb      = cv2.bitwise_and(closed, cur_weld_mask) if cur_weld_mask is not None else closed
            _cache['close_key'] = (applied_thresh, ck)
            _cache['closed']    = closed
            _cache['combined']  = comb
        else:
            closed = _cache['closed']
            comb   = _cache['combined']

        # Derive filter thresholds from image stats + current slider fractions.
        min_darkness = mask_mean * darkness_frac
        if weld_width is not None:
            min_area = float(np.pi * ((weld_width * min_diam_frac) / 2) ** 2)
        else:
            min_area = 25.0

        # ── open → findContours (recompute when thresh, close or open_ksize changes) ──
        ok = int(open_ksize) | 1
        if (applied_thresh, ck, ok) != _cache['open_key']:
            ok_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
            opened    = cv2.morphologyEx(comb, cv2.MORPH_OPEN, ok_kernel)
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            _cache['open_key'] = (applied_thresh, ck, ok)
            _cache['opened']   = opened
            _cache['contours'] = contours
        else:
            opened   = _cache['opened']
            contours = _cache['contours']

        # ── eroded weld mask (recompute only when erode_iters changes) ──
        if int(erode_iters) != _cache['erode']:
            eroded_mask = None
            if cur_weld_mask is not None and int(erode_iters) > 0:
                eroded_mask = cv2.erode(cur_weld_mask, erode_kernel, iterations=int(erode_iters))
            _cache['erode']  = int(erode_iters)
            _cache['eroded'] = eroded_mask
        else:
            eroded_mask = _cache['eroded']

        eroded_opened = cv2.bitwise_and(opened, eroded_mask) if eroded_mask is not None else opened

        # ── contour filtering ──
        pore_cnts = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            perim = cv2.arcLength(cnt, True)
            circ  = (4 * np.pi * area / perim ** 2) if perim > 0 else 0.0
            if circ < min_circ:
                continue
            _, (rw, rh), _ = cv2.minAreaRect(cnt)
            aspect = min(rw, rh) / max(rw, rh) if max(rw, rh) > 0 else 0.0
            if aspect < min_aspect:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            rp  = 15
            x0  = max(0, bx - rp);              y0 = max(0, by - rp)
            x1  = min(gray.shape[1], bx+bw+rp); y1 = min(gray.shape[0], by+bh+rp)
            crop_g  = gray[y0:y1, x0:x1]
            cnt_loc = cnt - np.array([[[x0, y0]]])
            roi_loc = np.zeros(crop_g.shape, dtype=np.uint8)
            cv2.drawContours(roi_loc, [cnt_loc], -1, 255, cv2.FILLED)
            mean_inside = cv2.mean(crop_g, mask=roi_loc)[0]
            ring_loc = cv2.subtract(cv2.dilate(roi_loc, ring_kernel), roi_loc)
            if cv2.countNonZero(ring_loc) > 0:
                if cv2.mean(crop_g, mask=ring_loc)[0] - mean_inside < min_darkness:
                    continue
            if eroded_mask is not None:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    if eroded_mask[cy, cx] == 0:
                        continue
            pore_cnts.append(cnt)

        result = img_rgb.copy()
        for cnt in pore_cnts:
            cv2.drawContours(result, [cnt], -1, (255, 50, 50), 2)

        return bm, closed, comb, opened, eroded_opened, result, len(pore_cnts)

    INIT_OTSU_MULT = 0.372
    INIT_DARKNESS  = 0.85
    INIT_DIAM_FRAC = 0.05
    INIT_CIRC      = 0.55
    INIT_ASPECT    = 0.5
    INIT_CLOSE     = 3
    INIT_OPEN      = 3
    INIT_ERODE     = 10

    bm0, closed0, combined0, opened0, eroded0, result0, n0 = _recompute(
        INIT_OTSU_MULT, INIT_DARKNESS, INIT_DIAM_FRAC, INIT_CIRC, INIT_ASPECT,
        INIT_CLOSE, INIT_OPEN, INIT_ERODE)

    # ── 3×3 layout ──────────────────────────────────────────────────────────────
    # Row 0: Original | Median blur | Otsu threshold
    # Row 1: After closing | Combined mask | After opening
    # Row 2: Opened∩eroded | YOLO weld | Pores on original
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.subplots_adjust(bottom=0.44, hspace=0.3)

    # Row 0 — fixed
    axes[0, 0].imshow(img_rgb);              axes[0, 0].set_title("Original");       axes[0, 0].axis("off")
    axes[0, 1].imshow(blurred, cmap="gray"); axes[0, 1].set_title("Median blur");    axes[0, 1].axis("off")

    # Row 0 col 2 — slider-driven
    im_global = axes[0, 2].imshow(bm0, cmap="gray")
    axes[0, 2].set_title("Otsu threshold"); axes[0, 2].axis("off")

    # Row 1 — slider-driven
    im_close = axes[1, 0].imshow(closed0,   cmap="gray")
    im_comb  = axes[1, 1].imshow(combined0, cmap="gray")
    im_open  = axes[1, 2].imshow(opened0,   cmap="gray")
    axes[1, 0].set_title("After closing");  axes[1, 0].axis("off")
    axes[1, 1].set_title("Combined mask");  axes[1, 1].axis("off")
    axes[1, 2].set_title("After opening");  axes[1, 2].axis("off")

    # Row 2 — slider-driven
    im_erode = axes[2, 0].imshow(eroded0, cmap="gray")
    im_res   = axes[2, 2].imshow(result0)
    axes[2, 0].set_title("Opened ∩ eroded mask"); axes[2, 0].axis("off")
    ttl = axes[2, 2].set_title(f"Detected pores: {n0}"); axes[2, 2].axis("off")

    # Row 2 col 1 — YOLO weld reference (fixed) or blank
    if has_weld_model:
        axes[2, 1].imshow(static_weld_vis if static_weld_vis is not None else img_rgb)
        axes[2, 1].set_title("YOLO weld (reference)")
    axes[2, 1].axis("off")

    weld_w_str = f"{weld_width:.1f}px" if weld_width is not None else "N/A"
    dyn_label  = (
        f"Fixed per image — Otsu: {otsu_val}   "
        f"Weld width: {weld_w_str}   "
        f"Mask mean: {mask_mean:.1f}"
    )
    fig.text(0.5, 0.41, dyn_label, ha="center", fontsize=8, color="#555555")

    sl_y      = [0.04 + i * 0.05 for i in range(8)]  # 0.04 … 0.39
    sl_axes   = [plt.axes([0.12, y, 0.62, 0.03]) for y in sl_y]
    sl_otsu   = widgets.Slider(sl_axes[7], "Otsu multiplier",    0.1,  3.0, valinit=INIT_OTSU_MULT)
    sl_dark   = widgets.Slider(sl_axes[6], "Darkness fraction",  0.0,  2.0, valinit=INIT_DARKNESS)
    sl_diam   = widgets.Slider(sl_axes[5], "Min diam fraction",  0.0,  0.5, valinit=INIT_DIAM_FRAC)
    sl_circ   = widgets.Slider(sl_axes[4], "Min circularity",    0.1,  1.0, valinit=INIT_CIRC)
    sl_aspect = widgets.Slider(sl_axes[3], "Min aspect ratio",   0.0,  1.0, valinit=INIT_ASPECT)
    sl_close  = widgets.Slider(sl_axes[2], "Close kernel",       1,    51,  valinit=INIT_CLOSE, valstep=2)
    sl_open   = widgets.Slider(sl_axes[1], "Open kernel",        1,    51,  valinit=INIT_OPEN,  valstep=2)
    sl_erode  = widgets.Slider(sl_axes[0], "Erode mask iters",   0,    30,  valinit=INIT_ERODE, valstep=1)

    def _update(_):
        bm, clsd, comb, opnd, er_opnd, res, n = _recompute(
            sl_otsu.val, sl_dark.val, sl_diam.val, sl_circ.val,
            sl_aspect.val, sl_close.val, sl_open.val, sl_erode.val)
        im_global.set_data(bm)
        im_close.set_data(clsd)
        im_comb.set_data(comb)
        im_open.set_data(opnd)
        im_erode.set_data(er_opnd)
        im_res.set_data(res)
        ttl.set_text(f"Detected pores: {n}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_release_event", _update)
    plt.show()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/oljk/Projects/weld-pipeline/data/porosity_val/IMG_1679 3.png"
    interactive_tune(path)
