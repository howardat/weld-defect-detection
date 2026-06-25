"""
Pore detection using classical image processing, with optional YOLO weld-region filtering.

Pipeline:
  0. (optional) YOLO weld detection   (restrict search to weld region, class 3)
  1. Grayscale → median blur          (noise suppression)
  2. Adaptive threshold               (dark-blob binary mask)
  3. Canny on blurred image           (edge map)
  4. Combine mask + dilated edges     (close pore boundaries)
  5. Mask to weld region              (if seg_model provided)
  6. Filter contours by area and circularity  (pore-shape classifier)
  7. Visualise all stages
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def _yolo_raw_mask(img_rgb: np.ndarray, seg_model, weld_conf: float) -> np.ndarray:
    """Run YOLO class-3 inference and return a raw uint8 pixel mask."""
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    results = seg_model.predict(img_rgb, conf=weld_conf, classes=[3], verbose=False)
    r = results[0]
    if r.masks is not None:
        for mask_tensor in r.masks.data:
            m = cv2.resize(mask_tensor.cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
            mask = cv2.bitwise_or(mask, (m > 0.5).astype(np.uint8) * 255)
    elif r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
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


def porosity_light_check(
    image_path: str,
    median_ksize: int        = 5,     # median blur kernel (must be odd)
    adaptive_block_size: int = 51,    # adaptive threshold neighbourhood (must be odd > 1)
    adaptive_c: int          = 330,   # constant subtracted from local mean/gaussian
    canny_low: int           = 30,    # Canny lower hysteresis threshold
    canny_high: int          = 100,   # Canny upper hysteresis threshold
    min_area_px: float       = 25.0,  # minimum contour area in pixels
    min_circularity: float   = 0.25,  # 4π·area/perimeter²  (1.0 = perfect circle)
    seg_model                = 'C:/Users/01/Projects/weld-defect-detection/models/wda11s-seg.pt',  # loaded ultralytics YOLO model; if set, restricts to weld region
    weld_conf: float         = 0.01,  # YOLO confidence threshold for weld detection (class 3)
    open_ksize: int          = 3,    # opening kernel applied after binary+weld masking
    erode_iters: int         = 10,    # inward erosion of weld mask; filters bead-edge shadows (0 = off)
    min_aspect_ratio: float  = 0.5,   # min(w,h)/max(w,h) of fitted rect; rejects elongated shapes
    visualize: bool          = True,
) -> list[dict]:
    """
    Detect pore-like shapes using classical image processing.
    When *seg_model* is provided, detection is restricted to the YOLO-detected
    weld region (class 3) so false positives outside the weld are suppressed.
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

    # ── 2. Adaptive threshold (dark pores → white blobs) ───────────────────────
    # THRESH_BINARY_INV: pixels darker than local Gaussian mean become 255
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        adaptive_c,
    )

    # ── 3. Canny edge detection ─────────────────────────────────────────────────
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # ── 4. Combine: dilate edges then OR with adaptive mask ─────────────────────
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_d  = cv2.dilate(edges, kernel, iterations=1)
    combined = cv2.bitwise_or(adaptive, edges_d)
    # Morphological close to fill small gaps inside pore boundaries
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── 5. Restrict to weld region (if YOLO model provided) ────────────────────
    if weld_mask is not None:
        combined = cv2.bitwise_and(combined, weld_mask)

    # ── 5b. Morphological opening to clean up after binary + weld masking ───────
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    opened      = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_kernel)

    # ── 6. Contour detection & shape filtering ──────────────────────────────────
    # Darkness threshold: 40% of the std dev of the analysis region so it adapts per image.
    _region = gray[weld_mask > 0] if (weld_mask is not None and weld_mask.any()) else gray.ravel()
    min_darkness = float(np.std(_region)) * 0.4

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

        # Relative darkness on a bounding-box crop — avoids full-image mask ops
        bx, by, bw, bh = cv2.boundingRect(cnt)
        rp = 15  # ring kernel radius
        x0 = max(0, bx - rp);           y0 = max(0, by - rp)
        x1 = min(gray.shape[1], bx + bw + rp); y1 = min(gray.shape[0], by + bh + rp)
        crop_g   = gray[y0:y1, x0:x1]
        cnt_loc  = cnt - np.array([[[x0, y0]]])
        roi_loc  = np.zeros(crop_g.shape, dtype=np.uint8)
        cv2.drawContours(roi_loc, [cnt_loc], -1, 255, cv2.FILLED)
        mean_inside = cv2.mean(crop_g, mask=roi_loc)[0]
        ring_loc = cv2.subtract(cv2.dilate(roi_loc, ring_kernel), roi_loc)
        if cv2.countNonZero(ring_loc) > 0:
            if cv2.mean(crop_g, mask=ring_loc)[0] - mean_inside < min_darkness:
                continue

        # Eroded-mask boundary: reject centroids too close to the weld edge
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

    # ── 7. Visualisation ────────────────────────────────────────────────────────
    if visualize:
        result = img_rgb.copy()
        for p in pores:
            cv2.drawContours(result, [p["contour"]], -1, (255, 50, 50), 2)
            x1, y1, x2, y2 = p["bbox"]
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 200, 0), 1)

        panels = [
            ("1: Original",           img_rgb,  None),
            ("2: Median blur",        blurred,  "gray"),
            ("3: Adaptive threshold", adaptive, "gray"),
            ("4: Canny edges",        edges,    "gray"),
            ("5: Combined mask",      combined, "gray"),
            ("6: After opening",      opened,   "gray"),
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

        plt.tight_layout()
        plt.show()
        plt.close("all")

    return pores


def interactive_tune(
    image_path: str,
    median_ksize: int      = 5,
    canny_low: int         = 30,
    canny_high: int        = 100,
    seg_model              = 'C:/Users/01/Projects/weld-defect-detection/models/wda11s-seg.pt',
    weld_conf: float       = 0.01,
) -> None:
    """
    Interactive matplotlib window with sliders for binarisation and shape parameters.
    If seg_model is provided, a static YOLO weld-region overlay is shown as a reference.
    """
    import matplotlib.widgets as widgets

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, median_ksize)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Run YOLO once and build a static weld overlay (never updated by sliders).
    cur_weld_mask  = None
    static_weld_vis = None
    has_weld_model = seg_model is not None
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

    # Auto-compute darkness threshold once from the analysis region std dev.
    _dk_region   = gray[cur_weld_mask > 0] if (cur_weld_mask is not None and cur_weld_mask.any()) else gray.ravel()
    auto_darkness = float(np.std(_dk_region)) * 0.4

    ring_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    use_adaptive = [False]   # mutable so the checkbox callback can flip it

    # Per-step cache: skip expensive ops when their inputs haven't changed.
    _cache = dict(
        thresh=None,  bg=None,
        block=None,   ac=None,   ba=None,
        pipe_key=None, combined=None, opened=None, contours=None,
        erode=None,   eroded=None,
    )

    def _recompute(thresh, min_area, min_circ, open_ksize, erode_iters,
                   block_size, adapt_c, min_aspect):
        # ── Step 1: global threshold (recompute only when thresh changes) ──
        if int(thresh) != _cache['thresh']:
            _, bg = cv2.threshold(blurred, int(thresh), 255, cv2.THRESH_BINARY_INV)
            _cache['thresh'] = int(thresh)
            _cache['bg'] = bg
        else:
            bg = _cache['bg']

        # ── Step 2: adaptive threshold (recompute only when block/C change) ──
        bs = int(block_size) | 1
        ac = int(adapt_c)
        if bs != _cache['block'] or ac != _cache['ac']:
            ba = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, bs, ac,
            )
            _cache['block'] = bs
            _cache['ac']    = ac
            _cache['ba']    = ba
        else:
            ba = _cache['ba']

        # ── Step 3: downstream pipeline (keyed on active binary identity + open_ksize) ──
        # id() is stable while the array object lives in _cache; changes when rebuilt above.
        binary   = ba if use_adaptive[0] else bg
        pipe_key = (id(binary), int(open_ksize))
        if pipe_key != _cache['pipe_key']:
            edges    = cv2.Canny(cv2.bitwise_not(binary), canny_low, canny_high)
            edges_d  = cv2.dilate(edges, kernel, iterations=1)
            combined = cv2.bitwise_or(binary, edges_d)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
            if cur_weld_mask is not None:
                combined = cv2.bitwise_and(combined, cur_weld_mask)
            ok     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_ksize), int(open_ksize)))
            opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, ok)
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            _cache['pipe_key'] = pipe_key
            _cache['combined'] = combined
            _cache['opened']   = opened
            _cache['contours'] = contours
        else:
            combined = _cache['combined']
            opened   = _cache['opened']
            contours = _cache['contours']

        # ── Step 4: eroded weld mask (recompute only when erode_iters changes) ──
        if int(erode_iters) != _cache['erode']:
            eroded_mask = None
            if cur_weld_mask is not None and int(erode_iters) > 0:
                eroded_mask = cv2.erode(cur_weld_mask, erode_kernel, iterations=int(erode_iters))
            _cache['erode']  = int(erode_iters)
            _cache['eroded'] = eroded_mask
        else:
            eroded_mask = _cache['eroded']

        # ── Step 5: contour filtering (always runs; fast with per-contour crop) ──
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

            # Darkness filter on a tight bounding-box crop (avoids full-image mask allocs)
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
                if cv2.mean(crop_g, mask=ring_loc)[0] - mean_inside < auto_darkness:
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

        return bg, ba, combined, opened, result, len(pore_cnts)

    INIT_THRESH    = 97
    INIT_AREA      = 25.0
    INIT_CIRC      = 0.28
    INIT_OPEN      = 15
    INIT_ERODE     = 10
    INIT_BLOCK     = 51
    INIT_C         = 10
    INIT_ASPECT    = 0.5

    bg0, ba0, combined0, opened0, result0, n0 = _recompute(
        INIT_THRESH, INIT_AREA, INIT_CIRC, INIT_OPEN, INIT_ERODE,
        INIT_BLOCK, INIT_C, INIT_ASPECT)

    n_panels = 6 if has_weld_model else 5
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 7))
    fig.subplots_adjust(bottom=0.52)

    im_global   = axes[0].imshow(bg0,       cmap="gray")
    im_adaptive = axes[1].imshow(ba0,       cmap="gray")
    im_comb     = axes[2].imshow(combined0, cmap="gray")
    im_open     = axes[3].imshow(opened0,   cmap="gray")
    im_res      = axes[4].imshow(result0)
    ttl_global   = axes[0].set_title("Global thresh [ACTIVE]"); axes[0].axis("off")
    ttl_adaptive = axes[1].set_title("Adaptive thresh");        axes[1].axis("off")
    axes[2].set_title("Combined mask");  axes[2].axis("off")
    axes[3].set_title("After opening"); axes[3].axis("off")
    ttl = axes[4].set_title(f"Detected pores: {n0}"); axes[4].axis("off")

    if has_weld_model:
        axes[5].imshow(static_weld_vis if static_weld_vis is not None else img_rgb)
        axes[5].set_title("YOLO weld region (reference)"); axes[5].axis("off")

    sl_axes   = [plt.axes([0.12, y, 0.62, 0.03]) for y in (0.43, 0.38, 0.33, 0.28, 0.23, 0.18, 0.13, 0.08)]
    sl_thresh = widgets.Slider(sl_axes[0], "Threshold",        0,   255, valinit=INIT_THRESH,  valstep=1)
    sl_block  = widgets.Slider(sl_axes[1], "Adaptive block",   11,  101, valinit=INIT_BLOCK,   valstep=2)
    sl_c      = widgets.Slider(sl_axes[2], "Adaptive C",        0,   50, valinit=INIT_C,       valstep=1)
    sl_area   = widgets.Slider(sl_axes[3], "Min area (px²)",    5,  500, valinit=INIT_AREA)
    sl_circ   = widgets.Slider(sl_axes[4], "Min circularity",  0.1,  1.0, valinit=INIT_CIRC)
    sl_open   = widgets.Slider(sl_axes[5], "Open kernel",       1,   51, valinit=INIT_OPEN,    valstep=2)
    sl_erode  = widgets.Slider(sl_axes[6], "Erode mask iters",  0,   30, valinit=INIT_ERODE,   valstep=1)
    sl_aspect = widgets.Slider(sl_axes[7], "Min aspect ratio", 0.0,  1.0, valinit=INIT_ASPECT)

    fig.text(0.76, 0.08, f"Auto darkness: {auto_darkness:.1f}", fontsize=9, color="gray",
             ha="left", va="bottom")

    # Adaptive sliders start dimmed (global is default)
    for sl in (sl_block, sl_c):
        sl.label.set_alpha(0.35)
        sl.poly.set_alpha(0.35)

    ax_check    = plt.axes([0.76, 0.36, 0.20, 0.10])
    cb_adaptive = widgets.CheckButtons(ax_check, ["Use adaptive"], [False])

    def _update(_):
        bg, ba, comb, opnd, res, n = _recompute(
            sl_thresh.val, sl_area.val, sl_circ.val,
            sl_open.val, sl_erode.val,
            sl_block.val, sl_c.val, sl_aspect.val)
        im_global.set_data(bg)
        im_adaptive.set_data(ba)
        im_comb.set_data(comb)
        im_open.set_data(opnd)
        im_res.set_data(res)
        ttl.set_text(f"Detected pores: {n}")
        fig.canvas.draw_idle()

    def _on_toggle(_):
        use_adaptive[0] = cb_adaptive.get_status()[0]
        if use_adaptive[0]:
            ttl_global.set_text("Global thresh")
            ttl_adaptive.set_text("Adaptive thresh [ACTIVE]")
            sl_thresh.label.set_alpha(0.35);  sl_thresh.poly.set_alpha(0.35)
            sl_block.label.set_alpha(1.0);    sl_block.poly.set_alpha(1.0)
            sl_c.label.set_alpha(1.0);        sl_c.poly.set_alpha(1.0)
        else:
            ttl_global.set_text("Global thresh [ACTIVE]")
            ttl_adaptive.set_text("Adaptive thresh")
            sl_thresh.label.set_alpha(1.0);   sl_thresh.poly.set_alpha(1.0)
            sl_block.label.set_alpha(0.35);   sl_block.poly.set_alpha(0.35)
            sl_c.label.set_alpha(0.35);       sl_c.poly.set_alpha(0.35)
        _update(None)

    cb_adaptive.on_clicked(_on_toggle)
    fig.canvas.mpl_connect("button_release_event", _update)

    plt.show()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/01/Projects/weld-defect-detection/data/porosity_val/9pddbrr9enw21_jpg.rf.f49f6375b564a79757b3d1f27780cc6c.jpg"
    interactive_tune(path)
