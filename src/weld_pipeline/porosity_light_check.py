"""
YOLO-free pore detection using classical image processing.

Pipeline:
  1. Grayscale → median blur          (noise suppression)
  2. Adaptive threshold               (dark-blob binary mask)
  3. Canny on blurred image           (edge map)
  4. Combine mask + dilated edges     (close pore boundaries)
  5. Filter contours by area and circularity  (pore-shape classifier)
  6. Visualise all stages
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def porosity_light_check(
    image_path: str,
    median_ksize: int       = 5,     # median blur kernel (must be odd)
    adaptive_block_size: int = 51,   # adaptive threshold neighbourhood (must be odd > 1)
    adaptive_c: int         = 330,     # constant subtracted from local mean/gaussian
    canny_low: int          = 30,    # Canny lower hysteresis threshold
    canny_high: int         = 100,   # Canny upper hysteresis threshold
    min_area_px: float      = 25.0,  # minimum contour area in pixels
    min_circularity: float  = 0.45,  # 4π·area/perimeter²  (1.0 = perfect circle)
    visualize: bool         = True,
) -> list[dict]:
    """
    Detect pore-like shapes without YOLO.
    Returns a list of dicts: {contour, area, circularity, bbox}.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

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

    # ── 5. Contour detection & shape filtering ──────────────────────────────────
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pores = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue

        perimeter    = cv2.arcLength(cnt, True)
        circularity  = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
        if circularity < min_circularity:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        pores.append({
            "contour":     cnt,
            "area":        round(float(area), 2),
            "circularity": round(float(circularity), 3),
            "bbox":        [x, y, x + w, y + h],
        })

    # ── 6. Visualisation ────────────────────────────────────────────────────────
    if visualize:
        result = img_rgb.copy()
        for p in pores:
            cv2.drawContours(result, [p["contour"]], -1, (255, 50, 50), 2)
            x1, y1, x2, y2 = p["bbox"]
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 200, 0), 1)

        panels = [
            ("1: Original",              img_rgb,  None),
            ("2: Median blur",           blurred,  "gray"),
            ("3: Adaptive threshold",    adaptive, "gray"),
            ("4: Canny edges",           edges,    "gray"),
            ("5: Combined mask",         combined, "gray"),
            (f"6: Pores ({len(pores)})", result,   None),
        ]

        fig, axes = plt.subplots(1, len(panels), figsize=(30, 5))
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
) -> None:
    """
    Interactive matplotlib window with sliders for binarisation and shape parameters.
    Median blur kernel, Canny thresholds are fixed; sliders cover the rest.
    """
    import matplotlib.widgets as widgets

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, median_ksize)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _recompute(thresh, min_area, min_circ):
        _, binary = cv2.threshold(blurred, int(thresh), 255, cv2.THRESH_BINARY_INV)
        edges     = cv2.Canny(cv2.bitwise_not(binary), canny_low, canny_high)
        edges_d   = cv2.dilate(edges, kernel, iterations=1)
        combined  = cv2.bitwise_or(binary, edges_d)
        combined  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pore_cnts = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            perim = cv2.arcLength(cnt, True)
            circ  = (4 * np.pi * area / perim ** 2) if perim > 0 else 0.0
            if circ < min_circ:
                continue
            pore_cnts.append(cnt)

        result = img_rgb.copy()
        for cnt in pore_cnts:
            cv2.drawContours(result, [cnt], -1, (255, 50, 50), 2)

        return binary, combined, result, len(pore_cnts)

    INIT_THRESH = 127
    INIT_AREA   = 25.0
    INIT_CIRC   = 0.45

    binary0, combined0, result0, n0 = _recompute(INIT_THRESH, INIT_AREA, INIT_CIRC)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.subplots_adjust(bottom=0.25)

    im_bin  = axes[0].imshow(binary0,   cmap="gray")
    im_comb = axes[1].imshow(combined0, cmap="gray")
    im_res  = axes[2].imshow(result0)
    axes[0].set_title("Binary threshold"); axes[0].axis("off")
    axes[1].set_title("Combined mask");    axes[1].axis("off")
    ttl = axes[2].set_title(f"Detected pores: {n0}"); axes[2].axis("off")

    sl_axes   = [plt.axes([0.12, y, 0.76, 0.03]) for y in (0.15, 0.10, 0.05)]
    sl_thresh = widgets.Slider(sl_axes[0], "Threshold",       0,   255, valinit=INIT_THRESH, valstep=1)
    sl_area   = widgets.Slider(sl_axes[1], "Min area (px²)",  5,   500, valinit=INIT_AREA)
    sl_circ   = widgets.Slider(sl_axes[2], "Min circularity", 0.1,  1.0, valinit=INIT_CIRC)

    def _update(_):
        bn, comb, res, n = _recompute(sl_thresh.val, sl_area.val, sl_circ.val)
        im_bin.set_data(bn)
        im_comb.set_data(comb)
        im_res.set_data(res)
        ttl.set_text(f"Detected pores: {n}")
        fig.canvas.draw_idle()

    for sl in (sl_thresh, sl_area, sl_circ):
        sl.on_changed(_update)

    plt.show()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/oljk/Projects/weld-pipeline/data/porosity_val/Screenshot 2026-06-23 at 16.19.09.png"
    interactive_tune(path)
