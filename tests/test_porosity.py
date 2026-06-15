"""
Standalone porosity test script with interactive pore inspection.
Click on any pore mask to see its grade and size.

Usage:
    poetry run python tests/test_porosity.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "wda11s-seg.pt"
IMAGE_PATH   = PROJECT_ROOT / "data" / "tmp" / "2.png"

PX_TO_MM              = 0.105
THROAT_THICKNESS       = 10.0
MARKER_SIZE_MM        = 10.0   # ArUco marker physical size for calibration
OTSU_MULTIPLIER       = 0.6    # fraction of Otsu threshold — lower = more pores kept
MIN_PORE_SIZE_MM      = 0.8   # minimum pore diameter in mm
PORE_MASK_PADDING_PCT = 0.000   # mask dilation as % of image area (0 = off)
PORE_CLUSTER_DIST_MM  = 0.05
USE_INTENSITY_FILTER  = True
IOU_THRESHOLD         = 0.5
FILL_ALPHA            = 0.5   # 0 = transparent, 1 = opaque

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def calibrate_from_aruco(image_bgr, marker_size_mm):
    """Returns (px_to_mm, corners) or (None, []) if no markers found."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image_bgr)
    if ids is None or len(corners) == 0:
        return None, []
    side_px = np.mean([
        np.mean([
            np.linalg.norm(c[0][1] - c[0][0]),
            np.linalg.norm(c[0][2] - c[0][1]),
            np.linalg.norm(c[0][3] - c[0][2]),
            np.linalg.norm(c[0][0] - c[0][3]),
        ])
        for c in corners
    ])
    return marker_size_mm / side_px, corners


def isolate_pore_pixels(crop_gray, mask_bin, otsu_multiplier):
    if not mask_bin.any():
        return np.zeros_like(crop_gray, dtype=np.uint8)
    masked_pixels = crop_gray[mask_bin > 0].reshape(-1, 1)
    thresh_val, _ = cv2.threshold(masked_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_mask = (crop_gray < int(thresh_val * otsu_multiplier)).astype(np.uint8) * 255
    return cv2.bitwise_and(dark_mask, (mask_bin * 255).astype(np.uint8))


def deduplicate_contours(contours_conf, iou_threshold=1.0):
    if not contours_conf:
        return []
    sorted_items = sorted(contours_conf, key=lambda x: cv2.contourArea(x[0]), reverse=True)
    kept = []
    for cnt, conf in sorted_items:
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        duplicate = False
        for k_cnt, _ in kept:
            x2, y2, w2, h2 = cv2.boundingRect(k_cnt)
            ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
            iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
            inter = ix * iy
            union = w1*h1 + w2*h2 - inter
            if union > 0 and inter / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append((cnt, conf))
    return kept


def merge_close_contours(contours_conf, proximity_px, crop_shape):
    if len(contours_conf) < 2 or proximity_px < 1:
        return contours_conf
    h, w = crop_shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    for cnt, _ in contours_conf:
        cv2.drawContours(canvas, [cnt], -1, 255, cv2.FILLED)
    ksize = proximity_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(canvas, kernel)
    num_labels, label_map = cv2.connectedComponents(dilated)
    merged = []
    for lbl in range(1, num_labels):
        region = (label_map == lbl).astype(np.uint8) & (canvas > 0).astype(np.uint8)
        if not region.any():
            continue
        confs = [conf for cnt, conf in contours_conf
                 if cv2.countNonZero(cv2.bitwise_and(
                     (label_map == lbl).astype(np.uint8),
                     cv2.drawContours(np.zeros((h, w), np.uint8), [cnt], -1, 1, cv2.FILLED)
                 )) > 0]
        out_cnts, _ = cv2.findContours(region * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in out_cnts:
            merged.append((c, max(confs) if confs else 0.0))
    return merged


def grade_pore(d_mm, lim_C, lim_D):
    if d_mm <= 0:
        return "B", (50, 255, 50)
    elif lim_C > 0 and d_mm <= lim_C:
        return "C", (255, 255, 0)
    elif d_mm <= lim_D:
        return "D", (255, 165, 0)
    else:
        return "F", (255, 0, 0)

# ---------------------------------------------------------------
# Load image
# ---------------------------------------------------------------
if not IMAGE_PATH.exists():
    print(f"Image not found: {IMAGE_PATH}")
    sys.exit(1)

print(f"Image : {IMAGE_PATH}")
print(f"Model : {MODEL_PATH}")

img_rgb  = np.array(Image.open(IMAGE_PATH).convert("RGB"))
img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
h_orig, w_orig = img_bgr.shape[:2]

# ArUco calibration
px_to_mm = PX_TO_MM
aruco_cal, aruco_corners = calibrate_from_aruco(img_bgr, MARKER_SIZE_MM)
if aruco_cal is not None:
    print(f"ArUco calibration: px_to_mm = {aruco_cal:.5f} (was {px_to_mm})")
    px_to_mm = aruco_cal

# ISO 5817 limits
if THROAT_THICKNESS <= 3.0:
    lim_C, lim_D = 0.0, 0.3 * THROAT_THICKNESS
else:
    lim_C = min(0.2 * THROAT_THICKNESS, 2.0)
    lim_D = min(0.3 * THROAT_THICKNESS, 3.0)

min_radius_px = (MIN_PORE_SIZE_MM / 2) / px_to_mm
proximity_px = max(0, int(PORE_CLUSTER_DIST_MM / px_to_mm))

# ---------------------------------------------------------------
# Detection
# ---------------------------------------------------------------
model = YOLO(MODEL_PATH)

pore_entries      = []   # graded, full-image coords
yolo_raw_contours = []   # raw YOLO masks before any filtering
crop_bboxes       = []   # (x1_c, y1_c, x2_c, y2_c) for each weld crop

PAD_PERCENT = 0.2   # padding as fraction of the weld bounding box size
stage1 = model.predict(img_bgr, conf=0.10, iou=0.2, classes=3, verbose=False, save=False)

if stage1[0].boxes is not None:
    for weld_box in stage1[0].boxes:
        x1_w, y1_w, x2_w, y2_w = map(int, weld_box.xyxy[0].cpu().numpy())
        pad_x = int((x2_w - x1_w) * PAD_PERCENT)
        pad_y = int((y2_w - y1_w) * PAD_PERCENT)
        x1_c = max(0, x1_w - pad_x);  y1_c = max(0, y1_w - pad_y)
        x2_c = min(w_orig, x2_w + pad_x); y2_c = min(h_orig, y2_w + pad_y)
        crop_bboxes.append((x1_c, y1_c, x2_c, y2_c))

        crop_rgb  = img_rgb[y1_c:y2_c, x1_c:x2_c]
        crop_bgr  = img_bgr[y1_c:y2_c, x1_c:x2_c]
        crop_gray = gray_full[y1_c:y2_c, x1_c:x2_c]
        c_h, c_w  = crop_rgb.shape[:2]

        stage2 = model.predict(crop_rgb, conf=0.2, iou=0.1, classes=1, verbose=False)
        if stage2[0].masks is None:
            continue

        boxes = stage2[0].boxes
        raw_contours = []

        pad_px = int((h_orig * w_orig * PORE_MASK_PADDING_PCT / 100) ** 0.5)
        if pad_px > 0:
            pad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad_px * 2 + 1, pad_px * 2 + 1))

        for i, mask_tensor in enumerate(stage2[0].masks.data):
            m_np  = cv2.resize(mask_tensor.cpu().numpy(), (c_w, c_h), interpolation=cv2.INTER_NEAREST)
            m_bin = (m_np > 0.5).astype(np.uint8)

            if pad_px > 0:
                m_bin = (cv2.dilate(m_bin * 255, pad_kernel) > 0).astype(np.uint8)

            # Raw YOLO contours (no filtering)
            yolo_cnts, _ = cv2.findContours((m_bin * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for yc in yolo_cnts:
                yc_mapped = yc.copy()
                yc_mapped[:, :, 0] += x1_c
                yc_mapped[:, :, 1] += y1_c
                yolo_raw_contours.append(yc_mapped)

            # Filtered mask
            refined = isolate_pore_pixels(crop_gray, m_bin, OTSU_MULTIPLIER) if USE_INTENSITY_FILTER else (m_bin * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            conf = float(boxes.conf[i].cpu().numpy())
            for cnt in cnts:
                _, r_px = cv2.minEnclosingCircle(cnt)
                if r_px >= min_radius_px:
                    raw_contours.append((cnt, conf))

        deduped = deduplicate_contours(raw_contours, IOU_THRESHOLD)
        merged  = merge_close_contours(deduped, proximity_px, crop_bgr.shape)

        for cnt, conf in merged:
            _, radius_px = cv2.minEnclosingCircle(cnt)
            d_mm  = float(radius_px) * 2 * px_to_mm
            label, color = grade_pore(d_mm, lim_C, lim_D)

            cnt_mapped = cnt.copy()
            cnt_mapped[:, :, 0] += x1_c
            cnt_mapped[:, :, 1] += y1_c

            pore_entries.append({
                "grade":      label,
                "size":       round(d_mm, 3),
                "confidence": round(conf, 5),
                "_contour":   cnt_mapped,
                "_color":     color,
            })

print(f"Raw YOLO masks : {len(yolo_raw_contours)}")
print(f"Filtered pores : {len(pore_entries)}")

# Stage 2 on full image (no cropping)
stage2_full = model.predict(img_bgr, classes=1, verbose=False)
fullimg_raw_contours = []
if stage2_full[0].masks is not None:
    for mask_tensor in stage2_full[0].masks.data:
        m_np  = cv2.resize(mask_tensor.cpu().numpy(), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        m_bin = (m_np > 0.5).astype(np.uint8)
        cnts, _ = cv2.findContours((m_bin * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fullimg_raw_contours.extend(cnts)
print(f"Full-image YOLO masks : {len(fullimg_raw_contours)}")

# ---------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------
ISO_COLORS = {'B': (0, 200, 0), 'C': (220, 180, 0), 'D': (255, 120, 0), 'F': (220, 0, 0)}
PALETTE    = [(255, 100, 100), (100, 200, 255), (200, 255, 100), (255, 200, 50)]

def draw_aruco_markers(image, corners, px_to_mm_val):
    for c in corners:
        pts = c[0].astype(int)
        cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(image, f"{px_to_mm_val:.4f} mm/px", (cx - 60, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

yolo_overlay = img_rgb.copy()
for i, cnt in enumerate(yolo_raw_contours):
    c = PALETTE[i % len(PALETTE)]
    cv2.drawContours(yolo_overlay, [cnt], -1, c, cv2.FILLED)
    cv2.drawContours(yolo_overlay, [cnt], -1, (255, 255, 255), 1)
for x1_c, y1_c, x2_c, y2_c in crop_bboxes:
    cv2.rectangle(yolo_overlay, (x1_c, y1_c), (x2_c, y2_c), (0, 200, 255), 2)
if aruco_corners:
    draw_aruco_markers(yolo_overlay, aruco_corners, px_to_mm)

fullimg_overlay = img_rgb.copy()
for i, cnt in enumerate(fullimg_raw_contours):
    c = PALETTE[i % len(PALETTE)]
    cv2.drawContours(fullimg_overlay, [cnt], -1, c, cv2.FILLED)
    cv2.drawContours(fullimg_overlay, [cnt], -1, (255, 255, 255), 1)

def build_filtered_overlay(selected_idx=None):
    base = img_rgb.astype(np.float32)
    img  = base.copy()
    for i, p in enumerate(pore_entries):
        cnt   = p["_contour"]
        color = np.array([255, 255, 255], dtype=np.float32) if i == selected_idx else np.array(p["_color"], dtype=np.float32)
        mask  = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
        img[mask > 0] = base[mask > 0] * (1 - FILL_ALPHA) + color * FILL_ALPHA
        outline = (0, 0, 0) if i == selected_idx else (255, 255, 255)
        cv2.drawContours(img, [cnt], -1, outline, 2)
    return img.astype(np.uint8)

fig, (ax_yolo, ax) = plt.subplots(1, 2, figsize=(20, 8))
ax_yolo.imshow(yolo_overlay)
ax_yolo.set_title(f"YOLO on crops  ({len(yolo_raw_contours)} masks)", fontsize=11)
ax_yolo.axis("off")

ax_img = ax.imshow(build_filtered_overlay())
ax.set_title(f"Filtered  ({len(pore_entries)} pores) — click to inspect", fontsize=11)
ax.axis("off")

annot = ax.annotate(
    "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
    fontsize=10, visible=False,
)

def on_click(event):
    if event.inaxes != ax or event.xdata is None:
        return
    cx, cy = event.xdata, event.ydata
    for i, p in enumerate(pore_entries):
        if cv2.pointPolygonTest(p["_contour"], (float(cx), float(cy)), False) >= 0:
            ax_img.set_data(build_filtered_overlay(selected_idx=i))
            annot.set_text(f"Grade : {p['grade']}\nSize  : {p['size']} mm\nConf  : {p['confidence']:.3f}")
            annot.xy = (cx, cy)
            annot.set_visible(True)
            fig.canvas.draw_idle()
            return
    ax_img.set_data(build_filtered_overlay())
    annot.set_visible(False)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_press_event", on_click)

legend_handles = [
    mpatches.Patch(color=tuple(c / 255 for c in col), label=f"Grade {g}")
    for g, col in ISO_COLORS.items()
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

plt.tight_layout()
plt.show()
