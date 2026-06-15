from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from ultralytics import YOLO
from PIL import Image
from weld_pipeline import timing


def _calibrate_from_aruco(image_bgr: np.ndarray, marker_size_mm: float) -> float | None:
    """Returns px_to_mm derived from the first ArUco marker found, or None."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image_bgr)
    if ids is None or len(corners) == 0:
        return None
    side_px = np.mean([
        np.mean([
            np.linalg.norm(c[0][1] - c[0][0]),
            np.linalg.norm(c[0][2] - c[0][1]),
            np.linalg.norm(c[0][3] - c[0][2]),
            np.linalg.norm(c[0][0] - c[0][3]),
        ])
        for c in corners
    ])
    return marker_size_mm / side_px


def _isolate_pore_pixels(crop_gray: np.ndarray, mask_bin: np.ndarray, otsu_multiplier: float) -> np.ndarray:
    """Keep pixels darker than Otsu*multiplier, scoped to the YOLO mask region."""
    if not mask_bin.any():
        return np.zeros_like(crop_gray, dtype=np.uint8)
    masked_pixels = crop_gray[mask_bin > 0].reshape(-1, 1)
    thresh_val, _ = cv2.threshold(masked_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_mask = (crop_gray < int(thresh_val * otsu_multiplier)).astype(np.uint8) * 255
    return cv2.bitwise_and(dark_mask, (mask_bin * 255).astype(np.uint8))


def _deduplicate_contours(contours_conf: list, iou_threshold: float = 1.0) -> list:
    """Remove overlapping contours by bounding-box IoU, keeping the larger area."""
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


def _merge_close_contours(contours_conf: list, proximity_px: int, crop_shape: tuple) -> list:
    """
    Merge contours within proximity_px of each other (ISO 5817 cluster logic).
    Dilation connects nearby pores; connected components define clusters;
    contours are re-extracted from the original fills per cluster.
    """
    if len(contours_conf) < 2:
        return contours_conf

    h, w = crop_shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    for cnt, _ in contours_conf:
        cv2.drawContours(canvas, [cnt], -1, 255, cv2.FILLED)

    ksize = max(3, proximity_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(canvas, kernel)

    num_labels, label_map = cv2.connectedComponents(dilated)

    merged = []
    for lbl in range(1, num_labels):
        region = (label_map == lbl).astype(np.uint8) & (canvas > 0).astype(np.uint8)
        if not region.any():
            continue
        confs = [conf for cnt, conf in contours_conf
                 if cv2.countNonZero(
                     cv2.bitwise_and(
                         (label_map == lbl).astype(np.uint8),
                         cv2.drawContours(np.zeros((h, w), np.uint8), [cnt], -1, 1, cv2.FILLED)
                     )
                 ) > 0]
        out_cnts, _ = cv2.findContours(region * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in out_cnts:
            merged.append((c, max(confs) if confs else 0.0))
    return merged


def porosity_check(image_path: str,
                   model_path: str,
                   px_to_mm: float,
                   throat_thickness: float = 10.0,
                   otsu_multiplier: float = 0.6,
                   marker_size_mm: float = 10.0,
                   min_pore_size_mm: float = 0.5,
                   pore_cluster_distance_mm: float = 1.0,
                   use_intensity_filter: bool = True,
                   visualize: bool = True,
                   seg_model=None) -> list:
    """
    Implements Double YOLO for Porosity Metrology.
    Returns: A list of dictionaries in the specific 'Crack' style format.
    """
    model = seg_model
    _img_name = Path(image_path).name

    pil_img = Image.open(image_path).convert("RGB")
    image_rgb = np.array(pil_img)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if image_rgb is None: return []

    h_orig, w_orig = image_rgb.shape[:2]
    gray_full = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # ArUco calibration: override px_to_mm if markers are present
    aruco_px_to_mm = _calibrate_from_aruco(image_bgr, marker_size_mm)
    if aruco_px_to_mm is not None:
        print(f"ArUco calibration: px_to_mm = {aruco_px_to_mm:.5f} (overrides {px_to_mm})")
        px_to_mm = aruco_px_to_mm

    # ISO 5817 Limits for Surface Pore (2017)
    if throat_thickness <= 3.0:
        # Strict limits for thin plates
        lim_B = 0.0  # Not permitted
        lim_C = 0.0  # Not permitted
        lim_D = 0.3 * throat_thickness
    else:
        # Limits for plates > 3mm
        lim_B = 0.0
        lim_C = min(0.2 * throat_thickness, 2.0)
        lim_D = min(0.3 * throat_thickness, 3.0)

    # Stage 1: Weld Detection (Class 3)
    with timing.track(_img_name, "porosity_check", "yolo_stage1"):
        stage1_results = model.predict(image_bgr, conf=0.10, iou=0.2, classes=3, verbose=False, show=False, save=False)

    pore_entries = []
    yolo_raw_contours = []
    pad_percent = 0.2

    if stage1_results[0].boxes is not None:
        for _box_idx, weld_box in enumerate(stage1_results[0].boxes):
            x1_w, y1_w, x2_w, y2_w = map(int, weld_box.xyxy[0].cpu().numpy())
            pad_x = int((x2_w - x1_w) * pad_percent)
            pad_y = int((y2_w - y1_w) * pad_percent)

            x1_c, y1_c = max(0, x1_w - pad_x), max(0, y1_w - pad_y)
            x2_c, y2_c = min(w_orig, x2_w + pad_x), min(h_orig, y2_w + pad_y)

            crop_rgb  = image_rgb[y1_c:y2_c, x1_c:x2_c]
            crop_gray = gray_full[y1_c:y2_c, x1_c:x2_c]

            # Stage 2: Pore Detection (Class 1)
            with timing.track(_img_name, "porosity_check", "yolo_stage2", _box_idx):
                stage2_results = model.predict(crop_rgb, conf=0.2, iou=0.1, classes=1, verbose=False)

            if stage2_results[0].masks is not None:
                c_h, c_w = crop_rgb.shape[:2]
                boxes = stage2_results[0].boxes
                min_radius_px = (min_pore_size_mm / 2) / px_to_mm

                # Collect all valid contours across all masks
                raw_contours = []
                for i, mask_tensor in enumerate(stage2_results[0].masks.data):
                    m_np = cv2.resize(mask_tensor.cpu().numpy(), (c_w, c_h), interpolation=cv2.INTER_NEAREST)
                    m_bin = (m_np > 0.5).astype(np.uint8)

                    # Capture raw YOLO contours (before any filtering) in full-image coords
                    yolo_cnts, _ = cv2.findContours((m_bin * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for yc in yolo_cnts:
                        yc_mapped = yc.copy()
                        yc_mapped[:, :, 0] += x1_c
                        yc_mapped[:, :, 1] += y1_c
                        yolo_raw_contours.append(yc_mapped)

                    refined_255 = _isolate_pore_pixels(crop_gray, m_bin, otsu_multiplier) if use_intensity_filter else (m_bin * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(refined_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    conf = float(boxes.conf[i].cpu().numpy())
                    for cnt in contours:
                        _, r_px = cv2.minEnclosingCircle(cnt)
                        if r_px >= min_radius_px:
                            raw_contours.append((cnt, conf))

                # Merge nearby pores (ISO 5817 cluster logic), then grade survivors
                proximity_px = max(1, int(pore_cluster_distance_mm / px_to_mm))
                deduped = _deduplicate_contours(raw_contours)
                merged = _merge_close_contours(deduped, proximity_px, crop_rgb.shape)
                for cnt, conf in merged:
                    _, radius_px = cv2.minEnclosingCircle(cnt)
                    d_mm = float(radius_px) * 2 * px_to_mm

                    if d_mm == 0:
                        label, grade_color = "B", (50, 255, 50)
                    elif d_mm <= lim_B and lim_B > 0:
                        label, grade_color = "B", (50, 255, 50)
                    elif d_mm <= lim_C and lim_C > 0:
                        label, grade_color = "C", (255, 255, 0)
                    elif d_mm <= lim_D:
                        label, grade_color = "D", (255, 165, 0)
                    else:
                        label, grade_color = "F", (255, 0, 0)

                    cnt_mapped = cnt.copy()
                    cnt_mapped[:, :, 0] += x1_c
                    cnt_mapped[:, :, 1] += y1_c
                    bx, by, bw, bh = cv2.boundingRect(cnt_mapped)

                    pore_entries.append({
                        "name": "Pore",
                        "class": 1,
                        "confidence": round(conf, 5),
                        "box": {
                            "x1": float(bx),
                            "y1": float(by),
                            "x2": float(bx + bw),
                            "y2": float(by + bh)
                        },
                        "size": round(float(d_mm), 3),
                        "grade": label,
                        "_contour": cnt_mapped,
                        "_color": grade_color
                    })

    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        titles = ["Stage 1: Detection", "Stage 2: Refined Masks", "Stage 3: ISO Grading"]
        for i, ax in enumerate(axes):
            canvas = image_rgb.copy()
            for p in pore_entries:
                b = p["box"]
                # if i == 0:
                #     cv2.rectangle(canvas, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), (50, 255, 50), 2)
                if i == 1:
                    cv2.drawContours(canvas, [p["_contour"]], -1, (0, 255, 0), 2)
                else:
                    cv2.drawContours(canvas, [p["_contour"]], -1, p["_color"], 2)
                    # cv2.rectangle(canvas, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), p["_color"], 3)
            ax.imshow(canvas); ax.set_title(titles[i]); ax.axis('off')
        plt.tight_layout(); plt.show()

    # Final cleanup to remove visualization-only keys
    final_output = []
    for p in pore_entries:
        final_output.append({
            "name": p["name"],
            "class": p["class"],
            "confidence": p["confidence"],
            "box": p["box"],
            "size": p["size"],
            "grade": p["grade"]
        })
    
    return final_output, pore_entries, yolo_raw_contours

if __name__ == '__main__':
    results = porosity_check(
        image_path="../../data/welding-defect.jpeg", 
        model_path="../../models/best.pt", 
        px_to_mm=0.105
    )
    print(json.dumps(results, indent=2))