import cv2
import numpy as np
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# TECHNICAL OVERLAY
# ============================================================

def draw_technical_overlay(
    image_path,
    line_params,
    discontinuity_masks,
    weld_mask,
    porosity_data,
    crack_masks
):
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    overlay = img.copy()

    has_discontinuity = (
        discontinuity_masks is not None and len(discontinuity_masks) > 0
    )

    # --- 1. Base Weld Mask (Blue) ---
    if weld_mask is not None and not has_discontinuity:
        pts = [np.array(poly, dtype=np.int32) for poly in weld_mask]
        cv2.fillPoly(overlay, pts, color=[255, 0, 0])  # Blue (BGR)

    # --- 2. Discontinuity Masks & Lines ---
    if has_discontinuity:
        for i, mask in enumerate(discontinuity_masks):
            hue = int(180 * i / max(len(discontinuity_masks), 1))
            hsv_color = np.uint8([[[hue, 255, 255]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()

            # Mask
            if isinstance(mask, np.ndarray):
                overlay[mask > 0] = bgr_color
            else:
                pts = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], color=bgr_color)

            # Line
            if line_params and i < len(line_params):
                lp = line_params[i]
                m, b = lp.get("m"), lp.get("b")
                if m is not None and b is not None:
                    x1, x2 = 0, w
                    y1, y2 = int(m * x1 + b), int(m * x2 + b)
                    cv2.line(img, (x1, y1), (x2, y2), bgr_color, 2, cv2.LINE_AA)

    # Blend stage
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # --- 3. Defect Overlays ---
    defect_overlay = img.copy()

    # A. Cracks (Red)
    if crack_masks:
        for poly_points in crack_masks:
            pts = np.array(poly_points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(defect_overlay, [pts], (0, 0, 255))

    # B. Porosity
    if porosity_data:
        iso_colors = {
            "B": (50, 255, 50),
            "C": (0, 255, 255),
            "D": (0, 165, 255),
            "F": (0, 0, 255),
            "FAIL": (0, 0, 255),
        }

        for p in porosity_data:
            contour = p.get("_contour")
            if contour is not None:
                grade = str(p.get("grade", "F")).upper()
                color = iso_colors.get(grade, (0, 0, 255))
                cv2.fillPoly(defect_overlay, [contour], color)
                cv2.drawContours(img, [contour], -1, color, 1)

    cv2.addWeighted(defect_overlay, 0.5, img, 0.5, 0, img)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================
# TEXT RENDERING
# ============================================================

def render_report_column(text, width=1200, height=900, title="Moondream Inspection Report"):
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("arial.ttf", 32)
        body_font = ImageFont.truetype("arial.ttf", 24)
    except:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    # Header
    draw.rectangle([0, 0, width, 55], fill=(44, 62, 80))
    draw.text((20, 12), title, fill=(255, 255, 255), font=title_font)

    y = 75
    for line in text.split("\n"):
        if not line.strip():
            y += 10
            continue

        for wrapped in textwrap.wrap(line, width=110):
            draw.text((20, y), wrapped, fill=(0, 0, 0), font=body_font)
            y += 26

    return img


# ============================================================
# FINAL COMPOSITION
# ============================================================

def create_comparison_composition(
    image_path,
    report_text,
    output_path,
    line_params=None,
    discontinuity_masks=None,
    weld_mask=None,
    porosity_data=None,
    crack_masks=None,
):
    # --- Overlay ---
    overlay_rgb = draw_technical_overlay(
        image_path,
        line_params,
        discontinuity_masks,
        weld_mask,
        porosity_data,
        crack_masks,
    )

    if overlay_rgb is None:
        return

    top_img = Image.fromarray(overlay_rgb)

    canvas_w = 1200
    new_h = int(canvas_w * (top_img.height / top_img.width))
    top_img = top_img.resize((canvas_w, new_h), Image.LANCZOS)

    # --- Text ---
    text_img = render_report_column(
        report_text,
        width=canvas_w,
        height=900,
        title="Moondream Inspection Report",
    )

    # --- Stitch ---
    final_img = Image.new(
        "RGB",
        (canvas_w, top_img.height + text_img.height),
        (255, 255, 255),
    )

    final_img.paste(top_img, (0, 0))
    final_img.paste(text_img, (0, top_img.height))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(output_path, quality=95)
