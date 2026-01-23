import cv2
import numpy as np
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def draw_technical_overlay(image_path, line_params, discontinuity_masks, weld_mask, porosity_data, crack_masks):
    img = cv2.imread(str(image_path))
    if img is None: return None
    
    h, w = img.shape[:2]
    overlay = img.copy()
    
    # Check if we should skip the base weld mask
    has_discontinuity = discontinuity_masks is not None and len(discontinuity_masks) > 0

    # --- 1. Base Weld Mask (Blue) ---
    # Only displayed if NO discontinuity is detected
    if weld_mask is not None and not has_discontinuity:
        pts = [np.array(poly, dtype=np.int32) for poly in weld_mask]
        cv2.fillPoly(overlay, pts, color=[255, 0, 0]) # BGR Blue

    # --- 2. Discontinuity Masks & Lines (Aligned Colors) ---
    if has_discontinuity:
        for i, mask in enumerate(discontinuity_masks):
            # Generate color matching the logic in discontinuity_check.py
            # Using 180 for OpenCV H (0-179)
            hue = int(180 * i / len(discontinuity_masks))
            hsv_color = np.uint8([[[hue, 255, 255]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
            
            # Draw the Mask on the overlay layer
            if isinstance(mask, np.ndarray):
                overlay[mask > 0] = bgr_color
            else:
                pts = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], color=bgr_color)

            # Draw the Line on the main image layer (Sync color with mask)
            if i < len(line_params):
                lp = line_params[i]
                m, b = lp.get('m'), lp.get('b')
                if m is not None and b is not None:
                    x1, x2 = 0, w
                    y1, y2 = int(m * x1 + b), int(m * x2 + b)
                    # Use the same bgr_color for the line
                    cv2.line(img, (x1, y1), (x2, y2), bgr_color, 2, cv2.LINE_AA)

    # Blend the masks (Stage 5 style transparency)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # --- 3. Porosity & Cracks (Semi-Transparent Overlays) ---
    # Create an overlay for elements that need transparency
    defect_overlay = img.copy()

    # A. Cracks (Red - Type 100)
    if crack_masks:
        for poly_points in crack_masks:
            pts = np.array(poly_points, np.int32).reshape((-1, 1, 2))
            # Draw onto the defect_overlay
            cv2.fillPoly(defect_overlay, [pts], (0, 0, 255)) 

    # B. Porosity (ISO 5817:2023 Color Hierarchy)
    if porosity_data:
        iso_colors = {
            'B': (50, 255, 50),   # Green
            'C': (0, 255, 255),   # Yellow
            'D': (0, 165, 255),   # Orange
            'FAIL': (0, 0, 255),  # Red
            'F': (0, 0, 255)      # Red
        }
        for p in porosity_data:
            contour = p.get('_contour')
            if contour is not None:
                grade = str(p.get('grade', 'F')).upper()
                draw_color = iso_colors.get(grade, (0, 0, 255))
                # Fill the pore with color on the overlay
                cv2.fillPoly(defect_overlay, [contour], draw_color)
                # Optional: Draw a sharper outline on the main image for clarity
                cv2.drawContours(img, [contour], -1, draw_color, 1)

    # --- 4. Final Blending ---
    # defect_alpha: 0.0 is invisible, 1.0 is solid
    defect_alpha = 0.5 
    cv2.addWeighted(defect_overlay, defect_alpha, img, 1 - defect_alpha, 0, img)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def render_report_column(text, width=600, height=1000, title=""):
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 32)
        body_font = ImageFont.truetype("arial.ttf", 20)
    except:
        title_font = ImageFont.load_default(); body_font = ImageFont.load_default()

    draw.rectangle([0, 0, width, 50], fill=(44, 62, 80))
    draw.text((20, 10), title, fill=(255, 255, 255), font=title_font)

    y_offset = 70
    for line in text.split('\n'):
        if not line.strip(): y_offset += 10; continue
        for w_line in textwrap.wrap(line, width=65):
            draw.text((20, y_offset), w_line, fill=(0, 0, 0), font=body_font)
            y_offset += 25
    return img

def create_comparison_composition(
    image_path, 
    report_v_text, 
    report_g_text, 
    output_path,
    line_params=None, 
    discontinuity_masks=None,  # This name must match the call
    weld_mask=None, 
    porosity_data=None,        # Renamed from porosity_masks to reflect dict data
    crack_masks=None,          # The polygon points from cracks_check
):
    # 1. Overlay
    overlay_rgb = draw_technical_overlay(image_path, line_params, discontinuity_masks, 
                                         weld_mask, porosity_data, crack_masks)
    top_img = Image.fromarray(overlay_rgb)
    canvas_w = 1200
    top_img = top_img.resize((canvas_w, int(canvas_w * (top_img.height/top_img.width))), Image.LANCZOS)
    
    # 2. Text Columns
    text_h = 900
    col_l = render_report_column(report_v_text, 600, text_h, "Visual Report")
    col_r = render_report_column(report_g_text, 600, text_h, "Grounded ISO Report")
    
    # 3. Final Stitch
    final_img = Image.new('RGB', (canvas_w, top_img.height + text_h), (255, 255, 255))
    final_img.paste(top_img, (0, 0))
    final_img.paste(col_l, (0, top_img.height))
    final_img.paste(col_r, (600, top_img.height))
    final_img.save(output_path, quality=95)