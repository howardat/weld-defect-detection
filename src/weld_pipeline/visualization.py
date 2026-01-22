import cv2
import numpy as np
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def draw_technical_overlay(image_path, line_params, discontinuity_masks, weld_mask, porosity_data, crack_masks):
    img = cv2.imread(str(image_path))
    if img is None: return None
    
    h, w = img.shape[:2]
    # Create a separate layer for transparent elements
    overlay = img.copy()
    
    # --- SECTION A: TRANSPARENT ELEMENTS ---
    # 1. Base Weld Mask (Green)
    if weld_mask is not None:
        pts = [np.array(poly, dtype=np.int32) for poly in weld_mask]
        cv2.fillPoly(overlay, pts, color=[0, 255, 0])

    # 2. Discontinuity Masks (Yellow)
    if discontinuity_masks:
        for d_mask in discontinuity_masks:
            # 1. Convert to numpy array
            pts = np.array(d_mask, dtype=np.int32)
            
            # 2. Reshape if it's a flat list [x, y, x, y...] 
            # or just ensure it is (Points, 1, 2)
            if pts.ndim == 2:
                pts = pts.reshape((-1, 1, 2))
            
            # 3. OpenCV expects a LIST of arrays
            cv2.fillPoly(overlay, [pts], color=[0, 255, 255])

    # Blend overlay into img (0.4 alpha makes it transparent)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # --- SECTION B: SOLID ELEMENTS (Draw directly on 'img') ---
    # 3. Crack Masks (Red)
    if crack_masks:
        for poly_points in crack_masks:
            pts = np.array(poly_points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], (0, 0, 255))
            cv2.polylines(img, [pts], True, (50, 50, 255), 2)

    # 4. Fitted Lines (Cyan)
    if line_params:
        for params in line_params:
            m, b = params.get('m'), params.get('b')
            if m is not None and b is not None:
                x1, x2 = 0, w
                y1, y2 = int(m * x1 + b), int(m * x2 + b)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # 5. Porosity (Grade Coded)
    if porosity_data:
        for p in porosity_data:
            contour = p.get('_contour')
            if contour is None: continue
            
            grade = p.get('grade', 'D').upper()
            # BGR Colors
            colors = {'A': (50, 255, 50), 'B': (0, 255, 255), 
                      'C': (0, 165, 255), 'D': (0, 69, 255)}
            color = colors.get(grade, (0, 0, 255))
            
            cv2.drawContours(img, [contour], -1, color, 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def render_report_column(text, width=600, height=1000, title=""):
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        body_font = ImageFont.truetype("arial.ttf", 16)
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