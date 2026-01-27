from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
import cv2
import textwrap
import math
import os

def draw_technical_overlay(image_path, line_params, discontinuity_masks, weld_mask, porosity_data, crack_masks, disc_bool=False,  show_labels=True, area_percent=0.008):
    """
    Draws a beautiful technical overlay. 
    area_percent: 0.01 means the font height is roughly 1% of the diagonal/base scale of the image.
    """
    img_cv = cv2.imread(str(image_path))
    if img_cv is None: return None
    
    img_h, img_w = img_cv.shape[:2]
    # Use geometric mean of dimensions for a stable linear scale factor
    base_scale = math.sqrt(img_w * img_h)
    
    # --- Area-Based Scalable Constants ---
    # Adjust 3.5 to make text larger or smaller globally
    FONT_SIZE = max(10, int(base_scale * area_percent * 3.5)) 
    LINE_WIDTH = max(2, int(base_scale * 0.002))
    THICK_LINE = max(2, int(base_scale * 0.005))
    PAD = int(FONT_SIZE * 0.35)
    
    base_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)).convert("RGBA")
    mask_layer = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    ui_layer = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    
    draw_mask = ImageDraw.Draw(mask_layer)
    draw_ui = ImageDraw.Draw(ui_layer)

    BRAND_BLUE = (0, 122, 255, 255)
    CRACK_RED = (255, 45, 85, 255)
    WHITE = (255, 255, 255, 255)

    # --- Robust Font Selection ---
    font_main = 'C:/Windows/Fonts/arial.ttf'
    # List of common bold system fonts across Windows, Mac, and Linux
    font_names = [
        "arial.ttf", "Arial.ttf", "DejaVuSans.ttf", 
        "LiberationSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf", "/Library/Fonts/Arial.ttf"
    ]

    for f_name in font_names:
        try:
            font_main = ImageFont.truetype(f_name, FONT_SIZE)
            break # Success!
        except:
            continue

    if font_main is None:
        print("Warning: No TTF fonts found. Scaling will not work with default font.")
        font_main = ImageFont.load_default()

    has_discontinuity = discontinuity_masks is not None and len(discontinuity_masks) > 0 and disc_bool

    # --- 1. Weld Mask ---
    if weld_mask is not None and not has_discontinuity:
        for poly in weld_mask:
            # 1. Convert polygon points
            # Ensure poly is handled correctly whether it's a list or a nested array
            pts = [tuple(p) for p in poly]
            
            # 2. Draw the transparent mask for THIS specific weld
            # Using i to potentially vary colors if you wanted to
            draw_mask.polygon(pts, fill=BRAND_BLUE[:3] + (100,))
        
            if show_labels:
                # 3. Calculate Bounding Box for this specific weld
                # Flatten points if necessary for cv2.boundingRect
                all_pts = np.array(poly).reshape(-1, 2)
                bx, by, bw, bh = cv2.boundingRect(all_pts)
                
                # 4. Draw UI elements for this specific weld
                draw_ui.rectangle([bx, by, bx + bw, by + bh], outline=BRAND_BLUE, width=LINE_WIDTH)
                
                # Label each weld (e.g., WELD 1, WELD 2)
                label = f"WELD"
                t_bbox = draw_ui.textbbox((bx, by), label, font=font_main)
                tw, th = t_bbox[2]-t_bbox[0], t_bbox[3]-t_bbox[1]
                
                # Draw label background and text
                draw_ui.rectangle([bx, by - th - (PAD*2), bx + tw + (PAD*2), by], fill=BRAND_BLUE)
                draw_ui.text((bx + PAD, by - th - PAD), label, font=font_main, fill=WHITE)

    # --- 2. Discontinuity Masks & Lines ---
    if has_discontinuity:
        for i, mask in enumerate(discontinuity_masks):
            # Generate rotating HSL color
            hue = int(360 * i / len(discontinuity_masks))
            rgb_tuple = ImageColor.getrgb(f"hsl({hue}, 75%, 50%)")
            base_color = rgb_tuple + (120,)
            fill_color = rgb_tuple + (255,)
            
            if isinstance(mask, np.ndarray):
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
                mask_layer.paste(base_color, (0, 0), mask=mask_pil)
                coords = np.argwhere(mask > 0)
                if coords.size > 0:
                    y0, x0 = coords.min(axis=0); y1, x1 = coords.max(axis=0)
                    dx, dy, dw, dh = x0, y0, x1 - x0, y1 - y0
                else: dx, dy, dw, dh = 0, 0, 0, 0
            else:
                pts = [tuple(p) for p in mask]
                draw_mask.polygon(pts, fill=base_color)
                np_pts = np.array(mask)
                dx, dy, dw, dh = cv2.boundingRect(np_pts)

            if i < len(line_params):
                lp = line_params[i]
                m, b = lp.get('m'), lp.get('b')
                if m is not None and b is not None:
                    lx1, lx2 = 0, base_img.width
                    ly1, ly2 = int(m * lx1 + b), int(m * lx2 + b)
                    draw_ui.line([(lx1, ly1), (lx2, ly2)], fill=fill_color, width=THICK_LINE)

            if show_labels and dw > 0:
                label = "WELD"
                draw_ui.rectangle([dx, dy, dx + dw, dy + dh], outline=fill_color, width=LINE_WIDTH)
                t_bbox = draw_ui.textbbox((dx, dy), label, font=font_main)
                tw, th = t_bbox[2] - t_bbox[0], t_bbox[3] - t_bbox[1]
                
                # Dynamic contrast logic
                lum = (rgb_tuple[0]*0.299 + rgb_tuple[1]*0.587 + rgb_tuple[2]*0.114)/255
                txt_color = (0, 0, 0, 255) if lum > 0.5 else WHITE
                
                draw_ui.rectangle([dx, dy - th - (PAD*2), dx + tw + (PAD*2), dy], fill=fill_color)
                draw_ui.text((dx + PAD, dy - th - PAD), label, font=font_main, fill=txt_color)

    # --- 3. Cracks ---
    if crack_masks:
        for poly in crack_masks:
            pts = [tuple(p) for p in poly]
            draw_mask.polygon(pts, fill=CRACK_RED[:3] + (160,))
            if show_labels:
                cx, cy, cw, ch = cv2.boundingRect(np.array(poly))
                draw_ui.rectangle([cx, cy, cx + cw, cy + ch], outline=CRACK_RED, width=LINE_WIDTH)
                label = "CRACK"
                t_bbox = draw_ui.textbbox((cx, cy), label, font=font_main)
                tw, th = t_bbox[2]-t_bbox[0], t_bbox[3]-t_bbox[1]
                draw_ui.rectangle([cx, cy - th - (PAD*2), cx + tw + (PAD*2), cy], fill=CRACK_RED)
                draw_ui.text((cx + PAD, cy - th - PAD), label, font=font_main, fill=WHITE)

    # --- 4. Porosity ---
    if porosity_data:
        iso_colors = {'B': (0, 255, 0), 'C': (255, 255, 0), 'D': (255, 128, 0), 'FAIL': (255, 0, 0)}
        for p in porosity_data:
            contour = p.get('_contour')
            if contour is not None:
                grade = str(p.get('grade', 'F')).upper()
                base_rgb = iso_colors.get(grade, iso_colors['FAIL'])
                pts = [tuple(pt[0]) for pt in contour]
                draw_mask.polygon(pts, fill=base_rgb + (100,), outline=base_rgb + (255,)) # Porosity opacity
                
                if show_labels:
                    px, py, pw, ph = cv2.boundingRect(contour)
                    draw_ui.rectangle([px, py, px + pw, py + ph], outline=base_rgb + (255,), width=max(1, LINE_WIDTH//2))
                    label = f"P-{grade}"
                    t_bbox = draw_ui.textbbox((px, py), label, font=font_main)
                    tw, th = t_bbox[2]-t_bbox[0], t_bbox[3]-t_bbox[1]
                    
                    lum = (base_rgb[0]*0.299 + base_rgb[1]*0.587 + base_rgb[2]*0.114)/255
                    txt_color = (0,0,0) if lum > 0.5 else WHITE
                    
                    draw_ui.rectangle([px, py - th - (PAD*2), px + tw + (PAD*2), py], fill=base_rgb + (255,))
                    draw_ui.text((px + PAD, py - th - PAD), label, font=font_main, fill=txt_color)

    # Composite Layers
    composite = Image.alpha_composite(base_img, mask_layer)
    if show_labels:
        composite = Image.alpha_composite(composite, ui_layer)
    
    return np.array(composite.convert("RGB"))

def render_report_column(text, width=600, height=1000, title=""):
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 32)
        body_font = ImageFont.truetype("arial.ttf", 20)
    except:
        title_font = ImageFont.load_default(); body_font = ImageFont.load_default()

    # Professional header bar
    draw.rectangle([0, 0, width, 60], fill=(44, 62, 80))
    draw.text((25, 12), title.upper(), fill=(255, 255, 255), font=title_font)

    y_offset = 80
    for line in text.split('\n'):
        if not line.strip(): y_offset += 15; continue
        for w_line in textwrap.wrap(line, width=60):
            draw.text((25, y_offset), w_line, fill=(30, 30, 30), font=body_font)
            y_offset += 28
    return img

def create_comparison_composition(image_path, report_v_text, report_g_text, output_path, **kwargs):
    # 1. Setup the main storage folder
    root_folder = "../../paper_imgs"
    
    # 2. Get the base filename (e.g., 'welding_sample_01')
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 3. Create a specific subfolder for this image
    # Path will look like: ../paper_imgs/welding_sample_01/
    target_dir = os.path.join(root_folder, base_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 4. Save Original Image
    img_original = Image.open(image_path)
    img_original.save(os.path.join(target_dir, f"{base_name}_original.jpg"))

    # Generate technical overlays
    clean_rgb = draw_technical_overlay(image_path, show_labels=False, **kwargs)
    label_rgb = draw_technical_overlay(image_path, show_labels=True, **kwargs)
    
    img_clean = Image.fromarray(clean_rgb)
    img_label = Image.fromarray(label_rgb)
    
    # 5. Save Clean and Labeled images into the subfolder
    img_clean.save(os.path.join(target_dir, f"{base_name}_clean.jpg"))
    img_label.save(os.path.join(target_dir, f"{base_name}_label.jpg"))

    # 6. Save Model Inference to Markdown inside the subfolder
    md_path = os.path.join(target_dir, f"{base_name}_inference.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Inference Report: {base_name}\n\n")
        f.write(f"## VLM Analysis\n{report_v_text}\n\n")
        f.write(f"--- \n\n")
        f.write(f"## VLM + YOLO Analysis\n{report_g_text}\n")
    
    # --- Assembly Logic for the Final Composition ---
    canvas_w = 1200
    h_img = int(canvas_w * (img_clean.height / img_clean.width))
    img_clean_res = img_clean.resize((canvas_w, h_img), Image.LANCZOS)
    img_label_res = img_label.resize((canvas_w, h_img), Image.LANCZOS)
    
    text_h = 900
    col_l = render_report_column(report_v_text, 600, text_h, "VLM")
    col_r = render_report_column(report_g_text, 600, text_h, "VLM + YOLO")
    
    total_height = (h_img * 2) + text_h
    final_img = Image.new('RGB', (canvas_w, total_height), (255, 255, 255))
    
    final_img.paste(img_clean_res, (0, 0))
    final_img.paste(img_label_res, (0, h_img))
    final_img.paste(col_l, (0, h_img * 2))
    final_img.paste(col_r, (600, h_img * 2))
    
    draw = ImageDraw.Draw(final_img)
    draw.line([(0, h_img), (canvas_w, h_img)], fill=(200, 200, 200), width=2)
    draw.line([(0, h_img*2), (canvas_w, h_img*2)], fill=(44, 62, 80), width=4)
    
    # Save the final comparison strip
    final_img.save(output_path, quality=95)