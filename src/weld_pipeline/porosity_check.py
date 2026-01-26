import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from ultralytics import YOLO

def porosity_check(image_path: str, 
                   model_path: str, 
                   px_to_mm: float, 
                   plate_thickness_s: float = 10.0,
                   visualize: bool = True):
    """
    Implements Double YOLO for Porosity Metrology.
    Returns: A list of dictionaries in the specific 'Crack' style format.
    """
    model = YOLO(model_path)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None: return []
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_bgr.shape[:2]
    gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # ISO 5817 Limits for Surface Pore (2017)
    if plate_thickness_s <= 3.0:
        # Strict limits for thin plates
        lim_B = 0.0  # Not permitted
        lim_C = 0.0  # Not permitted
        lim_D = 0.3 * plate_thickness_s
    else:
        # Limits for plates > 3mm
        lim_B = min(0.2 * plate_thickness_s, 2.0)
        lim_C = min(0.3 * plate_thickness_s, 3.0)
        lim_D = min(0.4 * plate_thickness_s, 4.0)

    # Stage 1: Weld Detection (Class 3)
    stage1_results = model.predict(image_rgb, conf=0.05, classes=3, verbose=False, save=False)
    
    pore_entries = []
    pad = 20

    if stage1_results[0].boxes is not None:
        for weld_box in stage1_results[0].boxes:
            x1_w, y1_w, x2_w, y2_w = map(int, weld_box.xyxy[0].cpu().numpy())
            
            x1_c, y1_c = max(0, x1_w - pad), max(0, y1_w - pad)
            x2_c, y2_c = min(w_orig, x2_w + pad), min(h_orig, y2_w + pad)
            
            crop_rgb = image_rgb[y1_c:y2_c, x1_c:x2_c]
            crop_gray = gray_full[y1_c:y2_c, x1_c:x2_c]
            dark_mask_crop = (crop_gray < 60).astype(np.uint8)

            # Stage 2: Pore Detection (Class 1)
            stage2_results = model.predict(crop_rgb, conf=0.05, iou=0.8, classes=1, verbose=False)

            if stage2_results[0].masks is not None:
                c_h, c_w = crop_rgb.shape[:2]
                # Get boxes and confidences to match requested format
                boxes = stage2_results[0].boxes
                
                for i, mask_tensor in enumerate(stage2_results[0].masks.data):
                    m_np = cv2.resize(mask_tensor.cpu().numpy(), (c_w, c_h), interpolation=cv2.INTER_NEAREST)
                    m_bin = (m_np > 0.5).astype(np.uint8)
                    
                    refined = cv2.bitwise_and(m_bin, dark_mask_crop)
                    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for cnt in contours:
                        area_px = cv2.contourArea(cnt)
                        if area_px < 2: continue 
                        
                        d_mm = np.sqrt(4 * (area_px * (px_to_mm ** 2)) / np.pi)
                        
                        # Grading for Color Visualization
                        # Apply Grading
                        if d_mm == 0:
                            label, grade_color = "B", (50, 255, 50)  # Perfect
                        elif d_mm <= lim_B and lim_B > 0:
                            label, grade_color = "B", (50, 255, 50)
                        elif d_mm <= lim_C and lim_C > 0:
                            label, grade_color = "C", (255, 255, 0)
                        elif d_mm <= lim_D:
                            label, grade_color = "D", (255, 165, 0)
                        else:
                            label, grade_color = "F", (255, 0, 0)
                        
                        # Map coordinates
                        cnt_mapped = cnt.copy()
                        cnt_mapped[:, :, 0] += x1_c
                        cnt_mapped[:, :, 1] += y1_c
                        bx, by, bw, bh = cv2.boundingRect(cnt_mapped)
                        
                        # Format entry to match user requirement
                        pore_entries.append({
                            "name": "Pore",
                            "class": 1,
                            "confidence": round(float(boxes.conf[i].cpu().numpy()), 5),
                            "box": {
                                "x1": float(bx),
                                "y1": float(by),
                                "x2": float(bx + bw),
                                "y2": float(by + bh)
                            },
                            "size": round(float(d_mm), 3),
                            "grade": label,
                            "_contour": cnt_mapped,     # Private key for viz
                            "_color": grade_color        # Private key for viz
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
    
    return final_output, pore_entries

if __name__ == '__main__':
    results = porosity_check(
        image_path="../../data/welding-defect.jpeg", 
        model_path="../../models/best.pt", 
        px_to_mm=0.105
    )
    print(json.dumps(results, indent=2))