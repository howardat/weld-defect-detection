from pathlib import Path
import cv2
import numpy as np
import json
from ultralytics import YOLO

def cracks_check(image_path: str, model_path: str):
    # Load model and original image
    model = YOLO(model_path)
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]
    
    # --- STAGE 1: Detection of weld/candidate areas (Class 3) ---
    stage1_results = model.predict(image_rgb, conf=0.25, iou=0.5, classes=3, verbose=False)
    
    all_crack_detections = [] # For JSON logging
    crack_masks_list = []     # For Visualization
    crack_boxes_list = []     # For Visualization
    
    pad = 30 

    if stage1_results[0].boxes is not None:
        for box in stage1_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            x1_p, y1_p = max(0, x1 - pad), max(0, y1 - pad)
            x2_p, y2_p = min(w, x2 + pad), min(h, y2 + pad)
            
            crop = image_rgb[y1_p:y2_p, x1_p:x2_p]
            
            # --- STAGE 2: Detection + Segmentation of cracks (Class 0) ---
            stage2_results = model.predict(crop, conf=0.05, iou=0.2, classes=0, verbose=False)
            result_crop = stage2_results[0]
            
            if result_crop.boxes is not None:
                for i in range(len(result_crop.boxes)):
                    box_data = result_crop.boxes[i]
                    
                    # Global Bounding Box
                    gx1, gy1 = float(box_data.xyxy[0][0]) + x1_p, float(box_data.xyxy[0][1]) + y1_p
                    gx2, gy2 = float(box_data.xyxy[0][2]) + x1_p, float(box_data.xyxy[0][3]) + y1_p
                    
                    current_box = [int(gx1), int(gy1), int(gx2), int(gy2)]
                    crack_boxes_list.append(current_box)

                    det = {
                        "name": "crack",
                        "confidence": float(box_data.conf[0]),
                        "box": {"x1": gx1, "y1": gy1, "x2": gx2, "y2": gy2}
                    }

                    # --- MASK DATA ---
                    if result_crop.masks is not None:
                        mask_coords = result_crop.masks.xyn[i] 
                        crop_h, crop_w = crop.shape[:2]
                        
                        abs_mask = []
                        for pt in mask_coords:
                            abs_x = (pt[0] * crop_w) + x1_p
                            abs_y = (pt[1] * crop_h) + y1_p
                            abs_mask.append([int(abs_x), int(abs_y)])
                        
                        det["segmentation"] = abs_mask
                        crack_masks_list.append(abs_mask)

                    all_crack_detections.append(det)

    # Save to JSON
    out_path = Path("../../data/json_output/test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_crack_detections, f, indent=4)

    # RETURN THREE VALUES to avoid unpacking errors in main.py
    return crack_boxes_list, crack_masks_list, all_crack_detections, stage1_results[0].masks.xy

if __name__ == '__main__':
    boxes, masks, raw_json = cracks_check(
        image_path="../../data/crack.jpeg", 
        model_path="../../models/best.pt"
    )
    print(f"Total cracks found: {len(boxes)}")  