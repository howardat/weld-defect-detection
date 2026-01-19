import cv2
import numpy as np
import json
from ultralytics import YOLO
from PIL import Image

def cracks_check(image_path: str, model_path: str):
    # Load model and original image
    model = YOLO(model_path)
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]
    
    # --- STAGE 1: Detection of weld/candidate areas ---
    # We look for the main object (e.g., class 3) to create crops
    stage1_results = model.predict(image_rgb, conf=0.25, iou=0.5, classes=3, verbose=False)
    
    all_crack_detections = []
    pad = 30 # Padding to ensure we don't cut off the edges of the weld

    if stage1_results[0].boxes is not None:
        for box in stage1_results[0].boxes:
            # Get coordinates for the crop
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Apply padding and handle boundaries
            x1_p, y1_p = max(0, x1 - pad), max(0, y1 - pad)
            x2_p, y2_p = min(w, x2 + pad), min(h, y2 + pad)
            
            # Create the crop
            crop = image_rgb[y1_p:y2_p, x1_p:x2_p]
            
            # --- STAGE 2: Detection of cracks inside the crop ---
            # Run prediction on the zoomed-in crop
            stage2_results = model.predict(crop, conf=0.05, iou=0.2, classes=0, verbose=False, save=True)
            
            # Parse crop detections back to original image coordinates
            if stage2_results[0].boxes is not None:
                crop_data = json.loads(stage2_results[0].to_json())
                
                for det in crop_data:
                    # Adjust box coordinates from crop-space to original-image-space
                    det['box']['x1'] += x1_p
                    det['box']['y1'] += y1_p
                    det['box']['x2'] += x1_p
                    det['box']['y2'] += y1_p
                    
                    # Remove heavy segment data as requested
                    det.pop("segments", None)
                    all_crack_detections.append(det)

    # Save aggregated results to JSON
    with open('../../data/json_output/test.json', 'w') as f:
        json.dump(all_crack_detections, f, indent=4)

    return all_crack_detections

if __name__ == '__main__':
    results = cracks_check(
        image_path="../../data/crack.jpeg", 
        model_path="../../models/best.pt"
    )
    print(f"Total cracks found in crops: {len(results)}")