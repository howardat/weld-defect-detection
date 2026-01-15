import numpy as np
from PIL import Image
import cv2

# Import from your internal package
from weld_pipeline.detector import WeldDetector, refine_mask
from weld_pipeline.utils import get_bounding_box_from_mask, deduplicate_masks
from weld_pipeline.visualizer import plot_weld_results

def run_pipeline(image_path, model_path, pad=20, min_area_ratio=0.05):
    # 1. Initialization
    image = np.array(Image.open(image_path))
    detector = WeldDetector(model_path)
    
    # 2. Stage 1: Initial Detection
    # Note: We use detector.model directly for the first pass
    results = detector.model.predict(image_path, conf=0.05, iou=0.5, 
                                    classes=3, agnostic_nms=True, verbose=False)

    if not results[0].masks or len(results[0].masks.data) == 0:
        print("No detections in Stage 1. Exiting.")
        return

    orig_h, orig_w = results[0].orig_shape[:2]
    all_masks = []

    # 3. Stage 2: Crop, Predict, and Refine
    for mask_data in results[0].masks.data:
        mask_np = mask_data.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        bbox = get_bounding_box_from_mask(mask_binary, padding=pad)
        if bbox is None:
            continue
            
        x1, y1, x2, y2 = bbox
        crop_image = image[y1:y2, x1:x2]
        
        # Predict on crop
        crop_results = detector.model.predict(crop_image, conf=0.05, iou=0.5, 
                                             classes=3, agnostic_nms=True, verbose=False)
        
        if not crop_results[0].masks or len(crop_results[0].masks.data) == 0:
            continue
            
        # Refine masks within the crop
        refined_masks_crop = refine_mask(crop_image, crop_results[0].masks.data, 
                                         min_area_ratio=min_area_ratio)
        
        # Place refined masks back into the full-sized image
        for m in refined_masks_crop:
            full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = m
            all_masks.append(full_mask)

    # 4. Cleanup
    final_masks = deduplicate_masks(all_masks, iou_threshold=0.1)
    print(f"Final weld count: {len(final_masks)}")

    # 5. Output
    plot_weld_results(image, final_masks)

if __name__ == "__main__":
    # You can move these to a CLI parser later
    IMG_IN = "../../data/zoom.jpg"
    MODEL_IN = "../../models/best.pt"
    
    run_pipeline(IMG_IN, MODEL_IN)