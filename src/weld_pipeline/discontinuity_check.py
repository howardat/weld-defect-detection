from skimage.morphology import skeletonize
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
import torch
from matplotlib.colors import hsv_to_rgb

def discontinuity_check(image_path: str, 
                        model_path: str, 
                        threshold: float = 0.90, 
                        visualize: bool = False) -> bool:
    """
    Checks for weld discontinuities using PIL for RGB consistency.
    """
    
    pad = 20
    gap = 5
    min_area_ratio = 0.05
    p_intensity = 50 # Adjust this if needed for your specific weld brightness

    # =============================
    # 1. LOAD IMAGE & MODEL (PIL)
    # =============================
    # PIL loads as RGB by default, which YOLO expects.
    pil_img = Image.open(image_path).convert("RGB")
    image_rgb = np.array(pil_img)
    
    model = YOLO(model_path)
    # YOLO handles RGB numpy arrays correctly
    results = model.predict(image_rgb, conf=0.05, classes=3, verbose=True, save=True)

    if results[0].masks is None or len(results[0].masks.data) == 0:
        print("No detections in Stage 1.")
        return False, [], []

    orig_h, orig_w = results[0].orig_shape[:2]

    # =============================
    # INTERNAL UTILITIES
    # =============================
    def refine_mask_internal(original_rgb, yolo_masks_data, min_area_ratio, p_int):
        h, w = original_rgb.shape[:2]
        min_segment_area = int(h * w * min_area_ratio)
        
        # CRITICAL: Since we use PIL, we must use RGB2GRAY
        gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
        
        # Pixel intensity filtering
        # We keep pixels that are NOT dark (>= p_int)
        non_dark = (gray >= p_int).astype(np.uint8)
        
        final_masks = []
        for mask_data in yolo_masks_data:
            if isinstance(mask_data, torch.Tensor): 
                mask_data = mask_data.cpu().numpy()
            
            mask_big = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_big = (mask_big > 0.5).astype(np.uint8)
            
            # Combine YOLO's shape prediction with intensity filtering
            weld_clean = cv2.bitwise_and(mask_big, non_dark)
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weld_clean, 8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_segment_area:
                    final_masks.append((labels == i).astype(np.uint8))
        return final_masks

    def get_bbox_internal(mask, padding=20):
        coords = np.argwhere(mask > 0)
        if len(coords) == 0: return None
        y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
        return max(0, x_min-padding), max(0, y_min-padding), min(orig_w, x_max+padding), min(orig_h, y_max+padding)

    def deduplicate_internal(masks, iou_threshold=0.1):
        if not masks: return []
        areas = [m.sum() for m in masks]
        order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
        kept = []
        for idx in order:
            is_dup = False
            for k in kept:
                inter = np.logical_and(masks[idx], k).sum()
                union = np.logical_or(masks[idx], k).sum()
                if union > 0 and (inter/union) > iou_threshold: 
                    is_dup = True; break
            if not is_dup: kept.append(masks[idx])
        return kept

    # =============================
    # 2. CROP & REFINEMENT
    # =============================
    raw_masks = []
    for mask_data in results[0].masks.data:
        mask_np = mask_data.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        raw_masks.append((mask_resized >= 0.5).astype(np.uint8))

    all_masks = []
    for mask_binary in raw_masks:
        bbox = get_bbox_internal(mask_binary, padding=pad)
        if bbox is None: continue
        x1, y1, x2, y2 = bbox
        
        # Crop from the RGB image
        crop_image_rgb = image_rgb[y1:y2, x1:x2]
        crop_results = model.predict(crop_image_rgb, conf=0.05, iou=0.5, classes=3, agnostic_nms=True, verbose=False, save=True)
        
        if crop_results[0].masks is not None:
            refined = refine_mask_internal(crop_image_rgb, crop_results[0].masks.data, min_area_ratio, p_intensity)
            for m in refined:
                full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = m
                all_masks.append(full_mask)

    all_masks = deduplicate_internal(all_masks)

    # =============================
    # 3. LINEAR FUNCTION ANALYSIS
    # =============================
    line_params = []
    for i, mask in enumerate(all_masks):
        skeleton = skeletonize(mask.astype(bool))
        coords = np.argwhere(skeleton)
        if len(coords) >= gap * 2:
            coords = coords[np.argsort(coords[:, 1])] # Sort by X
            y_pts = coords[gap:-gap, 0]
            x_pts = coords[gap:-gap, 1]
            if len(x_pts) > 2:
                m, b = np.polyfit(x_pts, y_pts, 1)
                line_params.append({'m': m, 'b': b, 'index': i})
                continue
        line_params.append({'m': None, 'b': None, 'index': i})

    found_discontinuity = False
    if len(line_params) < 2:
        return False, [], []
    
    print("\n--- Linear Function Similarity Comparisons (Distance-Based) ---")
    for i in range(len(line_params)):
        for j in range(i + 1, len(line_params)):
            # Extract slope (m) and intercept (b)
            m1, b1 = line_params[i]['m'], line_params[i]['b']
            m2, b2 = line_params[j]['m'], line_params[j]['b']
            
            if None not in [m1, m2, b1, b2]:
                # Define the parameter vectors
                v1 = np.array([m1, b1])
                v2 = np.array([m2, b2])
                
                # 1. Calculate the Euclidean distance between the two functions
                distance = np.linalg.norm(v1 - v2)
                
                # 2. Calculate the sum of the magnitudes (for normalization)
                mag_sum = np.linalg.norm(v1) + np.linalg.norm(v2)
                
                # 3. Calculate Normalized Similarity
                # If both lines are at the origin (0,0), mag_sum is 0. 
                if mag_sum == 0:
                    similarity = 1.0
                else:
                    similarity = 1.0 - (distance / mag_sum)
                
                print(f"Mask {i} vs Mask {j} | Similarity: {similarity:.4f}")
                
                if similarity >= threshold:
                    found_discontinuity = True
            else:
                print(f"Mask {i} vs Mask {j} | Similarity: N/A (Insufficient data)")

    # =============================
    # 4. VISUALIZATION
    # =============================
    if visualize:
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        num_detections = len(all_masks)
        colors = [hsv_to_rgb((h, 1, 1)) for h in np.linspace(0, 1, max(1, num_detections), endpoint=False)]

        axes[0].imshow(image_rgb); axes[0].set_title("Stage 1: Original (RGB)")
        
        # Raw YOLO overlays
        axes[1].imshow(image_rgb)
        for m in raw_masks:
            mask_overlay = np.zeros((orig_h, orig_w, 4))
            mask_overlay[m > 0] = [0, 1, 0, 0.4]
            axes[1].imshow(mask_overlay)
        axes[1].set_title("Stage 2: Raw YOLO")

        # Refined Overlays
        axes[2].imshow(image_rgb)
        for i, m in enumerate(all_masks):
            color = list(colors[i]) + [0.5]
            mask_overlay = np.zeros((orig_h, orig_w, 4))
            mask_overlay[m > 0] = color
            axes[2].imshow(mask_overlay)
        axes[2].set_title("Stage 3: Refined")

        # Skeletons
        skel_img = np.zeros((orig_h, orig_w, 3))
        for i, m in enumerate(all_masks):
            skel = skeletonize(m.astype(bool))
            skel_img[skel] = colors[i]
        axes[3].imshow(skel_img); axes[3].set_title("Stage 4: Skeletons")

        # Final
        axes[4].imshow(image_rgb)
        for i, lp in enumerate(line_params):
            if lp['m'] is not None:
                x_vals = np.array([0, orig_w])
                y_vals = lp['m'] * x_vals + lp['b']
                axes[4].plot(x_vals, y_vals, color=colors[i], lw=2)
        axes[4].set_title("Stage 5: Final Fit")
        
        for ax in axes: ax.axis('off')
        plt.tight_layout(); plt.show()

    return found_discontinuity, all_masks, line_params

if __name__ == '__main__':
    result = discontinuity_check(
        image_path="../../data/zoom2.jpg", 
        model_path="../../models/best.pt", 
        threshold=0.99, 
        visualize=True
    )
    print(f"\nDiscontinuity Found: {result[0]}")