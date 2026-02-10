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
                        visualize: bool = False, seg_model=None) -> bool:
    """
    Checks for weld discontinuities using PIL for RGB consistency.
    """
    
    pad = 20
    gap = 5
    min_area_ratio = 0.05
    p_intensity = 50 # Adjust this if needed for your specific weld brightness
    unfiltered_masks = []

    # =============================
    # 1. LOAD IMAGE & MODEL (PIL)
    # =============================
    # PIL loads as RGB by default, which YOLO expects.
    pil_img = Image.open(image_path).convert("RGB")
    image_rgb = np.array(pil_img)

    # 2. Create a BGR copy SPECIFICALLY for YOLO's internal saving
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    model = seg_model
    # YOLO handles RGB numpy arrays correctly
    results = model.predict(image_bgr, conf=0.01, classes=3, agnostic_nms=True, verbose=False, save=False)

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
        unfiltered_masks = []

        for mask_data in yolo_masks_data:
            if isinstance(mask_data, torch.Tensor): 
                mask_data = mask_data.cpu().numpy()
            
            mask_big = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_big = (mask_big > 0.5).astype(np.uint8)
            
            # Combine YOLO's shape prediction with intensity filtering
            weld_clean = cv2.bitwise_and(mask_big, non_dark)
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weld_clean, 8)
            for i in range(1, num_labels):
                segment = (labels == i).astype(np.uint8)
                unfiltered_masks.append(segment)

                if stats[i, cv2.CC_STAT_AREA] >= min_segment_area:
                    final_masks.append((labels == i).astype(np.uint8))
        return final_masks, unfiltered_masks

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
    all_unfiltered_full = [] # New list to store full-sized unfiltered masks

    for mask_binary in raw_masks:
        bbox = get_bbox_internal(mask_binary, padding=pad)
        if bbox is None: continue
        x1, y1, x2, y2 = bbox
        
        crop_image_bgr = image_bgr[y1:y2, x1:x2]
        crop_h, crop_w = crop_image_bgr.shape[:2] # THE TARGET SIZE

        crop_results = model.predict(crop_image_bgr, conf=0.01, iou=0.5, classes=3, agnostic_nms=False, verbose=False, save=False)
        
        if crop_results[0].masks is not None:
            # 1. Get refined and unfiltered segments from the crop
            refined, unfiltered = refine_mask_internal(crop_image_bgr, crop_results[0].masks.data, min_area_ratio, p_intensity)
            
            # 2. Resize and place UNFILTERED masks
            for m_unf in unfiltered:
                m_unf_fixed = cv2.resize(m_unf, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                full_unf = np.zeros((orig_h, orig_w), dtype=np.uint8)
                full_unf[y1:y2, x1:x2] = m_unf_fixed
                all_unfiltered_full.append(full_unf)

            # 3. Resize and place REFINED (filtered) masks
            for m in refined:
                m_fixed = cv2.resize(m, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = m_fixed
                all_masks.append(full_mask)

    # Replace the local unfiltered_masks with our full-sized collection for visualization
    unfiltered_masks = all_unfiltered_full 
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
        import matplotlib.patches as patches
        fig, axes = plt.subplots(1, 6, figsize=(30, 5))
        
        # Color generators for different stages
        # We need a color list for Stage 3 (unfiltered) and Stage 4+ (filtered)
        colors_unfiltered = [hsv_to_rgb((h, 1, 1)) for h in np.linspace(0, 1, max(1, len(unfiltered_masks)), endpoint=False)]
        colors_filtered = [hsv_to_rgb((h, 0.8, 1)) for h in np.linspace(0.5, 1, max(1, len(all_masks)), endpoint=False)]

        def add_thin_box(ax, mask, color):
            """Calculates and draws a thin bounding box around a mask."""
            coords = np.argwhere(mask > 0)
            if len(coords) == 0: return
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            # Patch uses (x, y, width, height)
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                     linewidth=0.8, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        # Stage 1 & 2 remain as you had them
        axes[0].imshow(image_rgb); axes[0].set_title("1: Original")
        
        axes[1].imshow(image_rgb)
        for m in raw_masks:
            mask_overlay = np.zeros((orig_h, orig_w, 4))
            mask_overlay[m > 0] = [0, 1, 0, 0.4]
            axes[1].imshow(mask_overlay)
        axes[1].set_title("2: Raw YOLO")

        # --- STAGE 3: Unfiltered Refined (Each chunk unique color + Box) ---
        axes[2].imshow(image_rgb)
        for i, m in enumerate(unfiltered_masks):
            color = colors_unfiltered[i % len(colors_unfiltered)]
            mask_overlay = np.zeros((orig_h, orig_w, 4))
            mask_overlay[m > 0] = list(color) + [0.4]
            axes[2].imshow(mask_overlay)
            add_thin_box(axes[2], m, color)
        axes[2].set_title("3: Unfiltered Refined")
        # --- SAVE STAGE 3 SPECIFICALLY ---
        # We use the extent of the 3rd axis (index 2) to save just that portion
        extent = axes[2].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig("stage_3_unfiltered.jpg", bbox_inches=extent.expanded(1.1, 1.1), dpi=300)
        print("Stage 3 saved as stage_3_unfiltered.jpg")

        # --- STAGE 4: Filtered Refined (Boxed) ---
        axes[3].imshow(image_rgb)
        for i, m in enumerate(all_masks):
            color = colors_filtered[i % len(colors_filtered)]
            mask_overlay = np.zeros((orig_h, orig_w, 4))
            mask_overlay[m > 0] = list(color) + [0.5]
            axes[3].imshow(mask_overlay)
            add_thin_box(axes[3], m, color)
        axes[3].set_title("4: Filtered Refined")

        # --- STAGE 5: Skeletons (Boxed) ---
        skel_img = np.zeros((orig_h, orig_w, 3))
        for i, m in enumerate(all_masks):
            color = colors_filtered[i % len(colors_filtered)]
            skel = skeletonize(m.astype(bool))
            skel_img[skel] = color
            # Draw skeletons on black, but add the box for reference
            add_thin_box(axes[4], m, color)
        axes[4].imshow(skel_img); axes[4].set_title("5: Skeletons")

        # --- STAGE 6: Final Fit (Boxed) ---
        axes[5].imshow(image_rgb)
        for i, m in enumerate(all_masks):
            color = colors_filtered[i % len(colors_filtered)]
            add_thin_box(axes[5], m, color)
            
        for i, lp in enumerate(line_params):
            if lp['m'] is not None:
                color = colors_filtered[i % len(colors_filtered)]
                x_vals = np.array([0, orig_w])
                y_vals = lp['m'] * x_vals + lp['b']
                axes[5].plot(x_vals, y_vals, color=color, lw=1.5)
        axes[5].set_title("6: Final Fit")
        
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