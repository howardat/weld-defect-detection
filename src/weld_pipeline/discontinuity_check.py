from pathlib import Path
from skimage.morphology import skeletonize
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
import torch
from matplotlib.colors import hsv_to_rgb
from weld_pipeline import timing

def discontinuity_check(image_path: str,
                        model_path: str,
                        threshold: float = 0.90,
                        otsu_multiplier: float = 1.0,
                        visualize: bool = False, seg_model=None) -> bool:
    """
    Checks for weld discontinuities using PIL for RGB consistency.
    """
    
    pad = 20
    gap = 5
    min_area_ratio = 0.03
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
    _img_name = Path(image_path).name
    # YOLO handles RGB numpy arrays correctly
    with timing.track(_img_name, "discontinuity_check", "yolo_stage1"):
        results = model.predict(image_bgr, conf=0.1, classes=3, agnostic_nms=True, verbose=False, save=False)

    if results[0].masks is None or len(results[0].masks.data) == 0:
        print("No detections in Stage 1.")
        return False, [], []

    orig_h, orig_w = results[0].orig_shape[:2]

    # =============================
    # INTERNAL UTILITIES
    # =============================
    def refine_mask_internal(original_rgb, yolo_masks_data, min_area_ratio):
        h, w = original_rgb.shape[:2]
        min_segment_area = int(h * w * min_area_ratio)
        
        # CRITICAL: Since we use PIL, we must use RGB2GRAY
        gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

        final_masks = []
        unfiltered_masks = []

        for mask_data in yolo_masks_data:
            if isinstance(mask_data, torch.Tensor):
                mask_data = mask_data.cpu().numpy()

            mask_big = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_big = (mask_big > 0.5).astype(np.uint8)

            # Otsu threshold computed only from weld-region pixels
            weld_pixels = gray[mask_big > 0]
            if len(weld_pixels) > 0:
                thresh_val, _ = cv2.threshold(weld_pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                non_dark = (gray >= thresh_val * otsu_multiplier).astype(np.uint8)
            else:
                _, non_dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

    def compute_line_params_internal(masks):
        params = []
        for mask in masks:
            skeleton = skeletonize(mask.astype(bool))
            coords = np.argwhere(skeleton).astype(float)
            if len(coords) >= gap * 2:
                centroid = coords.mean(axis=0)
                _, _, vt = np.linalg.svd(coords - centroid, full_matrices=False)
                direction = vt[0]
                angle = np.arctan2(direction[0], direction[1])
                if angle < 0:
                    angle += np.pi
                params.append({'angle': angle, 'centroid': centroid, 'direction': direction})
            else:
                params.append({'angle': None, 'centroid': None, 'direction': None})
        return params

    def check_alignment_internal(line_params, label=""):
        found = False
        img_diag = np.hypot(orig_h, orig_w)
        for i in range(len(line_params)):
            for j in range(i + 1, len(line_params)):
                lp_i, lp_j = line_params[i], line_params[j]
                if lp_i['angle'] is None or lp_j['angle'] is None:
                    print(f"{label}Mask {i} vs Mask {j} | Similarity: N/A (Insufficient data)")
                    continue
                angle_diff = min(abs(lp_i['angle'] - lp_j['angle']),
                                 np.pi - abs(lp_i['angle'] - lp_j['angle']))
                angle_sim = 1.0 - (2 * angle_diff / np.pi)
                diff = lp_j['centroid'] - lp_i['centroid']
                perp_dist = abs(float(np.cross(lp_i['direction'], diff)))
                pos_sim = max(0.0, 1.0 - (perp_dist / (img_diag * 0.15)))
                similarity = 0.6 * angle_sim + 0.6 * pos_sim
                print(f"{label}Mask {i} vs Mask {j} | angle_sim={angle_sim:.3f}  pos_sim={pos_sim:.3f}  combined={similarity:.4f}")
                if similarity >= threshold:
                    found = True
        return found

    # Early alignment check on raw Stage 1 masks — before any filtering.
    # If YOLO already splits the broken weld into 2+ separate detections, catch it here.
    found_discontinuity_early = False
    if len(raw_masks) >= 2:
        print("\n--- Stage 1 Raw Mask Alignment Check (pre-filter) ---")
        found_discontinuity_early = check_alignment_internal(
            compute_line_params_internal(raw_masks), label="[S1] "
        )

    all_masks = []
    all_unfiltered_full = [] # New list to store full-sized unfiltered masks

    for _mask_idx, mask_binary in enumerate(raw_masks):
        bbox = get_bbox_internal(mask_binary, padding=pad)
        if bbox is None: continue
        x1, y1, x2, y2 = bbox

        crop_image_bgr = image_bgr[y1:y2, x1:x2]
        crop_h, crop_w = crop_image_bgr.shape[:2]  # THE TARGET SIZE

        with timing.track(_img_name, "discontinuity_check", "yolo_stage2", _mask_idx):
            crop_results = model.predict(crop_image_bgr, conf=0.01, iou=0.5, classes=3, agnostic_nms=False, verbose=False, save=False)
        
        if crop_results[0].masks is not None:
            # 1. Get refined and unfiltered segments from the crop
            refined, unfiltered = refine_mask_internal(crop_image_bgr, crop_results[0].masks.data, min_area_ratio)

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

            # If every refined segment was filtered out by min_area_ratio, fall back to the
            # Stage 1 raw mask so this detection still participates in alignment comparison.
            if not refined:
                all_masks.append(mask_binary)
        else:
            # Stage 2 detected nothing — fall back to the Stage 1 raw mask so this detection
            # is not silently dropped from the alignment comparison.
            all_masks.append(mask_binary)

    # Replace the local unfiltered_masks with our full-sized collection for visualization
    unfiltered_masks = all_unfiltered_full 
    all_masks = deduplicate_internal(all_masks)

    # =============================
    # 3. LINEAR FUNCTION ANALYSIS
    # =============================
    line_params = compute_line_params_internal(all_masks)

    found_discontinuity = found_discontinuity_early

    print("\n--- Refined Mask Alignment Check (Angle + Perpendicular Distance) ---")
    found_discontinuity = found_discontinuity or check_alignment_internal(line_params)

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

        # --- STAGE 3: Efficient Overlay ---
        # Create ONE overlay for ALL masks in this stage
        stage3_overlay = np.zeros((orig_h, orig_w, 4), dtype=np.float32) 

        for i, m in enumerate(unfiltered_masks):
            color = colors_unfiltered[i % len(colors_unfiltered)]
            # Apply color only to the pixels where this mask exists
            stage3_overlay[m > 0] = list(color) + [0.4]
            add_thin_box(axes[2], m, color)

        axes[2].imshow(image_rgb)
        axes[2].imshow(stage3_overlay) # Only one heavy imshow call
        del stage3_overlay # Clean up immediately
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
            if lp['angle'] is not None:
                color = colors_filtered[i % len(colors_filtered)]
                cy, cx = lp['centroid']
                dy, dx = lp['direction']
                t = float(max(orig_w, orig_h))
                x_vals = [cx - t * dx, cx + t * dx]
                y_vals = [cy - t * dy, cy + t * dy]
                axes[5].plot(x_vals, y_vals, color=color, lw=1.5)
        axes[5].set_title("6: Final Fit")
        
        for ax in axes: ax.axis('off')
        plt.tight_layout(); 
        plt.show()

        # CRITICAL: Release memory immediately after showing
        plt.close(fig) 
        plt.close('all') # Force close everything just to be safe

    return found_discontinuity, all_masks, line_params

if __name__ == '__main__':
    result = discontinuity_check(
        image_path="../../data/zoom2.jpg", 
        model_path="../../models/best.pt", 
        threshold=0.99, 
        visualize=True
    )
    print(f"\nDiscontinuity Found: {result[0]}")