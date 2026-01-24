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
                        threshold: float = 0.98, 
                        visualize: bool = False) -> bool:
    """
    Checks for weld discontinuities by comparing linear function similarities.
    Returns: True if any pair similarity >= threshold, False otherwise.
    """
    
    pad = 20
    gap = 5
    min_area_ratio = 0.05

    # =============================
    # LOAD IMAGE & MODEL
    # =============================
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    model = YOLO(model_path)
    results = model.predict(image_rgb, conf=0.05, classes=3, agnostic_nms=True, save=True, verbose=False)

    if results[0].masks is None or len(results[0].masks.data) == 0:
        print("No detections in Stage 1.")
        return False

    orig_h, orig_w = results[0].orig_shape[:2]

    # =============================
    # STAGE 2: RAW YOLO MASKS
    # =============================
    raw_masks = []
    for mask_data in results[0].masks.data:
        mask_np = mask_data.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_resized = np.clip(mask_resized, 0.0, 1.0)
        mask_binary = (mask_resized >= 0.9).astype(np.uint8)
        raw_masks.append(mask_binary)

    # =============================
    # INTERNAL UTILITIES
    # =============================
    def refine_mask_internal(original_image, yolo_masks_data, min_area_ratio, p_intensity=50):
        h, w = original_image.shape[:2]
        min_segment_area = int(h * w * min_area_ratio)
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        dark = (gray < p_intensity).astype(np.uint8)
        non_dark = 1 - dark
        final_masks = []
        for mask_data in yolo_masks_data:
            if isinstance(mask_data, torch.Tensor): mask_data = mask_data.cpu().numpy()
            mask_big = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_big = (mask_big > 0.5).astype(np.uint8)
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
        h, w = mask.shape
        return max(0, x_min-padding), max(0, y_min-padding), min(w, x_max+padding), min(h, y_max+padding)

    def deduplicate_internal(masks, iou_threshold=0.1):
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
    # CROP & REFINEMENT
    # =============================
    all_masks = []
    for mask_binary in raw_masks:
        bbox = get_bbox_internal(mask_binary, padding=pad)
        if bbox is None: continue
        x1, y1, x2, y2 = bbox
        crop_image = image_rgb[y1:y2, x1:x2]
        crop_results = model.predict(crop_image, conf=0.05, iou=0.5, classes=3, agnostic_nms=True, verbose=False)
        if crop_results[0].masks is None: continue
        refined = refine_mask_internal(crop_image, crop_results[0].masks.data, min_area_ratio)
        for m in refined:
            full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = m
            all_masks.append(full_mask)

    all_masks = deduplicate_internal(all_masks)

    # =============================
    # LINEAR FUNCTION ANALYSIS
    # =============================
    line_params = []
    for i, mask in enumerate(all_masks):
        skeleton = skeletonize(mask.astype(bool))
        coords = np.argwhere(skeleton)
        if len(coords) >= gap * 2:
            # Sort by X for fitting
            coords = coords[np.argsort(coords[:, 1])]
            y_pts = coords[gap:, 0]
            x_pts = coords[:len(coords) - gap, 1]
            m, b = np.polyfit(x_pts, y_pts, 1)
            line_params.append({'m': m, 'b': b, 'index': i})
        else:
            line_params.append({'m': None, 'b': None, 'index': i})

    # Linear Comparisons & Scoring
    found_discontinuity = False
    if len(line_params) < 2:
        return False, [], []
    
    print("\n--- Linear Function Similarity Comparisons ---")
    for i in range(len(line_params)):
        for j in range(i + 1, len(line_params)):
            m1, m2 = line_params[i]['m'], line_params[j]['m']
            if m1 is not None and m2 is not None:
                # Cosine Similarity of [1, m] vectors
                v1, v2 = np.array([1, m1]), np.array([1, m2])
                similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                print(f"Mask {i} vs Mask {j} | Similarity: {similarity:.4f}")
                
                if similarity >= threshold:
                    found_discontinuity = True
            else:
                print(f"Mask {i} vs Mask {j} | Similarity: N/A (Insufficient data)")

    # =============================
    # FIVE STAGE VISUALIZATION
    # =============================
    if visualize:
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        num_detections = len(all_masks)
        hues = np.linspace(0, 1, num_detections, endpoint=False)
        colors = [hsv_to_rgb((h, 1, 1)) for h in hues]

        # Stage 1: Original
        axes[0].imshow(image_rgb); axes[0].axis('off'); axes[0].set_title("Stage 1: Original")

        # Stage 2: Raw YOLO
        axes[1].imshow(image_rgb)
        for mask in raw_masks:
            mask_rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
            mask_rgba[mask > 0] = [0, 255, 0, 120]
            axes[1].imshow(mask_rgba)
        axes[1].axis('off'); axes[1].set_title("Stage 2: Raw YOLO")

        # Stage 3: Refined
        axes[2].imshow(image_rgb)
        for i, mask in enumerate(all_masks):
            rgb_255 = (np.array(colors[i]) * 255).astype(np.uint8)
            mask_rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
            mask_rgba[mask > 0] = [rgb_255[0], rgb_255[1], rgb_255[2], 100]
            axes[2].imshow(mask_rgba)
        axes[2].axis('off'); axes[2].set_title("Stage 3: Refined")

        # Stage 4: Skeletons
        skeleton_composite = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for i, mask in enumerate(all_masks):
            skel = skeletonize(mask.astype(bool))
            thick = cv2.dilate((skel * 255).astype(np.uint8), np.ones((3,3)), iterations=2)
            rgb_255 = (np.array(colors[i]) * 255).astype(np.uint8)
            for c in range(3): skeleton_composite[:,:,c][thick > 0] = rgb_255[c]
        axes[3].imshow(skeleton_composite); axes[3].axis('off'); axes[3].set_title("Stage 4: Skeletons")

        # Stage 5: Final Analysis (Masks + Lines)
        axes[4].imshow(image_rgb)
        for i, mask in enumerate(all_masks):
            # Overlay refined mask
            rgb_255 = (np.array(colors[i]) * 255).astype(np.uint8)
            mask_rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
            mask_rgba[mask > 0] = [rgb_255[0], rgb_255[1], rgb_255[2], 100]
            axes[4].imshow(mask_rgba)
            
            # Overlay linear fit
            lp = line_params[i]
            if lp['m'] is not None:
                x_fit = np.array([0, orig_w])
                y_fit = lp['m'] * x_fit + lp['b']
                axes[4].plot(x_fit, y_fit, color=colors[i], linewidth=1.5, alpha=1.0)
                
        axes[4].axis('off'); axes[4].set_title("Stage 5: Final Analysis")
        
        plt.tight_layout()
        plt.show()

    return found_discontinuity, all_masks, line_params

if __name__ == '__main__':
    # Usage
    has_discontinuity = discontinuity_check(
        image_path="../../data/zoom2.jpg", 
        model_path="../../models/best.pt", 
        threshold=0.99, 
        visualize=True
    )
    print(f"\nFinal Result -> Discontinuity Detected: {has_discontinuity}")