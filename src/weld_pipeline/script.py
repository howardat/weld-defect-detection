from skimage.morphology import skeletonize
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
import torch
from typing import List, Union
from matplotlib.colors import hsv_to_rgb

# =============================
# MASK REFINEMENT FUNCTION
# =============================
def refine_mask(original_image: np.ndarray,
                yolo_masks_data: Union[List[torch.Tensor], List[np.ndarray]],
                min_segment_area: int = None,
                min_area_ratio: float = 0.05,
                p_intensity: int = 50) -> List[np.ndarray]:
    h, w = original_image.shape[:2]
    total_pixels = h * w
    if min_segment_area is None:
        min_segment_area = int(total_pixels * min_area_ratio)
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    dark = (gray < p_intensity).astype(np.uint8)
    non_dark = 1 - dark
    final_masks = []
    for mask_data in yolo_masks_data:
        if isinstance(mask_data, torch.Tensor):
            mask_data = mask_data.cpu().numpy()
        mask_big = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_big = (mask_big > 0.5).astype(np.uint8)
        weld_clean = cv2.bitwise_and(mask_big, non_dark)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weld_clean, 8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_segment_area:
                final_masks.append((labels == i).astype(np.uint8))
    return final_masks

# =============================
# UTILITY FUNCTIONS
# =============================
def get_bounding_box_from_mask(mask: np.ndarray, padding: int = 20):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    h, w = mask.shape
    return max(0, x_min-padding), max(0, y_min-padding), min(w, x_max+padding), min(h, y_max+padding)

def calculate_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return 0 if union==0 else inter/union

def deduplicate_masks(masks, iou_threshold=0.1):
    areas = [m.sum() for m in masks]
    order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
    kept = []
    for idx in order:
        if all(calculate_iou(masks[idx], k)<=iou_threshold for k in kept):
            kept.append(masks[idx])
    return kept

# =============================
# CONFIGURATION
# =============================
image_path = "../../data/zoom.jpg"
model_path = "../../models/best.pt"
pad = 20
gap = 5
min_area_ratio = 0.05

# =============================
# LOAD IMAGE & MODEL
# =============================
image = np.array(Image.open(image_path))
model = YOLO(model_path)
results = model.predict(image_path, conf=0.05, iou=0.5, classes=3, agnostic_nms=True, verbose=False)

if results[0].masks is None or len(results[0].masks.data)==0:
    print("No detections in Stage 1. Exiting.")
    exit()

orig_h, orig_w = results[0].orig_shape[:2]

# =============================
# STAGE 2: RAW YOLO MASKS
# =============================
raw_masks = []
for mask_data in results[0].masks.data:
    mask_np = mask_data.cpu().numpy()
    mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    mask_binary = (mask_resized > 0.5).astype(np.uint8)
    raw_masks.append(mask_binary)

# =============================
# CROP & REFINEMENT
# =============================
all_masks = []
for mask_data in results[0].masks.data:
    mask_np = mask_data.cpu().numpy()
    mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    mask_binary = (mask_resized > 0.5).astype(np.uint8)
    bbox = get_bounding_box_from_mask(mask_binary, padding=pad)
    if bbox is None:
        continue
    x1, y1, x2, y2 = bbox
    crop_image = image[y1:y2, x1:x2]
    crop_results = model.predict(crop_image, conf=0.05, iou=0.5, classes=3, agnostic_nms=True, verbose=False)
    if crop_results[0].masks is None or len(crop_results[0].masks.data)==0:
        continue
    refined_masks_crop = refine_mask(crop_image, crop_results[0].masks.data, min_segment_area=None, min_area_ratio=min_area_ratio)
    for m in refined_masks_crop:
        full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = m
        all_masks.append(full_mask)

# =============================
# DEDUPLICATE MASKS
# =============================
all_masks = deduplicate_masks(all_masks, iou_threshold=0.1)
print(f"Final weld count: {len(all_masks)}")

# =============================
# FIVE STAGE VISUALIZATION
# =============================
fig, axes = plt.subplots(1, 5, figsize=(25, 5))

# STAGE 1: Original Image
axes[0].imshow(image)
axes[0].axis('off')
axes[0].set_title("Stage 1: Original Image", fontsize=9, fontweight='medium')

# Generate distinct colors for each detection
num_detections = len(all_masks)
hues = np.linspace(0, 1, num_detections, endpoint=False)
colors = [hsv_to_rgb((h, 1, 1)) for h in hues]

# STAGE 2: Original + Raw YOLO Masks + Bounding Boxes
axes[1].imshow(image)
for mask in raw_masks:
    mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    mask_rgba[mask > 0] = [0, 255, 0, 120]  # Green, semi-transparent
    axes[1].imshow(mask_rgba)
    # Add bounding box
    bbox = get_bounding_box_from_mask(mask, padding=0)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, 
                         edgecolor='lime', facecolor='none')
        axes[1].add_patch(rect)
axes[1].axis('off')
axes[1].set_title(f"Stage 2: Raw YOLO Detection ({len(raw_masks)} masks)", fontsize=9, fontweight='medium')

# STAGE 3: Original + Refined Masks + Bounding Boxes
axes[2].imshow(image)
# Overlay refined masks with distinct colors
for i, mask in enumerate(all_masks):
    mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    # Convert RGB [0,1] to RGBA [0,255] with alpha channel
    rgb_255 = (np.array(colors[i]) * 255).astype(np.uint8)
    mask_rgba[mask > 0] = [rgb_255[0], rgb_255[1], rgb_255[2], 100]  # Distinct color, semi-transparent
    axes[2].imshow(mask_rgba)
    # Add bounding box
    bbox = get_bounding_box_from_mask(mask, padding=0)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, 
                         edgecolor=colors[i], facecolor='none')
        axes[2].add_patch(rect)
axes[2].axis('off')
axes[2].set_title(f"Stage 3: Refined Masks ({len(all_masks)} masks)", fontsize=9, fontweight='medium')

# STAGE 4: Skeletons on Black Background with distinct colors
skeleton_composite = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)  # RGB instead of grayscale
kernel = np.ones((3, 3), np.uint8)  # Larger kernel for thicker lines (try 3, 5, 7)
for i, mask in enumerate(all_masks):
    skeleton = skeletonize(mask.astype(bool))
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    # Dilate the skeleton to make it thicker
    skeleton_thick = cv2.dilate(skeleton_uint8, kernel, iterations=2)  # More iterations = thicker
    
    # Apply distinct color to this skeleton
    rgb_255 = (np.array(colors[i]) * 255).astype(np.uint8)
    for c in range(3):  # RGB channels
        channel = skeleton_composite[:, :, c]
        channel[skeleton_thick > 0] = rgb_255[c]
        skeleton_composite[:, :, c] = channel

axes[3].imshow(skeleton_composite)
axes[3].axis('off')
axes[3].set_title("Stage 4: Skeletons", fontsize=9, fontweight='medium')

# STAGE 5: Original + Refined Masks + Skeletons + Linear Functions
axes[4].imshow(image)

# Overlay refined masks with distinct colors
for i, mask in enumerate(all_masks):
    mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    # Convert RGB [0,1] to RGBA [0,255] with alpha channel
    rgb_255 = (np.array(colors[i]) * 255).astype(np.uint8)
    mask_rgba[mask > 0] = [rgb_255[0], rgb_255[1], rgb_255[2], 100]  # Distinct color, semi-transparent
    axes[4].imshow(mask_rgba)

# Plot skeletons and linear functions
for i, mask in enumerate(all_masks):
    skeleton = skeletonize(mask.astype(bool))
    # Plot skeleton
    skeleton_coords = np.argwhere(skeleton)
    
    # Calculate and plot linear function
    if len(skeleton_coords) >= gap * 2:
        coords = skeleton_coords[np.argsort(skeleton_coords[:, 1])]
        y = coords[gap:, 0]
        x = coords[:len(coords) - gap, 1]
        m, b = np.polyfit(x, y, 1)
        x_fit = np.array([0, mask.shape[1]])
        y_fit = m * x_fit + b
        axes[4].plot(x_fit, y_fit, color=colors[i], linewidth=1, alpha=1)

axes[4].axis('off')
axes[4].set_title("Stage 5: Final Analysis", fontsize=9, fontweight='medium')

plt.tight_layout()
plt.savefig('weld_detection_stages.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFive-stage visualization complete!")
print(f"Saved to: weld_detection_stages.png")
print(f"Raw detections: {len(raw_masks)}")
print(f"Refined detections: {len(all_masks)}")