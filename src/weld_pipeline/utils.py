import numpy as np
import cv2

def calculate_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return 0 if union == 0 else inter / union

def get_bounding_box_from_mask(mask: np.ndarray, padding: int = 20):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0: return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    h, w = mask.shape
    return max(0, x_min-padding), max(0, y_min-padding), min(w, x_max+padding), min(h, y_max+padding)

def deduplicate_masks(masks, iou_threshold=0.1):
    areas = [m.sum() for m in masks]
    order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
    kept = []
    for idx in order:
        if all(calculate_iou(masks[idx], k) <= iou_threshold for k in kept):
            kept.append(masks[idx])
    return kept