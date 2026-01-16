import numpy as np
import cv2

def get_bounding_box_from_mask(mask: np.ndarray, padding: int = 20):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0: return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    h, w = mask.shape
    return max(0, x_min-padding), max(0, y_min-padding), min(w, x_max+padding), min(h, y_max+padding)