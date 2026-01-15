import cv2
import torch
import numpy as np
from typing import List, Union
from ultralytics import YOLO

def refine_mask(original_image, yolo_masks_data, min_area_ratio=0.05, p_intensity=50):
    h, w = original_image.shape[:2]
    min_segment_area = int(h * w * min_area_ratio)
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    non_dark = (gray >= p_intensity).astype(np.uint8)
    
    final_masks = []
    for mask_data in yolo_masks_data:
        if isinstance(mask_data, torch.Tensor):
            mask_data = mask_data.cpu().numpy()
        mask_big = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_big = (mask_big > 0.5).astype(np.uint8)
        weld_clean = cv2.bitwise_and(mask_big, non_dark)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weld_clean, 8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_segment_area:
                final_masks.append((labels == i).astype(np.uint8))
    return final_masks

class WeldDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict_and_refine(self, image, pad=20, min_area_ratio=0.05):
        # Move your Stage 1 and Stage 2 crop logic here...
        # Returns all_masks
        pass