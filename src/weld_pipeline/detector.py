import cv2
import torch
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
from utils import get_bounding_box_from_mask

class WeldDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def smooth_mask(self, mask, kernel_size=5):
        if mask.max() <= 1: mask = (mask * 255).astype(np.uint8)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
        blurred = cv2.GaussianBlur(mask_cleaned, (kernel_size, kernel_size), 0)
        _, final_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        return (final_mask / 255).astype(np.uint8)

    def refine_mask(self, crop_image, yolo_masks_data, min_area_ratio=0.05):
        h, w = crop_image.shape[:2]
        min_segment_area = int(h * w * min_area_ratio)
        gray = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
        non_dark = (gray >= 50).astype(np.uint8)
        
        refined_fragments = []
        for mask_data in yolo_masks_data:
            if isinstance(mask_data, torch.Tensor): mask_data = mask_data.cpu().numpy()
            mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
            weld_clean = cv2.bitwise_and((mask_resized > 0.5).astype(np.uint8), non_dark)
            weld_smooth = self.smooth_mask(weld_clean)
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weld_smooth, 8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_segment_area:
                    refined_fragments.append((labels == i).astype(np.uint8))
        return refined_fragments

    def run_inference(self, image, conf=0.05):
        orig_h, orig_w = image.shape[:2]
        results = self.model.predict(image, conf=conf, classes=3, verbose=False)
        if not results or results[0].masks is None: return []

        all_fragments = []
        bboxes = []
        yolo_full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for mask_tensor in results[0].masks.data:
            mask_np = cv2.resize(mask_tensor.cpu().numpy(), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            yolo_full_mask = cv2.bitwise_or(yolo_full_mask, (mask_np > 0.5).astype(np.uint8))
            
            bbox = get_bounding_box_from_mask(mask_np, padding=30)
            if bbox is None: continue
            bboxes.append(bbox) # Store for visualization
            
            x1, y1, x2, y2 = bbox
            crop_img = image[y1:y2, x1:x2]
            crop_res = self.model.predict(crop_img, conf=conf, classes=3, verbose=False)
            
            if not crop_res or crop_res[0].masks is None: continue
            
            refined_masks = self.refine_mask(crop_img, crop_res[0].masks.data)
            for r_mask in refined_masks:
                skeleton = skeletonize(r_mask.astype(bool))
                local_coords = np.argwhere(skeleton)
                if len(local_coords) > 15:
                    global_coords = local_coords.copy()
                    global_coords[:, 0] += y1
                    global_coords[:, 1] += x1
                    
                    y_c, x_c = global_coords[:, 0], global_coords[:, 1]
                    m, b = np.polyfit(x_c, y_c, 1)
                    
                    full_r_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    full_r_mask[y1:y2, x1:x2] = r_mask

                    all_fragments.append({
                        "mask": full_r_mask,
                        "skeleton_coords": global_coords,
                        "trend": (m, b)
                    })
        
        return {
            "image": image, 
            "yolo_mask": yolo_full_mask, 
            "fragments": all_fragments,
            "bboxes": bboxes
        }