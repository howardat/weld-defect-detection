from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import seaborn as sns

from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage.util import invert

def discontinuity_assessment(image_path: str, model_path: str, padding: int = 30):
    cropped_welds = crop_weld(image_path, model_path, padding, show=False)
    all_data = [] # To store (crop, mask, skeleton, line_params)

    for weld in cropped_welds:
        # Step 2: Get Masks for this crop
        masks = intensity_filtering(weld, model_path)
        
        if masks:
            for mask in masks:
                # Step 3: Get Skeleton and Fit Line
                m, c, skeleton = fit_skeleton(mask)
                all_data.append({
                    'crop': np.array(weld),
                    'mask': mask,
                    'skeleton': skeleton,
                    'm': m,
                    'c': c
                })
    
    # Step 4: Final Visualization
    visualize(all_data)

def crop_weld(image_path: str, model_path: str, padding: int = 20, show: bool = True):
    # Crop weld and get crop mask and add padding
    model = YOLO(model_path)
    results = model.predict(image_path, conf=0.05, iou=0.2, classes=3, agnostic_nms=True, verbose=False, save=False)

    boxes = results[0].boxes

    if len(boxes) == 0:
        print("No welds detected.")
        return None

    all_coords = boxes.xyxy.cpu().numpy()

    x1 = all_coords[:, 0]
    y1 = all_coords[:, 1]
    x2 = all_coords[:, 2]
    y2 = all_coords[:, 3]

    crop_imgs = []

    for i in range(len(boxes)):
        image = []

        with Image.open(image_path) as img:
            width, hight = img.size
            image = img.copy()

        x1[i] -= padding if x1[i] - padding >=0 else 0
        y1[i] -= padding if y1[i] - padding >=0 else 0
        x2[i] += padding if x2[i] + padding <= width else width
        y2[i] += padding if y2[i] + padding <= hight else hight

        crop_imgs.append(image.crop((x1[i], y1[i], x2[i], y2[i])))
        crop_imgs[i].show() if show else None
    
    return crop_imgs

def intensity_filtering(crop, model_path: str):
    #Filtering out dark pixels from weld masks
    model = YOLO(model_path)
    results = model.predict(crop, conf=0.05, iou=0.3, classes=3, agnostic_nms=True, verbose=False, save=False)
    
    if len(results[0]) == 0:
        print("No welds detected in crop.")
        return None
    
    mask = results[0].masks[0].data.cpu().numpy()[0]

    crop_matrix = np.array(crop)
    intensity_matrix = cv2.cvtColor(crop_matrix, cv2.COLOR_RGB2GRAY)

    # instensity_mean = np.mean(intensity_matrix) #Adaptive treshold
    # treshold = instensity_mean * 1.2            #Adaptive treshold
    treshold = 50                                 #Fixed treshold
    binary_mask  = intensity_matrix > treshold
    
    h, w = results[0].orig_shape
    resized_mask = cv2.resize(mask, (w, h))

    filtered_mask = resized_mask * binary_mask

    #Find mask "Islands" & Size thresholding
    filtered_mask = filtered_mask.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filtered_mask)

    mask_list = []

    size_threshold = h * w * 0.01

    for i in range(1, num_labels):
        # Create a separate mask for just this object
        area_pixels = stats[i, cv2.CC_STAT_AREA]

        if area_pixels > size_threshold:
            individual_mask = (labels == i).astype(np.uint8) * 255
            mask_list.append(individual_mask)

    #Visualization
    # visualization = cv2.cvtColor(crop_matrix, cv2.COLOR_RGB2BGR)

    # colored_mask = np.zeros_like(visualization)
    # for i in range(len(mask_list)):
    #     colored_mask[mask_list[i] > 0] = [0, 0, 255]

    # alpha = 1
    # beta = 0.3
    # overlay = cv2.addWeighted(visualization, alpha, colored_mask, beta, 0)

    # cv2.imshow('Weld Overlay', overlay)
    # cv2.waitKey(0)

    return mask_list

def fit_skeleton(mask):
    image = invert(mask)
    skeleton = skeletonize(image)
    y_coords, x_coords = np.where(skeleton > 0)
    if len(x_coords) > 1:
        m, c = np.polyfit(x_coords, y_coords, 1)
        return m, c, skeleton
    return None, None, skeleton

def visualize(all_data):
    if not all_data:
        print("No data to visualize.")
        return

    num_samples = len(all_data)
    # Create a grid: 3 stages (Crop, Mask, Fit) for every detected discontinuity
    fig, axes = plt.subplots(num_samples, 3, figsize=(8, 1 * num_samples), squeeze=False)

    for i, data in enumerate(all_data):
        h, w = data['mask'].shape
        
        # Column 1: Original Crop
        axes[i, 0].imshow(data['crop'])
        axes[i, 0].set_title(f"Sample {i}: Original Crop")
        axes[i, 0].axis('off')

        # Column 2: Filtered Binary Mask
        axes[i, 1].imshow(data['mask'], cmap='gray')
        axes[i, 1].set_title("Intensity Filtered Mask")
        axes[i, 1].axis('off')

        # Column 3: Skeleton + Extended Linear Fit
        axes[i, 2].imshow(data['skeleton'], cmap='gray')
        
        if data['m'] is not None:
            # Calculate line points at image edges
            x_vals = np.array([0, w])
            y_vals = data['m'] * x_vals + data['c']
            
            axes[i, 2].plot(x_vals, y_vals, color='red', linewidth=2, label='Linear Fit')
            # Clip the plot to image dimensions
            axes[i, 2].set_xlim(0, w)
            axes[i, 2].set_ylim(h, 0)
            
        axes[i, 2].set_title(f"Skeleton & Full-Image Fit\ny={data['m']:.2f}x+{data['c']:.2f}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    discontinuity_assessment("../../data/zoom.jpg", "../../models/best.pt")