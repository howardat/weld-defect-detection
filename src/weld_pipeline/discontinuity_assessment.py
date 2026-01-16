from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage.util import invert

def discontinuity_assessment(image_path: str, model_path: str, padding: int = 30):
    copped_welds = crop_weld(image_path, model_path, padding, show=True)
    all_masks_list = []
    for weld in copped_welds:
        all_masks_list += intensity_filtering(weld, model_path)       
    fit_function(all_masks_list)

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
    visualization = cv2.cvtColor(crop_matrix, cv2.COLOR_RGB2BGR)

    colored_mask = np.zeros_like(visualization)
    for i in range(len(mask_list)):
        colored_mask[mask_list[i] > 0] = [0, 0, 255]

    alpha = 1
    beta = 0.3
    overlay = cv2.addWeighted(visualization, alpha, colored_mask, beta, 0)

    cv2.imshow('Weld Overlay', overlay)
    cv2.waitKey(0)

    return mask_list

def fit_function(mask_list):

    for i, mask in enumerate(mask_list):
        image = invert(mask)

        # perform skeletonization
        skeleton = skeletonize(image)

        # display results
        fig, axes = plt.subplots(nrows=1, ncols=len(mask_list), figsize=(8, 4), sharex=True, sharey=True)

        ax = axes.ravel()

        ax[i].imshow(skeleton, cmap=plt.cm.gray)
        ax[i].axis('off')
        ax[i].set_title('skeleton', fontsize=20)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    discontinuity_assessment("../../data/zoom.jpg", "../../models/best.pt")