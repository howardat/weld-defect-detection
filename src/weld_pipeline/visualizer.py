import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def plot_multi_stage_results(data):
    if not data:
        print("No data to plot.")
        return

    img = data['image']
    fragments = data['fragments']
    bboxes = data['bboxes']
    h, w = img.shape[:2]

    fig, axes = plt.subplots(1, 5, figsize=(25, 6), dpi=100)
    
    def get_overlay(base, mask, color=[0, 255, 0], alpha=0.5):
        overlay = base.copy()
        m_bool = mask.astype(bool)
        overlay[m_bool] = (overlay[m_bool] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        return overlay

    def add_bboxes(ax):
        for (x1, y1, x2, y2) in bboxes:
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='#00FF00', facecolor='none')
            ax.add_patch(rect)

    # Stage 1: Original + Boxes
    axes[0].imshow(img)
    add_bboxes(axes[0])
    axes[0].set_title("1. Original Image")

    # Stage 2: YOLO Mask + Boxes
    axes[1].imshow(get_overlay(img, data['yolo_mask']))
    add_bboxes(axes[1])
    axes[1].set_title("2. YOLO ROI")

    # Stage 3: All Refined Masks + Boxes
    refined_img = img.copy()
    for frag in fragments:
        refined_img = get_overlay(refined_img, frag['mask'], color=[255, 165, 0])
    axes[2].imshow(refined_img)
    add_bboxes(axes[2])
    axes[2].set_title("3. All Refined Masks")

    # Stage 4: All Skeletons (Black Background)
    skel_img = np.zeros((h, w, 3), dtype=np.uint8)
    axes[3].imshow(skel_img)
    for frag in fragments:
        coords = frag['skeleton_coords']
        axes[3].scatter(coords[:, 1], coords[:, 0], c='white', s=1)
    axes[3].set_title("4. All Skeletons")

    # Stage 5: Trend Lines (Full Function)
    axes[4].imshow(refined_img)
    x_full = np.array([0, w]) # Full image width
    for frag in fragments:
        m, b = frag['trend']
        y_full = m * x_full + b
        axes[4].plot(x_full, y_full, color='cyan', linewidth=2, label='Trend')
    
    # Clip view to image boundaries so lines don't expand the plot canvas
    axes[4].set_xlim(0, w)
    axes[4].set_ylim(h, 0)
    axes[4].set_title("5. Full Trend Lines")

    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()  