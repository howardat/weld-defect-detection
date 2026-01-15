import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from matplotlib.colors import hsv_to_rgb

def plot_weld_results(image, all_masks, gap=5):
    """
    Creates a comparison figure: 
    Left: Original image with all fitted lines.
    Right: Individual subplots for each detection.
    """
    num_detections = len(all_masks)
    if num_detections == 0:
        print("No masks to plot.")
        return

    # Create subplots (1 for the 'all' view + 1 per detection)
    fig, axes = plt.subplots(1, num_detections + 1, figsize=(5 * (num_detections + 1), 5))
    
    # Handle the case of exactly 1 detection (axes won't be a list)
    if num_detections == 1:
        axes = [axes[0], axes[1]]

    # --- 1) Main Axis: All Linear Functions ---
    ax_all = axes[0]
    ax_all.imshow(image)
    ax_all.axis('off')
    ax_all.set_title("All Linear Functions")

    hues = np.linspace(0, 1, num_detections, endpoint=False)
    colors = [hsv_to_rgb((h, 1, 1)) for h in hues]

    for i, mask in enumerate(all_masks):
        skeleton = skeletonize(mask.astype(bool))
        coords = np.argwhere(skeleton)
        
        if len(coords) < gap * 2:
            continue
            
        coords = coords[np.argsort(coords[:, 1])]
        y = coords[gap:, 0]
        x = coords[:len(coords)-gap, 1]
        
        m, b = np.polyfit(x, y, 1)
        x_fit = np.array([0, mask.shape[1]])
        y_fit = m * x_fit + b
        
        ax_all.plot(x_fit, y_fit, color=colors[i], linewidth=1)
        ax_all.plot([], [], color=colors[i], label=f"y={m:.2f}x+{b:.1f}")

    ax_all.legend(loc='upper left', fontsize=9)

    # --- 2) Individual Subplots ---
    for i, mask in enumerate(all_masks):
        ax = axes[i+1]
        ax.imshow(image)
        ax.axis('off')
        
        # Overlay semi-transparent green mask
        mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_rgba[mask > 0] = [0, 255, 0, 80]
        ax.imshow(mask_rgba)
        
        skeleton = skeletonize(mask.astype(bool))
        coords = np.argwhere(skeleton)
        
        if len(coords) >= gap * 2:
            coords = coords[np.argsort(coords[:, 1])]
            y = coords[gap:, 0]
            x = coords[:len(coords)-gap, 1]
            m, b = np.polyfit(x, y, 1)
            x_fit = np.array([0, mask.shape[1]])
            y_fit = m * x_fit + b
            
            ax.plot(x_fit, y_fit, color='blue', linewidth=1)
            ax.text(0, b, f"y={m:.2f}x+{b:.1f}", color='blue', fontsize=9, verticalalignment='bottom')
        
        ax.set_title(f'Detection {i+1}')

    plt.tight_layout()
    plt.show()