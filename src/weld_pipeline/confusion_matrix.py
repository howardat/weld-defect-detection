if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # 1. Load the data
    df = pd.read_csv('/Users/oljk/Downloads/Новая таблица - Matrix-2.csv', header=[0, 1])

    # 2. Define the columns for each category (Pred index, GT index)
    # Based on your CSV structure:
    tasks = [
        ("Crack grade", 1, 6),
        ("Discontinuity grade", 2, 7),
        ("Porosity grade", 3, 8),
        ("Overall grade", 4, 9)
    ]

    # 3. Setup the visualization grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, (name, pred_idx, gt_idx) in enumerate(tasks):
        # Clean data: convert to string, strip whitespace, and handle missing values
        y_pred = df.iloc[:, pred_idx].astype(str).str.strip().replace({'nan': 'None', '?': 'None'})
        y_gt = df.iloc[:, gt_idx].astype(str).str.strip().replace({'nan': 'None', '?': 'None'})

        # Get the unique labels (B, C, D, F, None) present in this specific task
        labels = sorted(list(set(y_pred.unique()) | set(y_gt.unique())))
        
        # Compute the Confusion Matrix
        cm = confusion_matrix(y_gt, y_pred, labels=labels)
        
        # Plot the heatmap on the corresponding subplot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, ax=axes[i])
        
        axes[i].set_title(f'Confusion Matrix: {name}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('Ground Truth Label')

    # 4. Final adjustments and save
    plt.tight_layout()
    plt.savefig('defect_analysis_results.png')
    plt.show()