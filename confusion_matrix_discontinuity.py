"""
Plot confusion matrix from predictions_discontinuity.json.
Run predict_discontinuity.py first to generate that file.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_JSON = Path("./predictions_discontinuity.json")
OUTPUT_PNG = Path("./confusion_matrix_discontinuity.png")

CLASSES = ["Clean", "Discontinuity"]


def main() -> None:
    if not INPUT_JSON.exists():
        print(f"Input file not found: {INPUT_JSON}")
        print("Run predict_discontinuity.py first.")
        sys.exit(1)

    with open(INPUT_JSON) as f:
        data = json.load(f)

    otsu      = data["otsu_multiplier"]
    threshold = data["disc_threshold"]
    records   = data["predictions"]

    cm = np.zeros((2, 2), dtype=int)
    for r in records:
        cm[int(r["gt"])][int(r["pred"])] += 1

    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total     = len(records)
    correct   = tn + tp
    accuracy  = correct / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"otsu_multiplier={otsu}  threshold={threshold}")
    print(f"Accuracy : {accuracy:.4f}  ({correct}/{total})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        linewidths=0, ax=ax1,
    )
    ax1.set_xlabel("Predicted", fontsize=12)
    ax1.set_ylabel("Actual", fontsize=12)
    ax1.set_title("Counts", fontsize=12)

    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        linewidths=0, vmin=0.0, vmax=1.0, ax=ax2,
    )
    ax2.set_xlabel("Predicted", fontsize=12)
    ax2.set_ylabel("Actual", fontsize=12)
    ax2.set_title("Normalized (row %)", fontsize=12)

    fig.suptitle(
        f"Discontinuity Detection — Confusion Matrix\n"
        f"otsu={otsu}  threshold={threshold}  "
        f"acc={accuracy:.2%}  F1={f1:.2%}",
        fontsize=12,
    )
    plt.tight_layout(pad=2.0)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {OUTPUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
