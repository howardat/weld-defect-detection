"""
Validates porosity_check against COCO ground-truth annotations.
Uses mask IoU (pixel-level) for matching, consistent with YOLO's segmentation eval.
Outputs a confusion matrix image and a JSON summary.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ─── Editable constants ────────────────────────────────────────────────────────
DATASET_DIR       = Path("./data/porosity_val")
ANNOTATIONS_FILE  = DATASET_DIR / "_annotations.coco.json"
MODEL_PATH        = Path("./models/wda11s-seg.pt")
OUTPUT_DIR        = Path("./reports/porosity_validation")
OUTPUT_JSON       = OUTPUT_DIR / "validation_results.json"
OUTPUT_MATRIX_PNG = OUTPUT_DIR / "confusion_matrix.png"

IOU_THRESHOLD = 0.5   # YOLO standard for segmentation: mask IoU ≥ 0.5

# porosity_check parameters — mirror your production settings
PX_TO_MM             = 0.105
THROAT_THICKNESS     = 10.0
MARKER_SIZE_MM       = 10.0
OTSU_MULTIPLIER      = 0.6
MIN_PORE_SIZE_MM     = 0.8
PORE_CLUSTER_DIST_MM = 0.05
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent / "src"))
from weld_pipeline.porosity_check import porosity_check

COCO_GRADE_MAP = {1: "C", 2: "D", 3: "C", 4: "Reject"}  # GT-porosity (id=3) treated as C until graded

CLASSES = ["C", "D", "Reject", "background"]  # same on both axes, YOLO-style

# BGR colors per grade for comparison images
GRADE_COLOR = {
    "C":      (0, 255, 255),   # yellow
    "D":      (0, 165, 255),   # orange
    "Reject": (0, 0, 255),     # red
}
COMPARISON_HEIGHT = 900   # px height of each panel in the saved comparison


# ─── Helpers ───────────────────────────────────────────────────────────────────

def coco_to_xyxy(bbox: list) -> list:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def mask_iou(gt_seg: list, gt_bbox_xyxy: list,
             pred_contour: np.ndarray, pred_bbox_xyxy: list) -> float:
    """
    Pixel-level IoU between a COCO segmentation polygon and an OpenCV contour.
    Rasterises both into a tight crop (union of their bboxes) to avoid
    allocating full-image arrays for every pair.
    """
    x1 = int(min(gt_bbox_xyxy[0], pred_bbox_xyxy[0]))
    y1 = int(min(gt_bbox_xyxy[1], pred_bbox_xyxy[1]))
    x2 = int(max(gt_bbox_xyxy[2], pred_bbox_xyxy[2])) + 1
    y2 = int(max(gt_bbox_xyxy[3], pred_bbox_xyxy[3])) + 1
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return 0.0

    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for seg in gt_seg:
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] -= x1
        pts[:, 1] -= y1
        cv2.fillPoly(gt_mask, [pts.astype(np.int32)], 1)

    pred_mask = np.zeros((h, w), dtype=np.uint8)
    cnt = pred_contour.copy().astype(np.int32)
    cnt[:, :, 0] -= x1
    cnt[:, :, 1] -= y1
    cv2.drawContours(pred_mask, [cnt], -1, 1, cv2.FILLED)

    inter = int(np.count_nonzero(gt_mask & pred_mask))
    union = int(np.count_nonzero(gt_mask | pred_mask))
    return inter / union if union > 0 else 0.0


def save_comparison(img_path: Path, gt_segs: list, gt_grades: list,
                    pred_contours: list, pred_grades: list,
                    output_path: Path) -> None:
    """Save predicted (left) vs GT (right) side-by-side with color-coded filled masks."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return
    h, w = img_bgr.shape[:2]

    def _apply_overlay(base, entries):
        overlay = base.copy()
        for mask, grade in entries:
            color = GRADE_COLOR.get(grade, (200, 200, 200))
            overlay[mask > 0] = color
        return cv2.addWeighted(overlay, 0.75, base, 0.25, 0)

    # GT panel
    gt_entries = []
    for seg, grade in zip(gt_segs, gt_grades):
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in seg:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        gt_entries.append((mask, grade))

    # Prediction panel
    pred_entries = []
    for contour, grade in zip(pred_contours, pred_grades):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour.astype(np.int32)], -1, 1, cv2.FILLED)
        pred_entries.append((mask, grade))

    pred_panel = _apply_overlay(img_bgr, pred_entries)
    gt_panel   = _apply_overlay(img_bgr, gt_entries)

    scale  = COMPARISON_HEIGHT / h
    new_w  = int(w * scale)
    pred_panel = cv2.resize(pred_panel, (new_w, COMPARISON_HEIGHT))
    gt_panel   = cv2.resize(gt_panel,   (new_w, COMPARISON_HEIGHT))

    cv2.imwrite(str(output_path), np.hstack([pred_panel, gt_panel]))


def greedy_match(iou_matrix: np.ndarray, threshold: float) -> list[tuple]:
    """Greedy matching on a pre-computed IoU matrix, highest IoU first."""
    if iou_matrix.size == 0:
        return []
    used_gt, used_pred, pairs = set(), set(), []
    for idx in np.argsort(iou_matrix.ravel())[::-1]:
        gi, pi = np.unravel_index(idx, iou_matrix.shape)
        if iou_matrix[gi, pi] < threshold:
            break
        if gi in used_gt or pi in used_pred:
            continue
        pairs.append((int(gi), int(pi)))
        used_gt.add(gi)
        used_pred.add(pi)
    return pairs


def plot_confusion_matrix(cm: np.ndarray, row_labels: list, col_labels: list,
                           title: str, output_path: Path) -> None:
    rows, cols = len(row_labels), len(col_labels)
    fig, ax = plt.subplots(figsize=(cols + 2, rows + 2))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ground Truth", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=12, pad=14)
    for i in range(rows):
        for j in range(cols):
            val = int(cm[i, j])
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    img_info = {img["id"]: img for img in coco["images"]}
    img_anns: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    print(f"Loading model from {MODEL_PATH} …")
    seg_model = YOLO(str(MODEL_PATH))

    row_labels = CLASSES
    col_labels = CLASSES
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    all_pairs: list[dict] = []

    for img_id, info in img_info.items():
        img_path = DATASET_DIR / info["file_name"]
        if not img_path.exists():
            print(f"  Image not found, skipping: {img_path.name}")
            continue

        print(f"\nProcessing {info['file_name']} …")
        throat = info.get("throat_thickness") or THROAT_THICKNESS

        try:
            _, pore_entries, _ = porosity_check(
                image_path=str(img_path),
                model_path=str(MODEL_PATH),
                px_to_mm=PX_TO_MM,
                throat_thickness=throat,
                marker_size_mm=MARKER_SIZE_MM,
                otsu_multiplier=OTSU_MULTIPLIER,
                min_pore_size_mm=MIN_PORE_SIZE_MM,
                pore_cluster_distance_mm=PORE_CLUSTER_DIST_MM,
                visualize=False,
                seg_model=seg_model,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        anns          = img_anns[img_id]
        gt_grades     = [COCO_GRADE_MAP.get(a["category_id"], "C") for a in anns]
        gt_segs       = [a["segmentation"] for a in anns]
        gt_bboxes     = [coco_to_xyxy(a["bbox"]) for a in anns]

        pred_grades   = [p["grade"] for p in pore_entries]
        pred_contours = [p["_contour"] for p in pore_entries]
        pred_bboxes   = [
            [p["box"]["x1"], p["box"]["y1"], p["box"]["x2"], p["box"]["y2"]]
            for p in pore_entries
        ]

        print(f"  GT: {len(gt_segs)} pores  |  Predicted: {len(pred_contours)} pores")

        comparison_path = OUTPUT_DIR / f"{img_path.stem}_comparison.jpg"
        save_comparison(img_path, gt_segs, gt_grades, pred_contours, pred_grades, comparison_path)
        print(f"  Comparison saved → {comparison_path.name}")

        # Build mask IoU matrix
        if gt_segs and pred_contours:
            iou_mat = np.array([
                [mask_iou(gt_segs[gi], gt_bboxes[gi], pred_contours[pi], pred_bboxes[pi])
                 for pi in range(len(pred_contours))]
                for gi in range(len(gt_segs))
            ])
        else:
            iou_mat = np.zeros((len(gt_segs), len(pred_contours)))

        matches      = greedy_match(iou_mat, IOU_THRESHOLD)
        matched_gt   = {gi for gi, _ in matches}
        matched_pred = {pi for _, pi in matches}

        # Matched pairs
        for gi, pi in matches:
            gt_g, pred_g = gt_grades[gi], pred_grades[pi]
            ri = CLASSES.index(gt_g) if gt_g in CLASSES else 0
            ci = CLASSES.index(pred_g) if pred_g in CLASSES else CLASSES.index("background")
            cm[ri, ci] += 1
            iou_val = round(float(iou_mat[gi, pi]), 3)
            all_pairs.append({"image": info["file_name"], "gt": gt_g, "pred": pred_g, "iou": iou_val})
            print(f"    MATCH  GT={gt_g:4s}  pred={pred_g:4s}  IoU={iou_val:.3f}  {'✓' if gt_g == pred_g else '✗'}")

        # False negatives — unmatched GT → "background" column
        for gi in range(len(gt_segs)):
            if gi not in matched_gt:
                gt_g = gt_grades[gi]
                ri = CLASSES.index(gt_g) if gt_g in CLASSES else 0
                cm[ri, CLASSES.index("background")] += 1
                all_pairs.append({"image": info["file_name"], "gt": gt_g, "pred": "background", "iou": 0.0})
                print(f"    MISSED  GT={gt_g}")

        # False positives — unmatched predictions → "background" row
        for pi in range(len(pred_contours)):
            if pi not in matched_pred:
                pred_g = pred_grades[pi]
                ri = CLASSES.index("background")
                ci = CLASSES.index(pred_g) if pred_g in CLASSES else CLASSES.index("background")
                cm[ri, ci] += 1
                all_pairs.append({"image": info["file_name"], "gt": "background", "pred": pred_g, "iou": 0.0})
                print(f"    FALSE POS  pred={pred_g}")

    # Summary stats
    tp = sum(1 for p in all_pairs if p["gt"] != "background" and p["pred"] != "background")
    fp = sum(1 for p in all_pairs if p["gt"] == "background")
    fn = sum(1 for p in all_pairs if p["gt"] != "background" and p["pred"] == "background")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    matched_pairs = [p for p in all_pairs if p["gt"] != "background" and p["pred"] != "background"]
    grade_correct = sum(1 for p in matched_pairs if p["gt"] == p["pred"])
    grade_acc     = grade_correct / len(matched_pairs) if matched_pairs else 0.0

    print("\n─── Validation Summary ───────────────────────────────────────────────")
    print(f"  GT pores total   : {tp + fn}")
    print(f"  Predictions total: {tp + fp}")
    print(f"  True Positives   : {tp}  (mask IoU ≥ {IOU_THRESHOLD})")
    print(f"  False Positives  : {fp}")
    print(f"  False Negatives  : {fn}")
    print(f"  Precision        : {precision:.3f}")
    print(f"  Recall           : {recall:.3f}")
    print(f"  F1               : {f1:.3f}")
    print(f"  Grade accuracy   : {grade_acc:.3f}  ({grade_correct}/{len(matched_pairs)} matched pairs)")

    plot_confusion_matrix(
        cm, row_labels, col_labels,
        title=(f"Porosity Grading — Confusion Matrix  (mask IoU ≥ {IOU_THRESHOLD})\n"
               f"Precision={precision:.2f}  Recall={recall:.2f}  F1={f1:.2f}"),
        output_path=OUTPUT_MATRIX_PNG,
    )

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "iou_threshold": IOU_THRESHOLD,
            "stats": {
                "tp": tp, "fp": fp, "fn": fn,
                "precision":      round(precision, 4),
                "recall":         round(recall,    4),
                "f1":             round(f1,        4),
                "grade_accuracy": round(grade_acc, 4),
            },
            "pairs": all_pairs,
        }, f, indent=2)
    print(f"Detailed results written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
