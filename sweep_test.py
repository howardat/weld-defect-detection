"""
Parameter sweep for discontinuity_check over otsu_multiplier × threshold grid.
Edit the constants below to change sweep ranges, image folder, or output paths.
"""
import json
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ─── Editable constants ────────────────────────────────────────────────────────
IMAGE_DIR   = Path("./data/tmp")
MODEL_PATH  = Path("./models/wda11s-seg.pt")
OUTPUT_JSON = Path("./sweep_results.json")

OTSU_MIN  = 0.5
OTSU_MAX  = 1.0
OTSU_STEP = 0.1

THRESH_MIN  = 0.5
THRESH_MAX  = 1.0
THRESH_STEP = 0.1
# ──────────────────────────────────────────────────────────────────────────────

# Add the project src to the path so the package resolves without installing
sys.path.insert(0, str(Path(__file__).parent / "src"))
from weld_pipeline.discontinuity_check import discontinuity_check


def _arange_rounded(start: float, stop: float, step: float, decimals: int = 2) -> list[float]:
    """np.arange with inclusive stop and float rounding to avoid drift."""
    n = round((stop - start) / step)
    return [round(start + i * step, decimals) for i in range(n + 1)]


def _load_images(image_dir: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    images: list[Path] = []
    for ext in exts:
        images.extend(image_dir.glob(ext))
    return sorted(images)


def _ground_truth(name: str) -> bool:
    return "discontinuity" in name.lower()


def main() -> None:
    # ── Build parameter grid ───────────────────────────────────────────────────
    otsu_values  = _arange_rounded(OTSU_MIN,  OTSU_MAX,  OTSU_STEP)
    thresh_values = _arange_rounded(THRESH_MIN, THRESH_MAX, THRESH_STEP)
    combos = [(o, t) for o in otsu_values for t in thresh_values]
    total_combos = len(combos)
    print(f"Grid: {len(otsu_values)} otsu × {len(thresh_values)} threshold = {total_combos} combos")

    # ── Load images ────────────────────────────────────────────────────────────
    images = _load_images(IMAGE_DIR)
    if not images:
        print(f"No images found in {IMAGE_DIR}. Adjust IMAGE_DIR and re-run.")
        sys.exit(1)
    print(f"Found {len(images)} images in {IMAGE_DIR}")

    ground_truth_list = [
        {"name": img.name, "discontinuity": _ground_truth(img.name)}
        for img in images
    ]
    gt_map = {entry["name"]: entry["discontinuity"] for entry in ground_truth_list}

    # ── Load YOLO model once ───────────────────────────────────────────────────
    print(f"Loading model from {MODEL_PATH} …")
    seg_model = YOLO(str(MODEL_PATH))

    # ── Sweep ─────────────────────────────────────────────────────────────────
    results: list[dict] = []

    for combo_idx, (otsu_mult, disc_thresh) in enumerate(combos, start=1):
        print(f"\nTesting combo {combo_idx}/{total_combos}  "
              f"otsu_multiplier={otsu_mult}  threshold={disc_thresh}")

        correct = 0
        total   = 0

        for img_path in images:
            try:
                found, _, _ = discontinuity_check(
                    image_path=str(img_path),
                    model_path=str(MODEL_PATH),
                    threshold=disc_thresh,
                    otsu_multiplier=otsu_mult,
                    visualize=False,
                    seg_model=seg_model,
                )
            except Exception as exc:
                print(f"  SKIP {img_path.name}: {exc}")
                continue

            if found == gt_map[img_path.name]:
                correct += 1
            total += 1

        accuracy = round(correct / total, 4) if total > 0 else 0.0
        results.append({
            "otsu_multiplier":        otsu_mult,
            "discontinuity_threshold": disc_thresh,
            "accuracy":               accuracy,
            "correct":                correct,
            "total":                  total,
        })
        print(f"  → accuracy={accuracy:.4f}  ({correct}/{total})")

    # ── Write JSON ────────────────────────────────────────────────────────────
    output = {
        "sweep_config": {
            "otsu":      [OTSU_MIN,  OTSU_MAX,  OTSU_STEP],
            "threshold": [THRESH_MIN, THRESH_MAX, THRESH_STEP],
        },
        "ground_truth": ground_truth_list,
        "results":      results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {OUTPUT_JSON}")

    top = sorted(results, key=lambda r: r["accuracy"], reverse=True)[:5]
    print("\n─── Top results ─────────────────────────────────────────────────")
    print(f"{'Rank':<6}{'otsu_multiplier':<18}{'threshold':<12}{'accuracy':<10}{'correct/total'}")
    for rank, r in enumerate(top, start=1):
        print(f"{rank:<6}{r['otsu_multiplier']:<18}{r['discontinuity_threshold']:<12}"
              f"{r['accuracy']:<10}{r['correct']}/{r['total']}")


if __name__ == "__main__":
    main()
