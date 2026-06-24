"""
Run discontinuity_check on every image in IMAGE_DIR and save per-image
predictions to predictions_discontinuity.json. Run once; re-run only when
the dataset or parameters change.
"""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent / "src"))
from weld_pipeline.discontinuity_check import discontinuity_check

# ─── Editable constants ────────────────────────────────────────────────────────
IMAGE_DIR   = Path("./data/discontinuity")
MODEL_PATH  = Path("./models/wda11s-seg.pt")
OUTPUT_JSON = Path("./predictions_discontinuity.json")

OTSU_MULTIPLIER = 0.6
DISC_THRESHOLD  = 1.0
# ──────────────────────────────────────────────────────────────────────────────


def _ground_truth(name: str) -> bool:
    return "discontinuity" in name.lower()


def main() -> None:
    images = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in IMAGE_DIR.glob(ext)
    )
    if not images:
        print(f"No images found in {IMAGE_DIR}")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH} …")
    seg_model = YOLO(str(MODEL_PATH))

    records = []
    for img_path in images:
        gt = _ground_truth(img_path.name)
        try:
            found, _, _ = discontinuity_check(
                image_path=str(img_path),
                model_path=str(MODEL_PATH),
                threshold=DISC_THRESHOLD,
                otsu_multiplier=OTSU_MULTIPLIER,
                visualize=False,
                seg_model=seg_model,
            )
        except Exception as exc:
            print(f"  SKIP {img_path.name}: {exc}")
            continue

        status = "✓" if found == gt else "✗"
        print(f"  {status}  {img_path.name:<45}  gt={'disc' if gt else 'clean'}  pred={'disc' if found else 'clean'}")
        records.append({"image": img_path.name, "gt": gt, "pred": found})

    payload = {
        "otsu_multiplier": OTSU_MULTIPLIER,
        "disc_threshold": DISC_THRESHOLD,
        "image_dir": str(IMAGE_DIR),
        "predictions": records,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved {len(records)} predictions → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
