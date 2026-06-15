from weld_pipeline.visualization import create_comparison_composition
# from weld_pipeline.vlm import WeldAuditor
from weld_pipeline.porosity_check import porosity_check
from weld_pipeline.discontinuity_check import discontinuity_check
from weld_pipeline.cracks_check import cracks_check
from weld_pipeline import timing
from ultralytics import YOLO
import torch

from pathlib import Path
from datetime import datetime
import json

def process_single_image(image_path, model_path, json_dir, report_dir, seg_model,
                         disc_threshold=0.9, otsu_multiplier=0.7, auditor=None):
    """Encapsulates the logic for a single weld analysis."""
    print(f"\n--- Processing: {image_path.name} ---")

    # 1. Run Analysis Modules
    # Discontinuity check
    disc_bool, refined_masks, line_params = discontinuity_check(
        image_path=str(image_path),
        model_path=str(model_path),
        threshold=disc_threshold,
        otsu_multiplier=otsu_multiplier,
        visualize=False,
        seg_model=seg_model
    )

    # Porosity check
    clean_json_list, raw_pore_data, _ = porosity_check(
        image_path=str(image_path),
        model_path=str(model_path),
        px_to_mm=PX_TO_MM,
        throat_thickness=THROAT_THICKNESS,
        marker_size_mm=MARKER_SIZE_MM,
        otsu_multiplier=PORE_OTSU_MULT,
        min_pore_size_mm=MIN_PORE_SIZE_MM,
        pore_cluster_distance_mm=PORE_CLUSTER_DIST_MM,
        visualize=False,
        seg_model=seg_model,
    )

    # Cracks check
    crack_boxes_list, crack_masks_list, all_crack_detections, weld_mask = cracks_check(
        image_path=str(image_path), 
        model_path=str(model_path),
        seg_model=seg_model
    )

    # 2. Format JSON for VLM Consumption
    discontinuity_data = [{'discontinuity': disc_bool}]
    final_json = discontinuity_data + all_crack_detections + clean_json_list
    
    for item in final_json:
        item.pop('confidence', None)
        item.pop('class', None)
        item.pop('segmentation', None)
        if 'box' in item:
            b = item['box']
            item['bbox'] = [int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])]
            item.pop('box', None)
    
    # Save image-specific JSON — filename encodes the parameter values used
    image_json_path = json_dir / f"{image_path.stem}_otsu{otsu_multiplier}_sim{disc_threshold}.json"
    with open(image_json_path, 'w') as f:
        json.dump(final_json, f, indent=4)

    # 3. Release YOLO's cached CUDA memory before the VLM to avoid VRAM contention
    torch.cuda.empty_cache()

    pore_summary = "\n".join(
        f"  Pore {i+1}: grade={p.get('grade','?')}  size={p.get('size','?')}mm"
        for i, p in enumerate(clean_json_list)
    ) or "  None detected"

    detection_report = (
        f"Discontinuity: {disc_bool}\n"
        f"Cracks: {len(all_crack_detections)}\n"
        f"Pores ({len(clean_json_list)}):\n{pore_summary}"
    )

    if auditor is not None:
        report_v, report_g = auditor.run_single_audit(image_path, image_json_path)
    else:
        report_v = detection_report
        report_g = "VLM analysis skipped"

    # 4. Generate Visual Composition
    output_filename = report_dir / f"{image_path.stem}_final_audit.jpg"
    create_comparison_composition(
        image_path=image_path,
        report_v_text=report_v,
        report_g_text=report_g,
        output_path=output_filename,
        line_params=line_params, 
        discontinuity_masks=refined_masks, 
        disc_bool=disc_bool,
        weld_mask=weld_mask, 
        porosity_data=raw_pore_data,
        crack_masks=crack_masks_list
    )
    print(f"Finished: {output_filename.name}")
    return disc_bool

# --- Shared settings (also imported by streamlit_app.py) ---
DISC_THRESHOLD       = 0.9    # Similarity threshold for discontinuity detection (>1 disables it)
OTSU_MULTIPLIER      = 0.75   # Otsu multiplier for discontinuity check
PX_TO_MM             = 0.105
THROAT_THICKNESS      = 10.0
MARKER_SIZE_MM       = 10.0   # Physical ArUco marker size in mm
MIN_PORE_SIZE_MM     = 0.8    # Minimum pore diameter in mm
PORE_OTSU_MULT       = 0.6    # Otsu multiplier for porosity intensity filter
PORE_CLUSTER_DIST_MM = 0.05   # Distance to merge nearby pores (mm)
USE_VLM              = False   # Set to True to enable VLM reports

def main():
    # Setup Paths
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent.parent

    # IMAGE_DIR = PROJECT_ROOT / "data" / "tmp"
    IMAGE_DIR = Path("/Users/oljk/Projects/weld-pipeline/data/discontinuity")
    MODEL_PATH = PROJECT_ROOT / "models" / "wda11s-seg.pt"
    seg_model = YOLO(MODEL_PATH)
    # Directories for results
    JSON_OUT_DIR = PROJECT_ROOT / "data" / "json_output"
    REPORT_OUT_DIR = PROJECT_ROOT / "reports" / "vlm_results" / "new"

    TIMING_CSV = PROJECT_ROOT / "reports" / "inference_times.csv"

    # Ensure directories exist
    JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gather all images
    extensions = ("*.jpeg", "*.jpg", "*.png")
    all_images = []
    for ext in extensions:
        all_images.extend(list(IMAGE_DIR.glob(ext)))

    if not all_images:
        print(f"No images found in {IMAGE_DIR}")
        return

    MAX_IMAGES = 100
    all_images = all_images[:MAX_IMAGES]
    print(f"Found {len(all_images)} images (capped at {MAX_IMAGES}). Starting batch processing...")
    if USE_VLM:
        from weld_pipeline.vlm import WeldAuditor
        auditor = WeldAuditor()
    else:
        auditor = None
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    DISC_RESULTS_PATH = REPORT_OUT_DIR / f"discontinuity_results_otsu{OTSU_MULTIPLIER}_sim{DISC_THRESHOLD}_{_ts}.json"
    discontinuity_results = {}

    # Batch Loop
    for target_image in all_images:

        try:
            disc_bool = process_single_image(
                target_image,
                MODEL_PATH,
                JSON_OUT_DIR,
                REPORT_OUT_DIR,
                seg_model=seg_model,
                disc_threshold=DISC_THRESHOLD,
                otsu_multiplier=OTSU_MULTIPLIER,
                auditor=auditor,
            )
            discontinuity_results[target_image.name] = bool(disc_bool)
        except Exception as e:
            print(f"FAILED to process {target_image.name}: {e}")
            discontinuity_results[target_image.name] = None

        with open(DISC_RESULTS_PATH, 'w') as f:
            json.dump(discontinuity_results, f, indent=4)

    timing.save_csv(TIMING_CSV)
    print(f"\nSUCCESS: Batch processing complete. Check {REPORT_OUT_DIR} for results.")
    print(f"Inference times saved to {TIMING_CSV}")

if __name__ == "__main__":
    main()