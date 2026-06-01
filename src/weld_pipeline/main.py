from weld_pipeline.visualization import create_comparison_composition
# from weld_pipeline.vlm import WeldAuditor
from weld_pipeline.porosity_check import porosity_check
from weld_pipeline.discontinuity_check import discontinuity_check
from weld_pipeline.cracks_check import cracks_check
from weld_pipeline import timing
from ultralytics import YOLO
import torch

from pathlib import Path
import json

def process_single_image(image_path, model_path, json_dir, report_dir, seg_model, auditor=None):
    """Encapsulates the logic for a single weld analysis."""
    print(f"\n--- Processing: {image_path.name} ---")
    
    # 1. Run Analysis Modules
    # Discontinuity check
    disc_bool, refined_masks, line_params = discontinuity_check(
        image_path=str(image_path), 
        model_path=str(model_path), 
        threshold=0.9,
        visualize=False,  # Disabled for batch to prevent popup windows
        seg_model=seg_model
    )

    # Porosity check
    clean_json_list, raw_pore_data = porosity_check(
        image_path=str(image_path), 
        model_path=str(model_path), 
        px_to_mm=0.105, 
        plate_thickness_s=10.0,
        visualize=False,
        seg_model=seg_model  # Disabled for batch to prevent popup windows
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
    
    # Save image-specific JSON
    image_json_path = json_dir / f"{image_path.stem}.json"
    with open(image_json_path, 'w') as f:
        json.dump(final_json, f, indent=4)

    # 3. Release YOLO's cached CUDA memory before the VLM to avoid VRAM contention
    torch.cuda.empty_cache()

    if auditor is not None:
        report_v, report_g = auditor.run_single_audit(image_path, image_json_path)
    else:
        report_v, report_g = "VLM analysis skipped" + "\nDiscontinuity" + str(disc_bool), "VLM analysis skipped"

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
    
    USE_VLM = False  # Set to True to enable VLM reports

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
    DISC_RESULTS_PATH = REPORT_OUT_DIR / "discontinuity_results.json"
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