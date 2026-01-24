from weld_pipeline.visualization import create_comparison_composition
from weld_pipeline.vlm import WeldAuditor
from weld_pipeline.porosity_check import porosity_check
from weld_pipeline.discontinuity_check import discontinuity_check
from weld_pipeline.cracks_check import cracks_check

from pathlib import Path
import json

def process_single_image(image_path, model_path, json_dir, report_dir):
    """Encapsulates the logic for a single weld analysis."""
    print(f"\n--- Processing: {image_path.name} ---")
    
    # 1. Run Analysis Modules
    # Discontinuity check
    disc_bool, refined_masks, line_params = discontinuity_check(
        image_path=str(image_path), 
        model_path=str(model_path), 
        threshold=0.8, 
        visualize=True  # Disabled for batch to prevent popup windows
    )

    # Porosity check
    clean_json_list, raw_pore_data = porosity_check(
        image_path=str(image_path), 
        model_path=str(model_path), 
        px_to_mm=0.105, 
        plate_thickness_s=10.0,
        visualize=False  # Disabled for batch to prevent popup windows
    )

    # Cracks check
    crack_boxes_list, crack_masks_list, all_crack_detections, weld_mask = cracks_check(
        image_path=str(image_path), 
        model_path=str(model_path)
    )

    # 2. Format JSON for VLM Consumption
    discontinuity_data = [{'discontinuity': disc_bool}]
    final_json = discontinuity_data + all_crack_detections + clean_json_list
    
    for item in final_json:
        item.pop('confidence', None)
        item.pop('class', None)
        if 'box' in item:
            b = item['box']
            item['bbox'] = [int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])]
            item.pop('box', None)
    
    # Save image-specific JSON
    image_json_path = json_dir / f"{image_path.stem}.json"
    with open(image_json_path, 'w') as f:
        json.dump(final_json, f, indent=4)

    # 3. Optional VLM Step (Uncomment if needed)
    auditor = WeldAuditor()
    report_v, report_g = auditor.run_single_audit(image_path, image_json_path)
    # report_v, report_g = "VLM analysis skipped", "VLM analysis skipped"

    # 4. Generate Visual Composition
    output_filename = report_dir / f"{image_path.stem}_final_audit.jpg"
    create_comparison_composition(
        image_path=image_path,
        report_v_text=report_v,
        report_g_text=report_g,
        output_path=output_filename,
        line_params=line_params, 
        discontinuity_masks=refined_masks, 
        weld_mask=weld_mask, 
        porosity_data=raw_pore_data,
        crack_masks=crack_masks_list
    )
    print(f"Finished: {output_filename.name}")

def main():
    # Setup Paths
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent.parent
    
    IMAGE_DIR = PROJECT_ROOT / "data" / "img"
    MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
    
    # Directories for results
    JSON_OUT_DIR = PROJECT_ROOT / "data" / "json_output"
    REPORT_OUT_DIR = PROJECT_ROOT / "reports" / "vlm_results"
    
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

    print(f"Found {len(all_images)} images. Starting batch processing...")

    # Batch Loop
    for target_image in all_images:
        try:
            process_single_image(
                target_image, 
                MODEL_PATH, 
                JSON_OUT_DIR, 
                REPORT_OUT_DIR
            )
        except Exception as e:
            print(f"FAILED to process {target_image.name}: {e}")

    print(f"\nSUCCESS: Batch processing complete. Check {REPORT_OUT_DIR} for results.")

if __name__ == "__main__":
    main()