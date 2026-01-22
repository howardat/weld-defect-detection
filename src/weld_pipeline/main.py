from weld_pipeline.visualization import create_comparison_composition
from weld_pipeline.vlm import WeldAuditor, run_comparative_batch
from weld_pipeline.porosity_check import porosity_check
from weld_pipeline.discontinuity_check import discontinuity_check
from weld_pipeline.cracks_check import cracks_check

from pathlib import Path
import json

def main():
    # 1. Setup Universal Paths
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent.parent
    
    # We keep these as folders
    IMAGE_DIR = PROJECT_ROOT / "data" / "img"
    MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
    JSON_OUT = PROJECT_ROOT / "data" / "json_output" / "merge.json"
    REPORT_OUT_DIR = PROJECT_ROOT / "reports" / "vlm_results"
    
    # 2. Pick the first image in the folder to analyze
    # This prevents the PermissionError by selecting a FILE, not the FOLDER
    all_images = list(IMAGE_DIR.glob("*.jpeg")) + list(IMAGE_DIR.glob("*.jpg"))
    
    if not all_images:
        print(f"No images found in {IMAGE_DIR}")
        return
        
    # Use the first image found for the specific checks
    target_image = all_images[0]
    print(f"Analyzing: {target_image.name}")

    # 3. Run Checks (using the FILE path)
    has_discontinuity  = [{'discontinuity': discontinuity_check(
        image_path=str(target_image), 
        model_path=str(MODEL_PATH), 
        threshold=0.8, 
        visualize=True
    )}]

    disc_bool, refined_masks, line_params = discontinuity_check(
        image_path=str(target_image), 
        model_path=str(MODEL_PATH), 
        threshold=0.8, 
        visualize=True
    )

    clean_json_list, raw_pore_data = porosity_check(
        image_path=str(target_image), 
        model_path=str(MODEL_PATH), 
        px_to_mm=0.105, 
        plate_thickness_s=10.0
    )

    crack_boxes_list, crack_masks_list, all_crack_detections = cracks_check(
        image_path=str(target_image), 
        model_path=str(MODEL_PATH)
    )

    final_json = has_discontinuity + all_crack_detections + clean_json_list
    for item in final_json:
        item.pop('confidence', None)
        item.pop('class', None)

        if 'box' in item:
            b = item['box']
            # Convert floats to ints and flatten the box structure
            item['bbox'] = [int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])]
            # Remove the old high-precision box dictionary
            item.pop('box', None)

    with open('../../data/json_output/merge.json', 'w') as f:
        json.dump(final_json, f, indent=4)  

    # 5. Run VLM (Gemma 3)
    auditor = WeldAuditor()
    # This now works because we added the method to vlm.py
    report_v, report_g = auditor.run_single_audit(target_image, JSON_OUT)

    # 6. Final Visualization
    output_filename = REPORT_OUT_DIR / f"{target_image.stem}_final_audit.jpg"
    create_comparison_composition(
        image_path=target_image,
        report_v_text=report_v,
        report_g_text=report_g,
        output_path=output_filename,
        line_params=line_params, 
        discontinuity_masks=refined_masks, 
        weld_mask=None, 
        porosity_data=raw_pore_data,
        crack_masks=crack_masks_list
    )

    print(f"\nSUCCESS: Full pipeline complete.")
    print(f"Final Report: {output_filename}")

if __name__ == "__main__":
    main()