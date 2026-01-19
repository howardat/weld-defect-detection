from weld_pipeline.vlm import run_comparative_batch
from weld_pipeline.porosity_check import porosity_check
from weld_pipeline.discontinuity_check import discontinuity_check
from weld_pipeline.cracks_check import cracks_check

import json

def main():
    IMAGE_PATH ='../../data/crack.jpeg'
    MODEL_PATH = '../../models/best.pt'

    has_discontinuity = [{'discontinuity': discontinuity_check(
        image_path=IMAGE_PATH, 
        model_path=MODEL_PATH, 
        threshold=0.8, 
        visualize=False
    )}]

    is_violation = porosity_check(
        image_path=IMAGE_PATH, 
        model_path=MODEL_PATH, 
        px_to_mm=0.105, 
        plate_thickness_s=10.0
    )

    results = cracks_check(
        image_path=IMAGE_PATH, 
        model_path=MODEL_PATH
    )

    final_json = has_discontinuity + results + is_violation
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

    run_comparative_batch(
        image_dir=IMAGE_PATH,
        json_dir='../../data/json_output/merge.json',
        output_dir="../../reports/vlm_results"
    )

if __name__ == "__main__":
    main()