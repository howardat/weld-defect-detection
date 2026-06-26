import numpy as np

from weld_pipeline.porosity_data import build_gt_mask


def test_build_gt_mask_fills_polygon():
    anns = [{
        "image_id": 0,
        "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],  # square
    }]
    mask = build_gt_mask(anns, image_id=0, h=50, w=50)
    assert mask[20, 20] == 255
    assert mask[0, 0] == 0


def test_build_gt_mask_ignores_other_images():
    anns = [{
        "image_id": 5,
        "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
    }]
    mask = build_gt_mask(anns, image_id=0, h=50, w=50)
    assert mask.sum() == 0


def test_build_gt_mask_handles_multiple_polygons_per_annotation():
    anns = [{
        "image_id": 0,
        "segmentation": [
            [2, 2, 8, 2, 8, 8, 2, 8],
            [40, 40, 46, 40, 46, 46, 40, 46],
        ],
    }]
    mask = build_gt_mask(anns, image_id=0, h=50, w=50)
    assert mask[5, 5] == 255
    assert mask[43, 43] == 255
