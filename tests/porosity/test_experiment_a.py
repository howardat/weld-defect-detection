import cv2
import numpy as np

from weld_pipeline.porosity_data import ImageRecord
from weld_pipeline.porosity_pipeline import PoreParams
from experiments.run_experiment_a import evaluate_predictions


def _record():
    size = 200
    gray = np.full((size, size), 200, np.uint8)
    gt = np.zeros((size, size), np.uint8)
    cv2.circle(gray, (100, 100), 14, 25, -1)
    cv2.circle(gt, (100, 100), 14, 255, -1)
    weld = np.full((size, size), 255, np.uint8)
    return ImageRecord(0, "x.jpg", gray, weld, gt)


def test_evaluate_predictions_reports_f1_per_record():
    rec = _record()
    good = PoreParams(51, 10, 3, 0, 0.5, 20)
    rows = evaluate_predictions([rec], [good])
    assert len(rows) == 1
    assert rows[0]["file_name"] == "x.jpg"
    assert 0.0 <= rows[0]["f1"] <= 1.0
