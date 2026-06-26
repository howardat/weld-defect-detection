import numpy as np

from weld_pipeline.porosity_pipeline import build_weld_mask, postprocess_weld_mask


def test_postprocess_fills_enclosed_hole():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 20:80] = 255
    mask[45:55, 45:55] = 0          # hole in the middle
    filled = postprocess_weld_mask(mask)
    assert filled[50, 50] == 255    # hole filled


def test_postprocess_empty_stays_empty():
    mask = np.zeros((50, 50), np.uint8)
    assert postprocess_weld_mask(mask).sum() == 0


class _FakeTensor:
    """Mimics an ultralytics tensor: box.xyxy[0].cpu().numpy()."""
    def __init__(self, arr): self._arr = np.asarray(arr, dtype=np.float32)
    def cpu(self): return self
    def numpy(self): return self._arr


class _FakeBox:
    def __init__(self, xyxy): self.xyxy = [_FakeTensor(xyxy)]


class _FakeResult:
    masks = None
    def __init__(self, boxes): self.boxes = [_FakeBox(b) for b in boxes]


class _FakeModel:
    def predict(self, img, conf, classes, verbose):
        return [_FakeResult([[10, 10, 40, 40]])]


def test_build_weld_mask_from_boxes():
    img = np.zeros((60, 60, 3), np.uint8)
    mask = build_weld_mask(img, _FakeModel(), weld_conf=0.01)
    assert mask[25, 25] == 255      # inside the box
    assert mask[5, 5] == 0          # outside the box
