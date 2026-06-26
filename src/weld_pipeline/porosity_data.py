"""Load the porosity validation dataset (images + COCO GT + YOLO weld masks)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from weld_pipeline.porosity_pipeline import build_weld_mask


def build_gt_mask(annotations: list[dict], image_id: int, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    for ann in annotations:
        if ann["image_id"] != image_id:
            continue
        for seg in ann["segmentation"]:
            pts = np.array(seg, np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    gray: np.ndarray
    weld_mask: np.ndarray
    gt_mask: np.ndarray


def load_dataset(data_dir, anno_file, seg_model, weld_conf: float = 0.01) -> list[ImageRecord]:
    data_dir = Path(data_dir)
    with open(anno_file) as f:
        coco = json.load(f)
    images_meta = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]

    records: list[ImageRecord] = []
    for img_id, meta in images_meta.items():
        img_path = data_dir / meta["file_name"]
        if not img_path.exists():
            print(f"  [WARN] missing: {img_path.name} — skipping")
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [WARN] unreadable: {img_path.name} — skipping")
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        records.append(ImageRecord(
            image_id=img_id,
            file_name=meta["file_name"],
            gray=gray,
            weld_mask=build_weld_mask(img_rgb, seg_model, weld_conf),
            gt_mask=build_gt_mask(annotations, img_id, h, w),
        ))
    return records
