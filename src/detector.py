"""
detector.py
─────────────────────────────────────────────────────────────────────────────
YOLOv8-based person detector.
Returns a list of Detection namedtuples: (box, conf, class_id).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List
from ultralytics import YOLO

from src.config import YOLO_MODEL, YOLO_CONF, YOLO_CLASS_ID


@dataclass
class Detection:
    """Single detected person."""
    box   : List[int]   # [x1, y1, x2, y2]
    conf  : float
    cls   : int
    center: tuple = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.box
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


class PersonDetector:
    """
    Wraps YOLOv8 and filters for the 'person' class.

    Parameters
    ----------
    model_path : str   Path/name of the YOLO weights file.
    conf       : float Minimum confidence to keep a detection.
    """

    def __init__(self,
                 model_path: str = YOLO_MODEL,
                 conf: float = YOLO_CONF):
        self.conf  = conf
        self.model = YOLO(model_path)
        print(f"[Detector] Loaded {model_path}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a BGR frame.
        Returns a list of Detection objects for every person found.
        """
        results = self.model(frame, conf=self.conf, classes=[YOLO_CLASS_ID],
                             verbose=False)
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf   = float(box.conf[0].cpu())
                cls    = int(box.cls[0].cpu())
                detections.append(Detection(box=coords, conf=conf, cls=cls))
        return detections

    def evaluate(self,
                 gt_boxes   : List[List[int]],
                 pred_boxes : List[List[int]],
                 iou_thresh : float = 0.5) -> dict:
        """
        Compute precision, recall, and F1 on a set of GT vs predicted boxes.
        Uses IoU ≥ iou_thresh to call a prediction a true positive.
        """
        tp = 0
        matched_gt = set()
        for pb in pred_boxes:
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                if _iou(pb, gb) >= iou_thresh:
                    tp += 1
                    matched_gt.add(gi)
                    break
        fp        = len(pred_boxes) - tp
        fn        = len(gt_boxes)  - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        return {"precision": precision, "recall": recall, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn}


# ── helpers ──────────────────────────────────────────────────────────────────

def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)
    return inter / union

