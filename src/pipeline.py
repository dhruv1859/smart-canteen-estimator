"""
pipeline.py
─────────────────────────────────────────────────────────────────────────────
Ties together:  preprocessor → detector → zone_manager → heatmap → logger.

Usage
-----
    from src.pipeline import CanteenPipeline

    pipe   = CanteenPipeline()
    result = pipe.process_frame(bgr_frame)

    # result keys:
    #   annotated_frame  – BGR frame with boxes, zones, heatmap overlay
    #   zone_counts      – dict[zone_name -> int]
    #   total            – int
    #   crowd_level      – "Low" | "Moderate" | "High"
    #   heatmap_snapshot – standalone BGR heatmap image
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any

from src.preprocessor import preprocess
from src.detector      import PersonDetector
from src.zone_manager  import ZoneManager
from src.heatmap       import HeatmapGenerator
from src.logger        import OccupancyLogger
from src.config        import YOLO_MODEL, YOLO_CONF


class CanteenPipeline:
    """
    Full end-to-end inference pipeline for one canteen camera feed.

    Parameters
    ----------
    model_path  : YOLOv8 weights file (default from config).
    conf        : Detection confidence threshold.
    zone_defs   : Optional list of zone dicts; falls back to DEFAULT_ZONES.
    log_path    : CSV log file path.
    enable_log  : Whether to write CSV logs.
    """

    def __init__(self,
                 model_path : str  = YOLO_MODEL,
                 conf       : float = YOLO_CONF,
                 zone_defs  = None,
                 log_path   : str  = "logs/occupancy_log.csv",
                 enable_log : bool = True):

        self.detector  = PersonDetector(model_path=model_path, conf=conf)
        self.zones     = ZoneManager(zone_defs=zone_defs)
        self.heatmap   = HeatmapGenerator()
        self.logger    = OccupancyLogger(path=log_path) if enable_log else None
        self.frame_no  = 0

    # ── main entry point ──────────────────────────────────────────────────────

    def process_frame(self, raw_frame: np.ndarray,
                      show_heatmap: bool = True) -> Dict[str, Any]:
        """
        Run the full pipeline on a single raw BGR frame.
        Returns a dict with annotated frames and counts.
        """
        self.frame_no += 1

        # 1. Pre-process
        frame = preprocess(raw_frame)

        # 2. Detect persons
        detections = self.detector.detect(frame)

        # 3. Zone counts
        zone_counts = self.zones.count(detections)
        total       = len(detections)
        level       = self.zones.crowd_level(total)

        # 4. Heatmap update
        self.heatmap.update(detections)

        # 5. Annotate frame
        annotated = frame.copy()
        if show_heatmap:
            annotated = self.heatmap.overlay(annotated)
        annotated = self.zones.draw_zones(annotated)
        annotated = self.zones.draw_detections(annotated, detections)
        annotated = self.zones.draw_crowd_level(annotated, total, level)

        # 6. Log
        if self.logger:
            self.logger.log(zone_counts, total, level)

        return {
            "annotated_frame" : annotated,
            "plain_frame"     : frame,
            "zone_counts"     : zone_counts,
            "total"           : total,
            "crowd_level"     : level,
            "heatmap_snapshot": self.heatmap.snapshot(),
            "detections"      : detections,
            "frame_no"        : self.frame_no,
        }

    def process_video(self, video_path: str, show_heatmap: bool = True):
        """
        Generator: yields result dicts for every frame of a video file.
        Caller is responsible for display / saving.
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield self.process_frame(frame, show_heatmap=show_heatmap)
        finally:
            cap.release()

    def process_webcam(self, cam_index: int = 0, show_heatmap: bool = True):
        """
        Generator: yields result dicts from a live webcam.
        """
        import cv2
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise IOError(f"Cannot open webcam index {cam_index}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield self.process_frame(frame, show_heatmap=show_heatmap)
        finally:
            cap.release()

    def reset_heatmap(self):
        """Clear accumulated heatmap data."""
        self.heatmap.reset()

