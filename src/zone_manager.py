"""
zone_manager.py
─────────────────────────────────────────────────────────────────────────────
Zone definitions and per-zone people counting.

A zone is a named polygon drawn on the frame.
cv2.pointPolygonTest checks whether a detection's bounding-box centre
lies inside the zone polygon.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from src.config import DEFAULT_ZONES, CROWD_THRESHOLDS, FONT, FONT_SCALE, FONT_THICK
from src.detector import Detection


@dataclass
class Zone:
    name   : str
    color  : Tuple[int, int, int]       # BGR
    points : List[Tuple[int, int]]
    count  : int = field(default=0, init=False)

    @property
    def np_points(self) -> np.ndarray:
        return np.array(self.points, dtype=np.int32)

    def contains(self, point: Tuple[int, int]) -> bool:
        """Return True if 'point' lies inside (or on the edge of) this polygon."""
        result = cv2.pointPolygonTest(self.np_points, point, measureDist=False)
        return result >= 0


class ZoneManager:
    """
    Manages a collection of Zone objects and counts people per zone.
    """

    def __init__(self, zone_defs: List[dict] | None = None):
        zone_defs = zone_defs or DEFAULT_ZONES
        self.zones: List[Zone] = [
            Zone(name=z["name"], color=z["color"], points=z["points"])
            for z in zone_defs
        ]

    # ── counting ─────────────────────────────────────────────────────────────

    def count(self, detections: List[Detection]) -> Dict[str, int]:
        """
        Assign each detection to its zone(s) and return per-zone counts.
        One detection can increment multiple overlapping zones.
        """
        for zone in self.zones:
            zone.count = 0

        for det in detections:
            cx, cy = det.center
            for zone in self.zones:
                if zone.contains((cx, cy)):
                    zone.count += 1

        return {z.name: z.count for z in self.zones}

    def crowd_level(self, total_count: int) -> str:
        """Classify overall crowd as Low / Moderate / High."""
        for label, (lo, hi) in CROWD_THRESHOLDS.items():
            if lo <= total_count <= hi:
                return label
        return "High"

    # ── drawing ───────────────────────────────────────────────────────────────

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw zone outlines and labels onto the frame (in-place)."""
        for zone in self.zones:
            pts = zone.np_points.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True,
                          color=zone.color, thickness=2)

            # Semi-transparent fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], zone.color)
            frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)

            # Zone label + count
            x0, y0 = zone.points[0]
            label  = f"{zone.name}: {zone.count}"
            cv2.putText(frame, label, (x0 + 4, y0 + 20),
                        FONT, FONT_SCALE, zone.color, FONT_THICK,
                        cv2.LINE_AA)
        return frame

    def draw_detections(self, frame: np.ndarray,
                        detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and confidence scores."""
        for det in detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          color=(0, 255, 255), thickness=2)
            label = f"{det.conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        FONT, FONT_SCALE, (0, 255, 255), FONT_THICK,
                        cv2.LINE_AA)
        return frame

    def draw_crowd_level(self, frame: np.ndarray,
                         total: int, level: str) -> np.ndarray:
        """Draw overall count + crowd level in the top-left corner."""
        colour_map = {"Low": (0, 200, 0), "Moderate": (0, 165, 255), "High": (0, 0, 220)}
        colour = colour_map.get(level, (255, 255, 255))
        text   = f"Total: {total}  |  Crowd: {level}"
        cv2.putText(frame, text, (10, 460),
                    FONT, 0.65, colour, 2, cv2.LINE_AA)
        return frame

