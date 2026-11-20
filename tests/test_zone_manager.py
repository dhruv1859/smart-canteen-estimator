"""
tests/test_zone_manager.py
Unit tests for zone-based counting.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.zone_manager import Zone, ZoneManager
from src.detector import Detection


def _make_detection(cx, cy):
    """Create a fake Detection with a given bounding-box centre."""
    half = 20
    det = Detection(box=[cx-half, cy-half, cx+half, cy+half], conf=0.9, cls=0)
    assert det.center == (cx, cy)
    return det


class TestZone:
    def test_contains_inside(self):
        z = Zone("Test", (0, 255, 0), [(0, 0), (100, 0), (100, 100), (0, 100)])
        assert z.contains((50, 50))

    def test_contains_outside(self):
        z = Zone("Test", (0, 255, 0), [(0, 0), (100, 0), (100, 100), (0, 100)])
        assert not z.contains((200, 200))

    def test_on_boundary(self):
        z = Zone("Test", (0, 255, 0), [(0, 0), (100, 0), (100, 100), (0, 100)])
        # pointPolygonTest returns >= 0 for on-edge
        assert z.contains((0, 0))


class TestZoneManager:
    def _manager(self):
        zones = [
            {"name": "A", "color": (0, 255, 0),
             "points": [(0, 0), (200, 0), (200, 200), (0, 200)]},
            {"name": "B", "color": (255, 0, 0),
             "points": [(200, 0), (400, 0), (400, 200), (200, 200)]},
        ]
        return ZoneManager(zone_defs=zones)

    def test_count_zone_a(self):
        zm  = self._manager()
        det = [_make_detection(100, 100)]   # inside zone A
        counts = zm.count(det)
        assert counts["A"] == 1
        assert counts["B"] == 0

    def test_count_zone_b(self):
        zm  = self._manager()
        det = [_make_detection(300, 100)]   # inside zone B
        counts = zm.count(det)
        assert counts["B"] == 1
        assert counts["A"] == 0

    def test_empty_detections(self):
        zm     = self._manager()
        counts = zm.count([])
        assert all(v == 0 for v in counts.values())

    def test_crowd_level_low(self):
        zm = self._manager()
        assert zm.crowd_level(3) == "Low"

    def test_crowd_level_moderate(self):
        zm = self._manager()
        assert zm.crowd_level(10) == "Moderate"

    def test_crowd_level_high(self):
        zm = self._manager()
        assert zm.crowd_level(20) == "High"

    def test_draw_zones_shape(self):
        import numpy as np
        zm    = self._manager()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out   = zm.draw_zones(frame)
        assert out.shape == (480, 640, 3)

