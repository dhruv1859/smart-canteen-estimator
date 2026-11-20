"""
tests/test_heatmap.py
Unit tests for the heatmap generator.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.heatmap   import HeatmapGenerator
from src.detector  import Detection


def _det(cx, cy):
    half = 20
    return Detection(box=[cx-half, cy-half, cx+half, cy+half], conf=0.9, cls=0)


class TestHeatmap:
    def test_initial_accumulator_zero(self):
        hm = HeatmapGenerator(width=100, height=100)
        assert hm.accumulator.max() == 0.0

    def test_update_increases_accumulator(self):
        hm = HeatmapGenerator(width=100, height=100)
        hm.update([_det(50, 50)])
        assert hm.accumulator.max() > 0.0

    def test_reset_clears_accumulator(self):
        hm = HeatmapGenerator(width=100, height=100)
        hm.update([_det(50, 50)])
        hm.reset()
        assert hm.accumulator.max() == 0.0

    def test_overlay_shape(self):
        hm    = HeatmapGenerator(width=640, height=480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hm.update([_det(320, 240)])
        out = hm.overlay(frame)
        assert out.shape == (480, 640, 3)

    def test_snapshot_shape(self):
        hm = HeatmapGenerator(width=640, height=480)
        snap = hm.snapshot()
        assert snap.shape == (480, 640, 3)

    def test_decay(self):
        hm = HeatmapGenerator(width=100, height=100, decay=0.5)
        hm.update([_det(50, 50)])
        v1 = hm.accumulator.max()
        hm.update([])    # no new detections → decay only
        v2 = hm.accumulator.max()
        assert v2 < v1

    def test_boundary_blob(self):
        """Blobs near the frame edge must not raise an index error."""
        hm = HeatmapGenerator(width=100, height=100)
        hm.update([_det(0, 0), _det(99, 99)])  # corners

