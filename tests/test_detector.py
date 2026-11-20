"""
tests/test_detector.py
Unit tests for the IoU helper and evaluate() method (no GPU required).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detector import _iou, Detection


class TestIoU:
    def test_perfect_overlap(self):
        b = [0, 0, 100, 100]
        assert _iou(b, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = [0, 0, 50, 50]
        b = [100, 100, 200, 200]
        assert _iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = [0, 0, 100, 100]
        b = [50, 0, 150, 100]
        # intersection = 50×100 = 5000; union = 10000+10000-5000=15000
        assert _iou(a, b) == pytest.approx(5000 / 15000, rel=1e-5)


class TestDetection:
    def test_center_computed(self):
        det = Detection(box=[0, 0, 100, 100], conf=0.9, cls=0)
        assert det.center == (50, 50)

    def test_center_non_square(self):
        det = Detection(box=[10, 20, 50, 80], conf=0.8, cls=0)
        assert det.center == (30, 50)


import pytest

