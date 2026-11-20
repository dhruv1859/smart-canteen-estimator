"""
tests/test_preprocessor.py
Unit tests for the preprocessing pipeline.
Run:  pytest tests/
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessor import resize_frame, reduce_noise, enhance_contrast, preprocess
from src.config import FRAME_WIDTH, FRAME_HEIGHT


def _dummy(h=480, w=640, c=3):
    """Create a reproducible dummy BGR frame."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, c), dtype=np.uint8)


class TestResize:
    def test_output_shape(self):
        frame = _dummy(1080, 1920)
        out   = resize_frame(frame)
        assert out.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)

    def test_none_raises(self):
        with pytest.raises((ValueError, Exception)):
            resize_frame(None)


class TestDenoise:
    def test_shape_preserved(self):
        frame = _dummy()
        out   = reduce_noise(frame)
        assert out.shape == frame.shape

    def test_even_ksize_auto_corrected(self):
        frame = _dummy()
        out   = reduce_noise(frame, ksize=4)      # even → corrected to 5
        assert out.shape == frame.shape


class TestEnhance:
    def test_shape_preserved(self):
        frame = _dummy()
        out   = enhance_contrast(frame)
        assert out.shape == frame.shape

    def test_dtype_preserved(self):
        frame = _dummy()
        out   = enhance_contrast(frame)
        assert out.dtype == np.uint8


class TestPreprocess:
    def test_full_pipeline_shape(self):
        frame = _dummy(1080, 1920)
        out   = preprocess(frame)
        assert out.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)

    def test_no_resize(self):
        frame = _dummy(100, 100)
        out   = preprocess(frame, do_resize=False)
        assert out.shape[:2] == (100, 100)

