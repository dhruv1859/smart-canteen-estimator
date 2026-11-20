"""
preprocessor.py
─────────────────────────────────────────────────────────────────────────────
Frame pre-processing utilities:
  • resize to target resolution
  • colour-space conversion (BGR → RGB for YOLO / for display)
  • noise reduction (Gaussian blur)
  • contrast enhancement (CLAHE on L-channel in LAB space)
"""

import cv2
import numpy as np
from src.config import FRAME_WIDTH, FRAME_HEIGHT


def resize_frame(frame: np.ndarray,
                 width: int = FRAME_WIDTH,
                 height: int = FRAME_HEIGHT) -> np.ndarray:
    """Resize frame to the configured resolution."""
    if frame is None:
        raise ValueError("resize_frame received None")
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV default) → RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert RGB → BGR."""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def reduce_noise(frame: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur to reduce sensor / compression noise.
    ksize must be odd.
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


def enhance_contrast(frame: np.ndarray,
                     clip_limit: float = 2.0,
                     tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE contrast enhancement on the Lightness channel (LAB colour space).
    Helps detection in dim indoor / canteen lighting.
    """
    lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq  = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocess(frame: np.ndarray,
               do_resize: bool = True,
               do_denoise: bool = True,
               do_enhance: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline applied to a single raw frame.
    Returns a BGR uint8 NumPy array ready for detection.
    """
    if do_resize:
        frame = resize_frame(frame)
    if do_enhance:
        frame = enhance_contrast(frame)
    if do_denoise:
        frame = reduce_noise(frame)
    return frame

