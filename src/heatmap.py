"""
heatmap.py
─────────────────────────────────────────────────────────────────────────────
Accumulates person detections over time and renders a crowd-density heatmap.

Each detected centre point adds a Gaussian blob to an accumulator.
The accumulator decays each frame so older crowd positions fade out.
cv2.applyColorMap + cv2.addWeighted blend the coloured heatmap onto the frame.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple

from src.config import (FRAME_WIDTH, FRAME_HEIGHT,
                        HEATMAP_ALPHA, HEATMAP_DECAY, HEATMAP_RADIUS)
from src.detector import Detection


class HeatmapGenerator:
    """
    Maintains a float32 accumulator of the same resolution as the video frame.
    Call `update()` each frame; call `overlay()` to get a coloured overlay.
    """

    def __init__(self,
                 width   : int   = FRAME_WIDTH,
                 height  : int   = FRAME_HEIGHT,
                 decay   : float = HEATMAP_DECAY,
                 radius  : int   = HEATMAP_RADIUS,
                 alpha   : float = HEATMAP_ALPHA):
        self.width  = width
        self.height = height
        self.decay  = decay
        self.radius = radius
        self.alpha  = alpha
        self.accumulator = np.zeros((height, width), dtype=np.float32)

        # Pre-build Gaussian kernel for speed
        k = radius * 2 + 1
        self._kernel = cv2.getGaussianKernel(k, radius / 2)
        self._kernel = self._kernel @ self._kernel.T    # 2-D Gaussian

    # ── update ───────────────────────────────────────────────────────────────

    def update(self, detections: List[Detection]) -> None:
        """Decay accumulator and add a Gaussian blob for every detection."""
        self.accumulator *= self.decay
        for det in detections:
            self._add_blob(det.center)

    def _add_blob(self, centre: Tuple[int, int]) -> None:
        cx, cy = centre
        r      = self.radius
        k      = self._kernel

        # Clipping to frame boundaries
        x1 = max(0, cx - r);  y1 = max(0, cy - r)
        x2 = min(self.width,  cx + r + 1)
        y2 = min(self.height, cy + r + 1)

        # Corresponding kernel slice
        kx1 = x1 - (cx - r);  ky1 = y1 - (cy - r)
        kx2 = kx1 + (x2 - x1); ky2 = ky1 + (y2 - y1)

        self.accumulator[y1:y2, x1:x2] += k[ky1:ky2, kx1:kx2]

    # ── overlay ──────────────────────────────────────────────────────────────

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Return the frame with the coloured heatmap blended in.
        Does NOT modify the original frame.
        """
        acc = self.accumulator
        if acc.max() > 0:
            norm = (acc / acc.max() * 255).astype(np.uint8)
        else:
            norm = acc.astype(np.uint8)

        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        blended  = cv2.addWeighted(coloured, self.alpha,
                                   frame,   1 - self.alpha, 0)
        return blended

    def snapshot(self) -> np.ndarray:
        """Return the current heatmap as a standalone BGR image (no overlay)."""
        acc = self.accumulator
        if acc.max() > 0:
            norm = (acc / acc.max() * 255).astype(np.uint8)
        else:
            norm = acc.astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def reset(self) -> None:
        """Clear the accumulator (e.g., between analysis sessions)."""
        self.accumulator[:] = 0

