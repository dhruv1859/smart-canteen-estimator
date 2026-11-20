"""
logger.py
─────────────────────────────────────────────────────────────────────────────
Time-series logger: writes per-zone counts + crowd level to CSV at a
configurable interval so occupancy trends can be charted later.
"""

from __future__ import annotations
import csv
import os
import time
from datetime import datetime
from typing import Dict

from src.config import LOG_CSV_PATH, LOG_INTERVAL_S

_FIELDNAMES = ["timestamp", "total", "crowd_level",
               "Seating Area", "Serving Counter", "Billing Counter", "Queue / Exit"]


class OccupancyLogger:
    """
    Thread-safe (single-writer) CSV logger.

    Parameters
    ----------
    path           : CSV output path.
    interval_secs  : Minimum seconds between successive rows.
    """

    def __init__(self,
                 path         : str = LOG_CSV_PATH,
                 interval_secs: int = LOG_INTERVAL_S):
        self.path          = path
        self.interval      = interval_secs
        self._last_written = 0.0
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Write header if file is new / empty
        write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES,
                                    extrasaction="ignore")
            if write_header:
                writer.writeheader()

    def log(self,
            zone_counts  : Dict[str, int],
            total        : int,
            crowd_level  : str) -> bool:
        """
        Write a row if at least `interval` seconds have elapsed.
        Returns True if a row was actually written.
        """
        now = time.time()
        if now - self._last_written < self.interval:
            return False

        row = {
            "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total"       : total,
            "crowd_level" : crowd_level,
        }
        row.update(zone_counts)

        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_FIELDNAMES,
                           extrasaction="ignore").writerow(row)

        self._last_written = now
        return True

    # ── reading back ─────────────────────────────────────────────────────────

    def load(self) -> list[dict]:
        """Return all logged rows as a list of dicts."""
        if not os.path.isfile(self.path):
            return []
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))

