"""Bounded local numeric reports for rejected stereo correspondences, no images."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .stereo_samples import _atomic_json


class LiveStereoDiagnostics:
    """Keep at most 64 rejected pairs per camera run, at most twice a second."""

    def __init__(self, path, calibration):
        self.path = Path(path)
        self.calibration = calibration.to_dict()
        self.samples = []
        self.last_timestamp = -math.inf
        self.error = ""

    def record(self, timestamp, report):
        if (len(self.samples) >= 64 or not math.isfinite(timestamp)
                or timestamp - self.last_timestamp < .5):
            return False
        self.last_timestamp = timestamp
        # Validate and detach before mutating the report history. Production
        # correspondences have already passed the finite image-space gates.
        try:
            sample = json.loads(json.dumps(dict(report, timestamp=timestamp), allow_nan=False))
            payload = {
                "format": "kon-stereo-live-diagnostics", "version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "calibration": self.calibration,
                "samples": [*self.samples, sample],
            }
            _atomic_json(self.path, payload)
        except (OSError, ValueError, TypeError) as exc:
            self.error = str(exc)
            return False
        self.samples.append(sample)
        self.error = ""
        return True
