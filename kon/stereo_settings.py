"""Atomic local stereo rig settings, separate from monocular aiming profiles."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

from .stereo import StereoCalibration, ScreenPlane


@dataclass(frozen=True)
class StereoRig:
    calibration: StereoCalibration
    screen_plane: ScreenPlane
    screen_name: str = ""
    screen_serial: str = ""
    screen_size: tuple[int, int] = (0, 0)

    def compatible(self, primary_source, secondary_source, screen_name, screen_serial, screen_size):
        return (self.calibration.primary_source == primary_source
                and self.calibration.secondary_source == secondary_source
                and self.screen_name == screen_name and self.screen_size == tuple(screen_size)
                and (not self.screen_serial or not screen_serial or self.screen_serial == screen_serial))

    def save(self, path: Path):
        payload = {"version": 1, "calibration": self.calibration.to_dict(),
                   "screen_plane": self.screen_plane.to_dict(), "screen_name": self.screen_name,
                   "screen_serial": self.screen_serial, "screen_size": list(self.screen_size)}
        # Serialize and validate before replacing a working rig.
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("雙鏡頭設定版本不相容")
        size = payload.get("screen_size")
        if (not isinstance(size, list) or len(size) != 2
                or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in size)):
            raise ValueError("雙鏡頭設定的螢幕尺寸無效")
        return cls(StereoCalibration.from_dict(payload["calibration"]),
                   ScreenPlane.from_dict(payload["screen_plane"]),
                   str(payload.get("screen_name", "")), str(payload.get("screen_serial", "")), tuple(size))
