"""Validated selection preferences, independent of camera or Qt state."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
import tempfile
from typing import ClassVar


@dataclass(frozen=True)
class SelectionSettings:
    window_seconds: float = 3.0
    minimum_score: float = 1.5
    stable_seconds: float = .3
    stable_multiplier: float = 2.0
    movement_speed_limit: float = .04
    release_cells: int = 2
    tracking_grace_seconds: float = .5

    RANGES: ClassVar[dict[str, tuple[float, float]]] = {
        "window_seconds": (.5, 10.0),
        "minimum_score": (.1, 50.0),
        "stable_seconds": (.05, 2.0),
        "stable_multiplier": (1.0, 5.0),
        "movement_speed_limit": (.005, .5),
        "release_cells": (0, 5),
        "tracking_grace_seconds": (.1, 2.0),
    }

    def __post_init__(self):
        for field, (lower, upper) in self.RANGES.items():
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not lower <= value <= upper
                    or field == "release_cells" and not isinstance(value, Integral)):
                raise ValueError(f"Invalid selection setting: {field}")
        if self.minimum_score > self.window_seconds * self.stable_multiplier:
            raise ValueError("Weighted lock threshold exceeds the window's maximum score")

    @classmethod
    def from_mapping(cls, saved) -> SelectionSettings:
        """Recover each valid field even when other saved fields are malformed."""
        defaults = cls()
        values = asdict(defaults)
        if not isinstance(saved, Mapping):
            return defaults
        for field, (lower, upper) in cls.RANGES.items():
            value = saved.get(field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not lower <= value <= upper
                    or field == "release_cells" and not isinstance(value, Integral)):
                continue
            values[field] = int(value) if field == "release_cells" else float(value)
        maximum = values["window_seconds"] * values["stable_multiplier"]
        if values["minimum_score"] > maximum:
            values["minimum_score"] = min(defaults.minimum_score, maximum)
        return cls(**values)

    def to_mapping(self):
        return asdict(self)


def save_selection_settings(destination: Path, settings: SelectionSettings) -> None:
    """Merge this section into preferences, then atomically replace the file."""
    try:
        preferences = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(preferences, dict):
            preferences = {}
    except (OSError, ValueError):
        preferences = {}
    preferences["selection_settings"] = settings.to_mapping()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                         prefix=".selection-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(preferences, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
