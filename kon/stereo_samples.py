"""Validated, atomic stereo corner checkpoints. Never store camera images."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from uuid import UUID, uuid4

import numpy as np


MAX_SAMPLES = 500
_MAX_FILE_BYTES = 8 * 1024 * 1024


def _index(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value <= 65535:
        raise ValueError("取樣檔的鏡頭編號無效。")
    return int(value)


def _text(value, label):
    if not isinstance(value, str) or len(value) > 2048 or "\x00" in value:
        raise ValueError(f"取樣檔的{label}無效。")
    return value


def _size(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2 or
            any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 32 <= v <= 16384 for v in value)):
        raise ValueError("取樣檔的影像尺寸無效。")
    return tuple(int(v) for v in value)


def validate_sample(value):
    """Copy exactly 9×6 finite in-frame corners and source metadata."""
    if not isinstance(value, dict):
        raise ValueError("棋盤取樣資料無效。")
    result = {}
    try:
        for side in ("primary", "secondary"):
            size = _size(value[f"{side}_size"])
            raw = np.asarray(value[side])
            if raw.shape != (54, 2) or raw.dtype.kind not in "fiu":
                raise ValueError("棋盤取樣須包含完整 9 × 6 個內角點。")
            points = np.array(raw, dtype=np.float64, copy=True)
            if (not np.isfinite(points).all() or np.any(points < 0) or
                    np.any(points >= np.asarray(size))):
                raise ValueError("棋盤角點不是有限數值或超出影像範圍。")
            result[side] = points
            result[f"{side}_size"] = size
            result[f"{side}_id"] = _text(value[f"{side}_id"], "裝置識別")
            result[f"{side}_name"] = _text(value.get(f"{side}_name", ""), "裝置名稱")
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("棋盤取樣缺少完整角點或鏡頭資料。") from error
    if result["primary_id"] and result["primary_id"] == result["secondary_id"]:
        raise ValueError("兩台取樣鏡頭不可使用相同裝置識別。")
    return result


def _atomic_json(path, payload):
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class StereoSampleSession:
    square_mm: float
    primary_source: int
    secondary_source: int
    samples: tuple[dict, ...]
    session_id: str = field(default_factory=lambda: uuid4().hex)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if (isinstance(self.square_mm, bool) or not isinstance(self.square_mm, (int, float)) or
                not math.isfinite(self.square_mm) or not 5 <= self.square_mm <= 100):
            raise ValueError("取樣檔的棋盤格邊長須介於 5 與 100 mm。")
        primary, secondary = _index(self.primary_source), _index(self.secondary_source)
        if primary == secondary:
            raise ValueError("取樣檔的兩台鏡頭編號必須不同。")
        if not isinstance(self.samples, (list, tuple)) or not 1 <= len(self.samples) <= MAX_SAMPLES:
            raise ValueError(f"取樣檔須包含 1 至 {MAX_SAMPLES} 組棋盤角點。")
        samples = tuple(validate_sample(sample) for sample in self.samples)
        first = samples[0]
        for sample in samples[1:]:
            if any(sample[key] != first[key] for key in
                   ("primary_size", "secondary_size", "primary_id", "secondary_id")):
                raise ValueError("取樣檔混用了不同裝置或影像尺寸。")
        try:
            if not isinstance(self.session_id, str):
                raise ValueError()
            UUID(self.session_id)
            timestamp = datetime.fromisoformat(self.created_at_utc)
            if timestamp.tzinfo is None:
                raise ValueError()
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("取樣檔的工作階段或時間資訊無效。") from error
        object.__setattr__(self, "square_mm", float(self.square_mm))
        object.__setattr__(self, "primary_source", primary)
        object.__setattr__(self, "secondary_source", secondary)
        object.__setattr__(self, "samples", samples)

    def to_dict(self):
        first = self.samples[0]
        return {
            "schema_version": 1, "kind": "kon.stereo_samples",
            "units": {"square_size": "millimeters", "corners": "pixels"},
            "pattern": {"columns": 9, "rows": 6},
            "session_id": self.session_id, "created_at_utc": self.created_at_utc,
            "square_mm": self.square_mm, "sample_count": len(self.samples),
            "sources": {
                side: {"index": index, "device_id": first[f"{side}_id"],
                       "name": first[f"{side}_name"], "image_size": list(first[f"{side}_size"])}
                for side, index in (("primary", self.primary_source), ("secondary", self.secondary_source))
            },
            "samples": [{side: sample[side].tolist() for side in ("primary", "secondary")}
                        for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, payload):
        try:
            if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or
                    payload["schema_version"] != 1 or payload.get("kind") != "kon.stereo_samples" or
                    payload.get("units") != {"square_size": "millimeters", "corners": "pixels"} or
                    payload.get("pattern") != {"columns": 9, "rows": 6}):
                raise ValueError("棋盤取樣檔的版本、單位或 9 × 6 圖樣不相容。")
            raw = payload["samples"]
            count = payload["sample_count"]
            if type(count) is not int or not isinstance(raw, list) or count != len(raw):
                raise ValueError("棋盤取樣檔的組數不一致。")
            sources = payload["sources"]
            samples = []
            for entry in raw:
                sample = {}
                for side in ("primary", "secondary"):
                    source = sources[side]
                    sample[side] = entry[side]
                    sample[f"{side}_size"] = source["image_size"]
                    sample[f"{side}_id"] = source["device_id"]
                    sample[f"{side}_name"] = source.get("name", "")
                samples.append(sample)
            return cls(payload["square_mm"], sources["primary"]["index"], sources["secondary"]["index"],
                       tuple(samples), payload["session_id"], payload["created_at_utc"])
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("棋盤取樣檔缺少完整資料。") from error

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > _MAX_FILE_BYTES:
            raise ValueError("棋盤取樣檔過大。")
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, UnicodeError, RecursionError) as error:
            raise ValueError("無法讀取棋盤取樣 JSON 檔。") from error

    def save(self, path):
        payload = self.to_dict()
        # Caller-owned dictionaries can be edited after construction. Validate
        # the final snapshot before creating directories or replacing a file.
        self.from_dict(payload)
        _atomic_json(path, payload)

    def require_matching_sources(self, primary, secondary):
        """Indices can move; both CURRENT native device IDs must be verified."""
        first = self.samples[0]
        for side, actual in (("primary", primary), ("secondary", secondary)):
            saved = first[f"{side}_id"]
            current = getattr(actual, "device_id", "")
            if not saved or not current or saved != current:
                raise ValueError("取樣檔與目前兩台鏡頭的裝置識別不符或無法驗證；不可只按編號沿用。")


def save_diagnostic_report(path, diagnostics, *, sample_checkpoint=None):
    if not isinstance(diagnostics, dict):
        raise ValueError("校正數值診斷資料無效。")
    _atomic_json(path, {"schema_version": 1, "kind": "kon.stereo_diagnostics",
                        "sample_checkpoint": str(sample_checkpoint) if sample_checkpoint is not None else None,
                        "diagnostics": diagnostics})
