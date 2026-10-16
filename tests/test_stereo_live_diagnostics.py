"""Finite, bounded numeric history must never make tracking depend on disk."""
import json

from kon.stereo_live_diagnostics import LiveStereoDiagnostics


class Calibration:
    def to_dict(self):
        return {"id": "synthetic-rig"}


def test_reports_are_rate_limited_bounded_and_exact(tmp_path):
    path = tmp_path / "nested" / "report.json"
    log = LiveStereoDiagnostics(path, Calibration())
    report = {"hand_primary_pixels": [[10.25, 20.5]], "failure_stages": ["hand"]}
    assert log.record(1., report)
    report["hand_primary_pixels"][0][0] = 99.
    assert not log.record(1.1, report)
    for index in range(1, 64):
        assert log.record(1. + index * .5, report)
    assert not log.record(100., report)
    saved = json.loads(path.read_text())
    assert saved["calibration"] == {"id": "synthetic-rig"}
    assert len(saved["samples"]) == 64
    assert saved["samples"][0]["hand_primary_pixels"] == [[10.25, 20.5]]
    assert not list(path.parent.glob("*.tmp"))


def test_invalid_or_unwritable_report_never_overwrites_history(tmp_path):
    path = tmp_path / "report.json"
    log = LiveStereoDiagnostics(path, Calibration())
    assert log.record(1., {"value": 1})
    old = path.read_bytes()
    assert not log.record(2., {"value": float("nan")})
    assert path.read_bytes() == old and len(log.samples) == 1
    assert log.error
    log.path = tmp_path  # replacing a directory cannot succeed
    assert not log.record(3., {"value": 2})
    assert len(log.samples) == 1 and log.error
