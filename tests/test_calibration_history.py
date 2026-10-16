"""Local numerical attempt history must survive retries without media files."""
import json

import pytest

from kon.calibration_diagnostics import analyze_calibration
from kon.calibration_history import load_last_attempt, save_attempt


TARGETS = [(x, y) for y in (.18, .5, .82) for x in (.18, .5, .82)]


def test_attempts_remain_independent_and_last_pointer_has_full_numerical_report(tmp_path):
    raw = [(x * .2, y * .2) for x, y in TARGETS]
    diagnostics = analyze_calibration(raw, TARGETS)
    config = {"fov": 82.15, "camera": 0}
    first = save_attempt(tmp_path, raw, TARGETS, diagnostics, config)
    second = save_attempt(tmp_path, raw, TARGETS, diagnostics, config)
    assert first != second and first.is_file() and second.is_file()
    saved = load_last_attempt(tmp_path, config)
    assert saved == json.loads(second.read_text())
    assert saved["fit"]["status"] == "passed"
    assert len(saved["raw_points"]) == len(saved["fit"]["predicted_points"]) == 9
    assert load_last_attempt(tmp_path, {"fov": 60, "camera": 0}) is None
    assert not list((tmp_path / ".kon").rglob("*.tmp"))


def test_degenerate_fit_can_be_persisted_without_nonfinite_json(tmp_path):
    raw = [(.1, .1)] * 9
    diagnostics = analyze_calibration(raw, TARGETS)
    save_attempt(tmp_path, raw, TARGETS, diagnostics, {})
    report = load_last_attempt(tmp_path, {})
    assert report["fit"]["status"] == "insufficient_spread"
    assert report["fit"]["rms_error"] is None
    assert report["fit"]["predicted_points"] == []


@pytest.mark.parametrize("contents", ["[]", "null", "not JSON", '{}', '{"version":2,"config":{}}'])
def test_missing_or_malformed_attempt_is_ignored(tmp_path, contents):
    assert load_last_attempt(tmp_path, {}) is None
    (tmp_path / ".kon").mkdir()
    (tmp_path / ".kon/calibration-last-attempt.json").write_text(contents)
    assert load_last_attempt(tmp_path, {}) is None
