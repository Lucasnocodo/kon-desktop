"""Stereo persistence and nine-point mapping isolation without any devices."""

from dataclasses import replace
import json

import numpy as np
import pytest

from kon.calibration_store import compatible_configuration, load_calibration, save_calibration
from kon.geometry import ScreenCalibration
from kon.stereo import CameraCalibration, ScreenPlane, StereoCalibration
from kon.stereo_settings import StereoRig


def calibration(*, primary=0, secondary=1, baseline=.20, primary_id="usb", secondary_id="phone"):
    intrinsics = CameraCalibration((1280, 720),
                                  np.array([[900, 0, 640], [0, 900, 360], [0, 0, 1.]]),
                                  np.zeros(5))
    return StereoCalibration(intrinsics, intrinsics, np.eye(3), [-baseline, 0, 0],
                             primary, secondary, primary_id, secondary_id,
                             rms=.4, primary_rms=.3, secondary_rms=.35, sample_count=18)


def plane(*, camera_x=.30):
    return ScreenPlane.from_measurements(.60, .34, camera_x, -.025, .02)


def rig():
    return StereoRig(calibration(), plane(), "Display", "display-serial", (1920, 1080))


def mono_config():
    return {"eye": "right", "camera": 0, "fov": 70., "screen": [1920, 1080],
            "screen_name": "Display", "screen_serial": "display-serial",
            "machine_id": "test-machine", "screen_origin": [0, 0]}


def stereo_config():
    return mono_config() | {"acquisition_mode": "stereo", "secondary_camera": 1,
                            "stereo_calibration_id": calibration().calibration_id,
                            "screen_plane_id": plane().plane_id}


def mapping(offset=0.):
    targets = [(x, y) for y in (0, .5, 1) for x in (0, .5, 1)]
    raw = [(x + offset, y - offset) for x, y in targets]
    return ScreenCalibration.fit(raw, targets)


def test_rig_roundtrip_retains_metric_geometry_and_source_identity(tmp_path):
    original = rig()
    path = tmp_path / "stereo.json"
    original.save(path)
    loaded = StereoRig.load(path)
    assert loaded.calibration.calibration_id == original.calibration.calibration_id
    assert loaded.screen_plane.plane_id == original.screen_plane.plane_id
    assert loaded.calibration.primary_id == "usb"
    assert loaded.calibration.secondary_id == "phone"
    np.testing.assert_allclose(loaded.calibration.translation, [-.2, 0, 0])
    assert loaded.screen_size == (1920, 1080)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("contents", ["not json", "null", "[]", '{"version": 22}',
                                      '{"version": 1, "screen_size": [true, 1080]}',
                                      '{"version": 1, "screen_size": [0, 1080]}'])
def test_corrupt_or_invalid_rig_is_rejected(tmp_path, contents):
    path = tmp_path / "stereo.json"
    path.write_text(contents)
    with pytest.raises(ValueError):
        StereoRig.load(path)


def test_corrupted_calibration_fingerprint_is_rejected(tmp_path):
    path = tmp_path / "stereo.json"
    rig().save(path)
    payload = json.loads(path.read_text())
    payload["calibration"]["translation"][0] = -.30
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="指紋"):
        StereoRig.load(path)


def test_fingerprint_changes_with_camera_identity_extrinsics_and_screen_geometry():
    original = calibration()
    assert calibration(baseline=.25).calibration_id != original.calibration_id
    assert calibration(primary_id="replacement-usb").calibration_id != original.calibration_id
    assert calibration(secondary=2).calibration_id != original.calibration_id
    assert plane(camera_x=.31).plane_id != plane().plane_id
    # Diagnostics do not change the metric mapping's identity.
    assert replace(original, rms=.5, sample_count=20).calibration_id == original.calibration_id


@pytest.mark.parametrize("change", [
    {"primary_source": 2}, {"secondary_source": 2}, {"screen_name": "Other"},
    {"screen_serial": "replacement"}, {"screen_size": (2560, 1440)},
])
def test_rig_compatibility_rejects_different_camera_indices_or_display(change):
    current = dict(primary_source=0, secondary_source=1, screen_name="Display",
                   screen_serial="display-serial", screen_size=(1920, 1080))
    assert rig().compatible(**current)
    assert not rig().compatible(**(current | change))


def test_atomic_save_failure_preserves_previous_rig_and_removes_temporary(tmp_path, monkeypatch):
    path = tmp_path / "stereo.json"
    original = rig()
    original.save(path)
    previous = path.read_bytes()

    def denied(*_args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("kon.stereo_settings.os.replace", denied)
    candidate = replace(original, calibration=calibration(baseline=.25))
    with pytest.raises(OSError, match="simulated"):
        candidate.save(path)
    assert path.read_bytes() == previous
    assert StereoRig.load(path).calibration.calibration_id == original.calibration.calibration_id
    assert not list(tmp_path.glob("*.tmp"))


def test_stereo_and_monocular_nine_point_models_remain_independent(tmp_path):
    mono, stereo = mapping(.10), mapping(-.20)
    save_calibration(tmp_path, mono, mono_config())
    flat = tmp_path / ".kon/calibration.json"
    flat_metadata = tmp_path / ".kon/calibration-meta.json"
    previous = (flat.read_bytes(), flat_metadata.read_bytes())
    save_calibration(tmp_path, stereo, stereo_config())
    assert (flat.read_bytes(), flat_metadata.read_bytes()) == previous
    restored_mono = load_calibration(tmp_path, mono_config())
    restored_stereo = load_calibration(tmp_path, stereo_config())
    assert restored_mono.predict((.5, .5)) == pytest.approx(mono.predict((.5, .5)))
    assert restored_stereo.predict((.5, .5)) == pytest.approx(stereo.predict((.5, .5)))
    assert restored_mono.predict((.5, .5)) != pytest.approx(restored_stereo.predict((.5, .5)))
    assert len(list((tmp_path / ".kon/aim-calibrations").glob("*.meta.json"))) == 2


@pytest.mark.parametrize("change", [
    {"acquisition_mode": "single"}, {"secondary_camera": 2},
    {"stereo_calibration_id": "different-rig"}, {"stereo_calibration_id": ""},
    {"screen_plane_id": "different-plane"}, {"screen_plane_id": ""},
])
def test_stereo_nine_point_mapping_cannot_be_reused_with_different_geometry(tmp_path, change):
    save_calibration(tmp_path, mapping(), stereo_config())
    assert load_calibration(tmp_path, stereo_config()) is not None
    assert load_calibration(tmp_path, stereo_config() | change) is None
    assert not compatible_configuration(stereo_config(), stereo_config() | change)
    assert load_calibration(tmp_path, stereo_config()) is not None


def test_existing_mono_profile_without_acquisition_field_remains_compatible(tmp_path):
    save_calibration(tmp_path, mapping(), mono_config())
    assert compatible_configuration(mono_config(), mono_config() | {"acquisition_mode": "single"})
    assert load_calibration(tmp_path, mono_config() | {"acquisition_mode": "single"}) is not None
    assert load_calibration(tmp_path, stereo_config()) is None
