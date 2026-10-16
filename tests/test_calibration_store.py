import json

import pytest

from kon.calibration_store import compatible_configuration, load_calibration, save_calibration
from kon.geometry import ScreenCalibration
from kon.personal_calibration import PersonalCalibration


def configuration():
    return {"eye": "right", "camera": 0, "fov": 82.15, "screen": [3440, 1440],
            "screen_name": "MSI", "screen_serial": "display-1", "screen_origin": [0, 0],
            "machine_id": "test-machine"}


def model():
    screen = [(x, y) for y in (.01, .5, .99) for x in (.01, .5, .99)]
    raw = [(x * .2 - .1, y * .2 - .1) for x, y in screen]
    return PersonalCalibration.fit(raw, screen, screen_size=(3440, 1440))


def test_restart_restores_exact_mapping_and_optional_verification(tmp_path):
    original = model()
    checks = original.validation_targets
    raw = [(x * .2 - .1, y * .2 - .1) for x, y in checks]
    original.validate(raw, checks, screen_size=(3440, 1440), frame_samples=[[point] * 6 for point in raw])
    save_calibration(tmp_path, original, configuration())
    restored = load_calibration(tmp_path, configuration())
    assert restored.raw_points == original.raw_points
    assert restored.screen_points == original.screen_points
    assert restored.ready_for_live
    for point in original.raw_points:
        assert restored.predict(point) == pytest.approx(original.predict(point))
    assert not list(tmp_path.rglob("*.tmp"))


def test_origin_name_and_missing_serial_do_not_force_new_calibration(tmp_path):
    original = model()
    save_calibration(tmp_path, original, configuration())
    changed = configuration() | {"screen_name": "Renamed display", "screen_origin": [1920, -120], "screen_serial": ""}
    assert load_calibration(tmp_path, changed).raw_points == original.raw_points


@pytest.mark.parametrize("change", [{"eye": "left"}, {"camera": 1}, {"fov": 70},
                                    {"screen": [1920, 1080]}, {"machine_id": "another-machine"},
                                    {"screen_serial": "another-display"}])
def test_different_physical_configuration_keeps_data_without_using_wrong_profile(tmp_path, change):
    save_calibration(tmp_path, model(), configuration())
    assert load_calibration(tmp_path, configuration() | change) is None
    assert load_calibration(tmp_path, configuration()) is not None


def test_switching_camera_configuration_can_return_to_each_saved_profile(tmp_path):
    first = model()
    second_config = configuration() | {"camera": 1}
    save_calibration(tmp_path, first, configuration())
    save_calibration(tmp_path, first, second_config)
    assert load_calibration(tmp_path, configuration()) is not None
    assert load_calibration(tmp_path, second_config) is not None
    assert len(list((tmp_path / ".kon/aim-calibrations").glob("*.meta.json"))) == 2


def test_existing_flat_profile_without_machine_key_is_migratable(tmp_path):
    original = model()
    original.save(tmp_path / ".kon/calibration.json")
    legacy = configuration()
    legacy.pop("machine_id")
    (tmp_path / ".kon/calibration-meta.json").write_text(json.dumps(legacy))
    assert load_calibration(tmp_path, configuration()).raw_points == original.raw_points


def test_valid_legacy_affine_profile_still_loads(tmp_path):
    screen = [(x, y) for y in (0, .5, 1) for x in (0, .5, 1)]
    original = ScreenCalibration.fit(screen, screen)
    save_calibration(tmp_path, original, configuration())
    assert load_calibration(tmp_path, configuration()).predict((.5, .5)) == pytest.approx((.5, .5))


def test_missing_and_corrupt_profiles_are_ignored(tmp_path):
    assert load_calibration(tmp_path, configuration()) is None
    (tmp_path / ".kon").mkdir()
    (tmp_path / ".kon/calibration-meta.json").write_text("null")
    (tmp_path / ".kon/calibration.json").write_text("not-json")
    assert load_calibration(tmp_path, configuration()) is None
    assert not compatible_configuration({}, configuration())
