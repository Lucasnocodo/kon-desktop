"""Grid selection and calibration coaching without devices or real file actions."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from kon.app import Window
from kon.desktop import DesktopTarget, validate_target_position
from kon.geometry import ScreenCalibration
from kon.gesture_calibration import GestureCalibration
from kon.vision import TrackingFrame
from test_gesture_calibration import personal_hand
from test_hand_only_app import FakeWorker


@pytest.fixture
def session(tmp_path, monkeypatch):
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr("kon.app.ROOT", tmp_path)
    clock = {"now": 100.0}
    monkeypatch.setattr("kon.app.time.monotonic", lambda: clock["now"])
    window = Window(aim_mode="eye_ray")
    # These selection-policy cases count elapsed seconds without motion bonuses.
    window.stable_dwell_multiplier = 1
    window.overlay.resize(3440, 1440)
    samples = [(x, y) for y in (0, .5, 1) for x in (0, .5, 1)]
    window.calibration = ScreenCalibration.fit(samples, samples)
    window.running = window.testing = window.armed = True
    # Selection tests exercise the mapped continuous point. Filter behavior
    # remains covered by the smoother's motion/noise tests and mode tests.
    monkeypatch.setattr(window.hand_smoother, "update", lambda _time, point, _size: point)
    window.targets = window.overlay.targets = [DesktopTarget("a", "A.txt", (172, 180))]
    window.overlay.grid_enabled = True
    yield window, clock
    window.close()
    application.processEvents()


def send(window, clock, timestamp, point=(120, 130), *, arrival=None, pose=True, **kwargs):
    clock["now"] = timestamp if arrival is None else arrival
    raw = None if point is None else (point[0] / window.overlay.width(), point[1] / window.overlay.height())
    window.on_tracking(TrackingFrame(timestamp, (), (), 1 if pose else 0, pose, None, None, raw, "追蹤中", **kwargs))


def test_full_cell_selects_beyond_old_radius_and_locks_as_soon_as_minimum_is_met(session):
    window, clock = session
    send(window, clock, 100)
    assert window.overlay.target_id == "a"
    assert window.overlay.point == pytest.approx((120, 130))
    assert window.overlay.grid_active_cell == (1, 1)
    assert "A.txt" in window.overlay.hint
    for index in range(1, 15):
        send(window, clock, 100 + index * .1)
        assert window.gate.locked_target is None and not window.voice_active
        assert window.gate.progress == pytest.approx(index * .1 / 1.5)
    send(window, clock, 101.5)
    assert window.gate.locked_target == "a" and window.voice_active
    assert window.overlay.point == (172, 180)
    assert window.gate.locked_at == 101.5


def test_empty_neighbor_border_buffer_preserves_candidate_and_accumulated_dwell(session):
    window, clock = session
    send(window, clock, 100, (220, 180))
    send(window, clock, 100.1, (232, 180))
    assert window.grid_selector.active_cell == (1, 1)
    send(window, clock, 100.2, (240, 180))
    assert window.grid_selector.active_cell is None
    assert window.overlay.point == pytest.approx((240, 180))
    assert window.gate.leading_target == "a"
    assert window.gate.leading_dwell_seconds == pytest.approx(.2)
    assert window.gate.progress == pytest.approx(.2 / 1.5)
    assert window.gate.locked_target is None and not window.voice_active


@pytest.mark.parametrize("pause", ["missing_ray", "brief_pose", "collision", "duplicate", "invalid_filter"])
def test_separate_visits_and_a_short_pause_accumulate_one_and_a_half_seconds_before_three_seconds(session, monkeypatch, pause):
    window, clock = session
    window.targets.append(DesktopTarget("b", "B.txt", (400, 180)))
    if pause == "collision":
        window.targets.extend([DesktopTarget("c", "C.txt", (516, 180)),
                               DesktopTarget("d", "D.txt", (560, 230))])
    if pause == "invalid_filter":
        monkeypatch.setattr(window.hand_smoother, "update",
                            lambda timestamp, point, _size: None if timestamp == 101.5 else point)
    # Valid intervals: A .6s, B .3s, A .5s, a .2s gap, then A .4s.
    # No dwell is inferred between the last valid sample (101.4) and resume (101.6).
    for index in range(21):
        timestamp = 100 + index * .1
        point = (400, 180) if 6 <= index < 9 else (120, 130)
        if index == 15 and pause == "missing_ray":
            send(window, clock, timestamp, None)
        elif index == 15 and pause == "brief_pose":
            send(window, clock, timestamp, pose=False, hand_detected=True,
                 depth_status="ready", gesture_reason="personal gesture mismatch")
        elif index == 15 and pause == "collision":
            send(window, clock, timestamp, (516, 180))
        elif index == 15 and pause == "duplicate":
            send(window, clock, 101.4, (400, 180), arrival=101.5)
        else:
            send(window, clock, timestamp, point)
        if index < 20:
            assert window.gate.locked_target is None and not window.voice_active
        if index in (14, 15, 16):
            assert window.gate.leading_target == "a"
            assert window.gate.leading_dwell_seconds == pytest.approx(1.1)
            assert window.gate.progress == pytest.approx(1.1 / 1.5)
        if index == 15 and pause in ("missing_ray", "duplicate", "invalid_filter"):
            assert "累積" in window.overlay.hint and "已保留" in window.overlay.hint
            assert "A.txt" in window.overlay.hint and "1.1" in window.overlay.hint
    assert window.gate.locked_target == "a" and window.gate.locked_at == 102
    assert window.overlay.target_id == window.voice_target == "a"
    assert window.overlay.point == window.targets[0].center
    assert window.overlay.progress == 1


def test_threshold_can_lock_a_previous_winner_while_current_aim_is_another_nearby_file(session):
    window, clock = session
    window.targets.append(DesktopTarget("b", "B.txt", (400, 180)))
    for index in range(16):
        send(window, clock, 100 + index * .1, (120, 130) if index < 15 else (400, 180))
    assert window.gate.locked_target == "a" and window.gate.locked_at == 101.5
    assert window.overlay.target_id == window.voice_target == "a"
    assert window.overlay.point == window.targets[0].center
    send(window, clock, 101.6, (400, 180))
    assert window.gate.locked_target == "a" and window.voice_active
    fired = []
    window.summon = fired.append
    window.on_keyword(clock["now"], "kon")
    assert fired == ["a"]


def test_valid_empty_cell_at_threshold_still_locks_a_previous_winner(session):
    window, clock = session
    for index in range(16):
        send(window, clock, 100 + index * .1, (120, 130) if index < 15 else (280, 180))
    assert window.gate.locked_target == "a"
    assert window.overlay.point == window.targets[0].center
    assert window.overlay.target_id == window.voice_target == "a"
    assert window.voice_active


def test_noncurrent_winner_uses_its_own_file_center_for_finder_validation(session, monkeypatch):
    window, clock = session
    winner = DesktopTarget("a", "A.txt", (172, 180), object())
    window.targets = window.overlay.targets = [winner, DesktopTarget("b", "B.txt", (400, 180))]
    window.overlay.practice = False
    for index in range(16):
        send(window, clock, 100 + index * .1, (120, 130) if index < 15 else (400, 180))
    assert window.gate.locked_target == "a" and window.voice_target == "a"
    checks, played = [], []
    monkeypatch.setattr("kon.app.validate_target_position", lambda target, point, size: checks.append((target.id, point)) or target)
    monkeypatch.setattr(window.overlay, "play", played.append)
    window.job = lambda function, done: done(function())
    window.summon("a")
    assert checks == [("a", winner.center)]
    assert window.pending_action[1] == "a" and played == [winner]


@pytest.mark.parametrize("allocation", ["empty_wins", "below_minimum"])
def test_empty_majority_or_insufficient_accumulated_dwell_cannot_enable_voice(session, allocation):
    window, clock = session
    window.targets.append(DesktopTarget("b", "B.txt", (400, 180)))
    for index in range(31):
        if allocation == "empty_wins":
            point = (120, 130) if index < 10 else (280, 180)
        else:
            point = (120, 130) if index < 12 else ((400, 180) if index < 23 else (280, 180))
        send(window, clock, 100 + index * .1, point)
    assert window.gate.locked_target is None and not window.gate.ready
    assert not window.voice_active


def test_tied_empty_and_file_votes_wait_until_the_file_is_uniquely_ahead(session):
    window, clock = session
    for index in range(31):
        send(window, clock, 100 + index * .1, (280, 180) if index < 15 else (120, 130))
    assert window.gate.leading_dwell_seconds == pytest.approx(1.5)
    assert window.gate.locked_target is None and not window.voice_active
    send(window, clock, 103.1)
    assert window.gate.locked_target == "a" and window.voice_active


def test_rolling_window_expires_old_visits_instead_of_accumulating_forever(session):
    window, clock = session
    window.targets.append(DesktopTarget("b", "B.txt", (400, 180)))
    for index in range(45):
        point = (120, 130) if index < 10 or index >= 30 else ((400, 180) if index < 18 else (280, 180))
        send(window, clock, 100 + index * .1, point)
        assert window.gate.locked_target is None and not window.voice_active
        if index == 36:
            # The .6s return cannot be added to all of the original 1s visit;
            # its oldest .6s has already left the three-second window.
            assert window.gate.leading_dwell_seconds == pytest.approx(1.0)
    send(window, clock, 104.5)
    assert window.gate.locked_target == "a" and window.voice_active


def test_many_fast_frames_cannot_supply_one_and_a_half_seconds_of_dwell(session):
    window, clock = session
    for index in range(400):
        send(window, clock, 100 + index * .001)
    assert window.gate.progress == pytest.approx(.399 / 1.5)
    assert window.gate.locked_target is None and not window.voice_active


def test_accumulated_dwell_counts_capture_time_despite_fresh_dispatch_delays(session):
    window, clock = session
    for index in range(15):
        timestamp = 100 + index * .1
        delay = 0 if index == 0 else (.29 if index == 14 else .14)
        send(window, clock, timestamp, arrival=timestamp + delay)
        assert window.gate.progress == pytest.approx(index * .1 / 1.5)
        assert window.gate.locked_target is None and not window.voice_active
    assert clock["now"] > 101.5
    send(window, clock, 101.5, arrival=101.75)
    assert window.gate.locked_target == "a" and window.voice_active


@pytest.mark.parametrize("source_time", [100.4, 100.2])
def test_duplicate_or_reordered_source_frame_pauses_without_erasing_or_adding_votes(session, source_time):
    window, clock = session
    window.targets.append(DesktopTarget("b", "B.txt", (400, 180)))
    for index in range(5):
        send(window, clock, 100 + index * .1)
    send(window, clock, source_time, (400, 180), arrival=100.5)
    assert window.gate.locked_target is None and not window.voice_active
    assert window.overlay.target_id != "b"
    assert window.gate.leading_dwell_seconds == pytest.approx(.4)
    assert window.gate.progress == pytest.approx(.4 / 1.5)
    for index in range(11):
        send(window, clock, 100.6 + index * .1)
        assert window.gate.locked_target is None
    send(window, clock, 101.7)
    assert window.gate.locked_target == "a" and window.voice_active


def test_a_long_tracking_gap_discards_retained_votes(session):
    window, clock = session
    for index in range(11):
        send(window, clock, 100 + index * .1)
    send(window, clock, 101.1, None)
    assert window.gate.leading_dwell_seconds == pytest.approx(1.0)
    for index in range(2, 8):
        send(window, clock, 101 + index * .1, None)
    assert window.gate.progress == 0 and window.gate.leading_target is None
    for index in range(15):
        send(window, clock, 101.8 + index * .1)
        assert window.gate.locked_target is None
    send(window, clock, 103.3)
    assert window.gate.locked_target == "a" and window.voice_active


def test_confirmed_gesture_release_discards_retained_votes(session):
    window, clock = session
    for index in range(11):
        send(window, clock, 100 + index * .1)
    for timestamp in (101.1, 101.25, 101.36):
        send(window, clock, timestamp, pose=False, hand_detected=True,
             depth_status="ready", gesture_reason="personal gesture mismatch")
    assert window.gate.progress == 0 and window.gate.leading_target is None
    for index in range(15):
        send(window, clock, 101.5 + index * .1)
        assert window.gate.locked_target is None
    send(window, clock, 103)
    assert window.gate.locked_target == "a" and window.voice_active


def test_malformed_target_metadata_discards_retained_votes(session):
    window, clock = session
    for index in range(11):
        send(window, clock, 100 + index * .1)
    window.targets.append(DesktopTarget("bad", "bad.txt", (float("nan"), 180)))
    send(window, clock, 101.1)
    assert window.gate.progress == 0 and window.gate.leading_target is None
    window.targets.pop()
    for index in range(15):
        send(window, clock, 101.2 + index * .1)
        assert window.gate.locked_target is None
    send(window, clock, 102.7)
    assert window.gate.locked_target == "a" and window.voice_active


@pytest.mark.parametrize("change", ["move", "remove", "resize"])
def test_layout_changes_discard_old_votes_and_require_new_accumulated_dwell(session, change):
    window, clock = session
    window.targets.append(DesktopTarget("b", "B.txt", (400, 180)))
    for index in range(11):
        send(window, clock, 100 + index * .1)
    if change == "move":
        window.targets[0] = DesktopTarget("a", "A.txt", (860, 180))
    elif change == "remove":
        window.targets.pop(0)
    else:
        window.overlay.resize(3000, 1440)
    for index in range(15):
        send(window, clock, 101.1 + index * .1, (400, 180))
        assert window.gate.locked_target is None and not window.voice_active
    send(window, clock, 102.6, (400, 180))
    assert window.gate.locked_target == "b" and window.voice_target == "b"


def test_selection_instructions_describe_recent_cumulative_dwell_and_early_locking(session):
    window, clock = session
    assert "3 秒" in window.summon_caption.text() and "1.5 秒" in window.summon_caption.text()
    assert "加權" in window.summon_caption.text() and "門檻" in window.summon_caption.text()
    assert "0.65" not in window.summon_caption.text()
    window.activate_targets(window.targets)
    assert "最近 3 秒" in window.overlay.hint and "1.5 秒" in window.overlay.hint
    send(window, clock, 100)
    assert "最近 3 秒" in window.overlay.hint and "1.5 秒" in window.overlay.hint
    assert "加權" in window.overlay.hint and "0.65" not in window.overlay.hint
    window.return_to_control()
    window.running = False
    window.aim_method.setCurrentIndex(window.aim_method.findData("hand"))
    for text in (window.summon_caption.text(), window.footnote.text(), window.gesture_instructions.text()):
        assert "3 秒" in text and "1.5 秒" in text
        assert "加權" in text and "0.65" not in text


def test_two_files_in_one_cell_never_acquire_a_lock_or_enable_commands(session):
    window, clock = session
    window.targets.append(DesktopTarget("b", "B.txt", (222, 230)))
    for index in range(41):
        send(window, clock, 100 + index * .1)
    assert window.overlay.grid_ambiguous_cell == (1, 1)
    assert window.overlay.target_id is None
    assert "多個檔案" in window.overlay.hint
    assert window.gate.locked_target is None and not window.voice_active


def test_control_room_exit_disables_grid_and_clears_candidate(session):
    window, clock = session
    send(window, clock, 100)
    window.return_to_control()
    assert not window.overlay.grid_enabled
    assert window.overlay.grid_active_cell is None
    assert window.grid_selector.active_cell is None
    assert window.overlay.point is None and window.overlay.target_id is None
    assert window.overlay.progress == 0


@pytest.mark.parametrize("size", [(3440, 1440), (1920, 1080), (800, 800)])
def test_practice_files_are_centered_in_nine_unique_cells_for_each_screen_size(session, size):
    window, _ = session
    window.overlay.resize(*size)
    window.speech = FakeWorker()
    window.speech.start()
    window.begin_targets()
    assert window.overlay.grid_enabled and window.overlay.practice
    assert len(window.targets) == 9
    cells = [window.grid_selector.cell_for(target.center, size) for target in window.targets]
    assert len(set(cells)) == 9
    assert cells == [(7, 6), (15, 5), (22, 6), (7, 2), (15, 2), (22, 2), (7, 9), (15, 9), (22, 9)]
    for target, cell in zip(window.targets, cells):
        x, y, width, height = window.grid_selector.cell_bounds(cell, size)
        assert target.center == pytest.approx((x + width / 2, y + height / 2))
        # Every document retains its own identity and full-cell hit area.
        picked = window.grid_selector.pick(window.targets, (x + width * .2, y + height * .2), size)
        assert picked.id == target.id


def test_finder_recheck_rejects_a_new_same_cell_file_even_outside_old_hit_circle(monkeypatch):
    snapshot = object()
    target = DesktopTarget("a", "A.txt", (172, 180), snapshot)
    intruder = DesktopTarget("b", "B.txt", (222, 230), object())
    monkeypatch.setattr("kon.desktop.scan_desktop", lambda _size: [target, intruder])
    with pytest.raises(ValueError, match="取消召喚"):
        validate_target_position(target, target.center, (3440, 1440))


def test_six_fast_aim_frames_cannot_replace_one_second_stable_hold(session):
    window, clock = session
    window.begin_calibration()
    window.capture_calibration()
    assert window.calibration_capture_deadline == 106
    for index in range(6):
        send(window, clock, 100.01 + index * .06)
    assert not window.calibration_raw
    assert "穩定時間" in window.overlay._calibration_reason
    for timestamp in (100.45, 100.60, 100.75, 100.9, 101.05):
        send(window, clock, timestamp)
    assert len(window.calibration_raw) == 1
    assert window.calibration_capture_deadline is None
    assert not window.voice_active and not window.armed


def test_gesture_confirmation_shows_live_candidate_match_even_if_old_pose_fails(session):
    window, clock = session
    window.begin_gesture_calibration()
    points = personal_hand()
    window.gesture_candidate = GestureCalibration.fit([points] * 12, source="world", handedness="right")
    window.gesture_dialog.set_step(1)
    clock["now"] = 100.1
    window.on_tracking(TrackingFrame(
        100.1, (), (), 0, False, None, None, None, "舊手勢未通過",
        hand_world_points=points, handedness="Right"))
    assert "第一次記錄相符" in window.gesture_dialog.match_label.text()
    assert not window.voice_active and not window.armed
