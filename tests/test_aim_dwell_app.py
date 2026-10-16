"""Physical hand motion changes dwell weighting, independently of rendered aim."""

import pytest

from kon.desktop import DesktopTarget
from test_grid_app import send, session


@pytest.fixture
def motion_session(session, monkeypatch):
    window, clock = session
    window.stable_dwell_multiplier = 2
    window.stable_dwell_seconds = .3
    window.hand_movement_speed_limit = .04
    weights = []
    update = window.gate.update

    def record_update(*args, **kwargs):
        weights.append(kwargs.get("dwell_weight", 1))
        return update(*args, **kwargs)

    monkeypatch.setattr(window.gate, "update", record_update)
    return window, clock, weights


@pytest.mark.parametrize("step", [0, .002])
def test_still_or_slow_real_hand_in_one_cell_gains_double_weight_after_stability(motion_session, step):
    window, clock, weights = motion_session
    for index in range(12):
        timestamp = 100 + index * .1
        send(window, clock, timestamp, hand_cursor=(.35 + index * step, .5))
        if index:
            assert not window._aim_dwell_motion.moving
            assert window._aim_dwell_motion.speed <= window.hand_movement_speed_limit
        expected = sum(.1 * min(weights[index - 1], weights[index])
                       for index in range(1, len(weights)))
        assert window.gate.leading_dwell_seconds == pytest.approx(expected)
        if window.gate.locked_target is not None:
            break
    assert weights[:3] == [1, 1, 1]
    assert 2 in weights
    assert window.gate.locked_target == "a" and window.voice_active
    assert window.gate.locked_at < 101.5
    assert window.overlay.point == window.targets[0].center


@pytest.mark.parametrize("hand_data", ["fast", "missing"])
def test_stable_mapped_aim_cannot_fake_slow_physical_hand_motion(motion_session, hand_data):
    window, clock, weights = motion_session
    for index in range(15):
        hand_cursor = (.2 + .6 * (index % 2), .5) if hand_data == "fast" else None
        send(window, clock, 100 + index * .1, hand_cursor=hand_cursor)
        assert window.gate.leading_dwell_seconds == pytest.approx(index * .1)
        assert window.gate.locked_target is None and not window.voice_active
        if index and hand_data == "fast":
            assert window._aim_dwell_motion.moving
            assert window._aim_dwell_motion.speed > window.hand_movement_speed_limit
    assert weights == [1] * 15
    send(window, clock, 101.5, hand_cursor=(.8, .5) if hand_data == "fast" else None)
    assert window.gate.locked_target == "a" and window.voice_active


def test_changing_the_actual_grid_cell_restarts_the_stability_period(motion_session):
    window, clock, weights = motion_session
    window.targets.append(DesktopTarget("b", "B.txt", (400, 180)))
    for index in range(5):
        send(window, clock, 100 + index * .1, hand_cursor=(.35, .5))
    assert 2 in weights
    for index in range(5, 8):
        send(window, clock, 100 + index * .1, (400, 180), hand_cursor=(.35, .5))
        assert weights[-1] == 1
        assert not window._aim_dwell_motion.stable
    send(window, clock, 100.8, (400, 180), hand_cursor=(.35, .5))
    assert weights[-1] == 2 and window._aim_dwell_motion.stable
    assert window.gate.locked_target is None


def test_a_short_pause_retains_dwell_but_restarts_motion_stability(motion_session):
    window, clock, weights = motion_session
    for index in range(5):
        send(window, clock, 100 + index * .1, hand_cursor=(.35, .5))
    retained = window.gate.leading_dwell_seconds
    assert 2 in weights
    send(window, clock, 100.5, None, hand_cursor=(.35, .5))
    assert window.gate.leading_dwell_seconds == pytest.approx(retained)
    assert window._aim_dwell_motion.weight == 1 and not window._aim_dwell_motion.stable
    send(window, clock, 100.6, hand_cursor=(.35, .5))
    assert weights[-1] == 1
    assert window.gate.leading_dwell_seconds == pytest.approx(retained)
    for timestamp in (100.7, 100.8):
        send(window, clock, timestamp, hand_cursor=(.35, .5))
        assert weights[-1] == 1
    send(window, clock, 100.9, hand_cursor=(.35, .5))
    send(window, clock, 101.0, hand_cursor=(.35, .5))
    assert weights[-1] == 2


def test_render_ticks_never_advance_hand_stability_or_weighted_dwell(motion_session):
    window, clock, weights = motion_session
    window.overlay.show()
    for index in range(3):
        send(window, clock, 100 + index * .1, hand_cursor=(.35, .5))
    latest, retained = window.latest, window.gate.leading_dwell_seconds
    assert weights[-1] == 1 and not window._aim_dwell_motion.stable
    for index in range(1, 30):
        clock["now"] = 100.2 + index * .01
        window.animate_cursor()
    assert window.latest is latest and len(weights) == 3
    assert window._aim_dwell_motion.weight == 1 and not window._aim_dwell_motion.stable
    assert window.gate.leading_dwell_seconds == pytest.approx(retained)
    assert window.gate.locked_target is None and not window.voice_active
