"""Verify continuous liquid physics independently of Qt or the paint rate."""
import math

import pytest

from kon.effects import (
    LIQUID_DROP_COUNT, LIQUID_DURATION, LIQUID_EMISSION_WINDOW,
    ballistic_motion, liquid_drops, liquid_particles, liquid_state,
)


@pytest.mark.parametrize("age", [0, .03, .15, .60])
def test_zero_drag_matches_constant_acceleration(age):
    vx, vy, gravity = 410.0, -250.0, 1200.0
    assert ballistic_motion(age, vx, vy, gravity, 0) == pytest.approx((
        vx * age, vy * age + gravity * age * age / 2,
        vx, vy + gravity * age,
    ))


@pytest.mark.parametrize("drag", [.8, 3.5])
def test_dragged_trajectory_obeys_velocity_and_force_equations(drag):
    age, delta, gravity = .23, 1e-6, 1000.0
    current = ballistic_motion(age, 620.0, -320.0, gravity, drag)
    earlier = ballistic_motion(age - delta, 620.0, -320.0, gravity, drag)
    later = ballistic_motion(age + delta, 620.0, -320.0, gravity, drag)
    derivatives = [(after - before) / (2 * delta) for before, after in zip(earlier, later)]
    _, _, vx, vy = current
    assert derivatives == pytest.approx((vx, vy, -drag * vx, gravity - drag * vy),
                                       rel=1e-5, abs=1e-3)


@pytest.mark.parametrize("drag", [1e-12, 1e-8, 1e-5])
def test_negligible_drag_stays_close_to_ballistic_motion(drag):
    ballistic = ballistic_motion(.4, 400.0, -75.0, 600.0, 0)
    actual = ballistic_motion(.4, 400.0, -75.0, 600.0, drag)
    assert all(math.isfinite(value) for value in actual)
    assert actual == pytest.approx(ballistic, rel=1e-5, abs=1e-5)


def test_drag_slows_forward_motion_and_gravity_turns_spray_downward():
    motion = [ballistic_motion(age, 700, -300, 800, 2) for age in (0, .1, .4, .7)]
    assert all(a[0] < b[0] for a, b in zip(motion, motion[1:]))
    assert all(a[2] > b[2] > 0 for a, b in zip(motion, motion[1:]))
    assert motion[1][1] < motion[0][1]
    assert motion[1][3] < 0 < motion[2][3]
    assert motion[-1][1] > motion[-2][1]


def test_emissions_are_varied_deterministic_and_end_on_time():
    drops = liquid_drops(348)
    assert drops == liquid_drops(348)
    assert len(drops) == LIQUID_DROP_COUNT == 72
    assert len({drop.index for drop in drops}) == len(drops)
    assert len({(drop.vx0, drop.vy0) for drop in drops}) == len(drops)
    assert {drop.jet for drop in drops} == {-1, 0, 1, 2}
    assert all(0 <= drop.emitted_at <= LIQUID_EMISSION_WINDOW for drop in drops)
    assert all(drop.emitted_at + drop.lifetime <= LIQUID_DURATION for drop in drops)
    assert all(drop.vx0 > 0 and drop.gravity > 0 and drop.radius > 0 for drop in drops)
    assert not liquid_particles(-.001, 348)
    assert len(liquid_particles(.10, 348)) == len(drops)
    assert not liquid_particles(LIQUID_DURATION, 348)


def test_drop_lifetimes_and_fades_do_not_teleport_or_pop():
    for drop in liquid_drops(348):
        assert liquid_state(drop, drop.emitted_at - .001) is None
        birth = liquid_state(drop, drop.emitted_at)
        assert birth is not None and birth.age == 0
        assert (birth.x, birth.y, birth.vx, birth.vy) == pytest.approx((
            drop.x0, drop.y0, drop.vx0, drop.vy0,
        ))
        middle = liquid_state(drop, drop.emitted_at + (drop.fade_start + drop.lifetime) / 2)
        final = liquid_state(drop, drop.emitted_at + drop.lifetime - 1e-6)
        assert middle is not None and final is not None
        assert birth.opacity > middle.opacity > final.opacity >= 0
        assert final.opacity < 1e-5
        assert liquid_state(drop, drop.emitted_at + drop.lifetime) is None


def test_smaller_beads_have_more_air_drag_and_longer_motion_tails():
    ordered = sorted(liquid_drops(348), key=lambda drop: drop.radius)
    assert all(a.radius < b.radius and a.drag > b.drag and a.tail_seconds > b.tail_seconds
               for a, b in zip(ordered, ordered[1:]))
    assert all(0 < drop.tail_seconds < drop.lifetime for drop in ordered)


def test_sample_order_and_screen_scale_do_not_change_particle_motion():
    early = liquid_particles(.16, 200)
    liquid_particles(.35, 200)
    assert liquid_particles(.16, 200) == early
    larger = liquid_particles(.16, 400)
    assert [state.index for state in larger] == [state.index for state in early]
    for small, big in zip(early, larger):
        assert (big.x, big.y, big.vx, big.vy, big.radius) == pytest.approx(tuple(
            value * 2 for value in (small.x, small.y, small.vx, small.vy, small.radius)
        ))
        assert big.age == small.age and big.opacity == small.opacity


@pytest.mark.parametrize("size", [0, -1, math.inf, math.nan])
def test_invalid_particle_size_is_rejected(size):
    with pytest.raises(ValueError):
        liquid_drops(size)


@pytest.mark.parametrize("age,drag", [(-.01, 2), (.1, -1), (math.inf, 2), (.1, math.nan)])
def test_invalid_motion_inputs_are_rejected(age, drag):
    with pytest.raises(ValueError):
        ballistic_motion(age, 700, -300, 800, drag)
