"""Deterministic liquid motion for the bite effect, independent of frame rate.

Distances scale with the fox's width in pixels, time is in seconds, and positive
y points down the screen. Gravity and linear air drag give each emitted drop a
continuous trajectory. Linear drag is a visual approximation, not a fluid
simulation: thin filaments can be drawn between these drops briefly, then the
same drops continue as separate beads without changing their motion.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import random


LIQUID_DROP_COUNT = 72
LIQUID_DURATION = .66
LIQUID_EMISSION_WINDOW = .065


@dataclass(frozen=True)
class LiquidDrop:
    """An immutable emission, with position measured from the bite's lip."""

    index: int
    jet: int  # 0–2 are clustered jets; -1 is loose spray.
    emitted_at: float
    x0: float
    y0: float
    vx0: float
    vy0: float
    radius: float
    gravity: float
    drag: float  # Inverse seconds; smaller beads experience more drag.
    lifetime: float
    fade_start: float  # Seconds after this drop was emitted.
    tail_seconds: float


@dataclass(frozen=True)
class LiquidState:
    """One visible drop, with age measured since its own emission."""

    index: int
    jet: int
    age: float
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    opacity: float
    tail_seconds: float


def ballistic_motion(age: float, vx: float, vy: float, gravity: float,
                     drag: float) -> tuple[float, float, float, float]:
    """Return displacement and velocity for gravity plus linear air drag.

    This solves dvx/dt = -drag*vx and dvy/dt = gravity-drag*vy analytically.
    The zero-drag branch is ordinary ballistic motion. A small-time series
    preserves the gravity displacement when drag is almost zero.
    """
    if not all(math.isfinite(value) for value in (age, vx, vy, gravity, drag)):
        raise ValueError("Motion inputs must be finite")
    if age < 0 or drag < 0:
        raise ValueError("Age and drag must be nonnegative")
    if drag == 0:
        return vx * age, vy * age + gravity * age * age / 2, vx, vy + gravity * age

    decay = math.exp(-drag * age)
    velocity_time = -math.expm1(-drag * age) / drag
    q = drag * age
    if q < 1e-4:
        gravity_time = age * age * (.5 - q / 6 + q * q / 24 - q ** 3 / 120)
    else:
        gravity_time = (age - velocity_time) / drag
    return (vx * velocity_time, vy * velocity_time + gravity * gravity_time,
            vx * decay, vy * decay + gravity * velocity_time)


@lru_cache(maxsize=32)
def liquid_drops(size: float) -> tuple[LiquidDrop, ...]:
    """Return the same 72 varied emissions for any given fox width.

    Three narrow cones supply the early filaments; the remaining drops make a
    wider fan. Emissions end after 65 ms. Randomness is fixed at construction,
    so painting different timestamps never changes a drop's trajectory.
    """
    if not math.isfinite(size) or size <= 0:
        raise ValueError("Fox size must be positive and finite")
    rng = random.Random(0xF0_71_2026)
    drops = []
    for index in range(LIQUID_DROP_COUNT):
        jet = index % 3 if index < 54 else -1
        angle = (rng.gauss((-.75, -.35, .20)[jet], .105) if jet >= 0
                 else rng.uniform(-1.17, .78))
        angle = min(max(angle, -1.26), 1.08)
        speed = size * rng.uniform(1.8, 3.2)
        radius_fraction = .0045 + .0135 * rng.random() ** 1.5
        if index % 13 == 0:
            radius_fraction += .007
        emitted_at = 0.0 if index < 3 else rng.uniform(0, LIQUID_EMISSION_WINDOW)
        lifetime = min(rng.uniform(.48, .61), LIQUID_DURATION - emitted_at)
        drops.append(LiquidDrop(
            index=index,
            jet=jet,
            emitted_at=emitted_at,
            x0=size * rng.uniform(-.003, .006),
            y0=size * rng.uniform(-.006, .006),
            vx0=math.cos(angle) * speed,
            vy0=math.sin(angle) * speed,
            radius=size * radius_fraction,
            gravity=size * 5.5,
            drag=.75 + .015 / radius_fraction,
            lifetime=lifetime,
            fade_start=min(rng.uniform(.39, .44), lifetime - .055),
            # Heavy beads stay round; fine spray has a longer visible streak.
            tail_seconds=.006 + .014 * (1 - radius_fraction / .025),
        ))
    return tuple(drops)


def liquid_state(drop: LiquidDrop, age: float) -> LiquidState | None:
    """Evaluate an emission at the global age of the spray effect."""
    if not math.isfinite(age):
        raise ValueError("Spray age must be finite")
    if age < drop.emitted_at or age >= drop.emitted_at + drop.lifetime:
        return None
    drop_age = age - drop.emitted_at
    x, y, vx, vy = ballistic_motion(drop_age, drop.vx0, drop.vy0,
                                   drop.gravity, drop.drag)
    fade = min(max((drop_age - drop.fade_start) /
                   (drop.lifetime - drop.fade_start), 0.0), 1.0)
    opacity = 1 - fade * fade * (3 - 2 * fade)
    return LiquidState(drop.index, drop.jet, drop_age, drop.x0 + x, drop.y0 + y,
                       vx, vy, drop.radius, opacity, drop.tail_seconds)


def liquid_particles(age: float, size: float) -> tuple[LiquidState, ...]:
    """Return visible drop states without advancing or mutating a simulation."""
    states = (liquid_state(drop, age) for drop in liquid_drops(size))
    return tuple(state for state in states if state is not None)
