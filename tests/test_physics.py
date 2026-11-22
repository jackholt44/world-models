"""Unit tests for wm.env.physics (integration + wall resolution) and wm.env.sim."""

from __future__ import annotations

import random

import numpy as np
import pytest

from wm.env import physics as P
from wm.env import sim


def test_compute_next_state_zero_force_moves_by_velocity() -> None:
    state = (10.0, 20.0, 3.0, -1.0)
    action = (0.0, 0.0)
    dt = 0.5
    x, y, vx, vy = P._compute_next_state(state, action, dt)
    assert x == pytest.approx(11.5)
    assert y == pytest.approx(19.5)
    assert vx == pytest.approx(3.0)
    assert vy == pytest.approx(-1.0)


def test_compute_next_state_constant_force_hand_computed() -> None:
    """Semi-implicit Euler: v += (F * ACTION_SCALE / m) dt, then x += v dt."""
    state = (0.0, 0.0, 0.0, 0.0)
    action = (1.0, 0.0)
    dt = 0.1
    x, y, vx, vy = P._compute_next_state(state, action, dt)
    expected_vx = P.ACTION_SCALE * dt / P.MASS
    assert vx == pytest.approx(expected_vx)
    assert vy == pytest.approx(0.0)
    assert x == pytest.approx(expected_vx * dt)
    assert y == pytest.approx(0.0)


def _wall_cases() -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    """A ball 1 unit past each wall is reflected 1 unit inside it, with that velocity flipped."""
    x0, x1, y0, y1 = P.center_bounds()
    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
    return [
        ((x0 - 1, mid_y, -2.0, 0.0), (x0 + 1, mid_y, 2.0, 0.0)),  # past left wall
        ((x1 + 1, mid_y, 2.0, 0.0), (x1 - 1, mid_y, -2.0, 0.0)),  # past right wall
        ((mid_x, y0 - 1, 0.0, -3.0), (mid_x, y0 + 1, 0.0, 3.0)),  # past top (y = 0)
        ((mid_x, y1 + 1, 0.0, 4.0), (mid_x, y1 - 1, 0.0, -4.0)),  # past bottom
    ]


@pytest.mark.parametrize(("incoming", "expected"), _wall_cases())
def test_resolve_wall_collision_axis(
    incoming: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
) -> None:
    out = P._resolve_wall_collision(incoming, *P.center_bounds())
    assert out == pytest.approx(expected)


def test_resolve_wall_collision_corner_outside_both_axes() -> None:
    x0, x1, y0, y1 = P.center_bounds()
    out = P._resolve_wall_collision((x1 + 2, y1 + 3, 1.0, 1.0), x0, x1, y0, y1)
    assert out == pytest.approx((x1 - 2, y1 - 3, -1.0, -1.0))


def test_physics_step_left_wall_bounce() -> None:
    """At the left legal x, moving left: the overshoot is reflected inside and vx flips."""
    x_min, _, _, _ = P.center_bounds()
    state = (x_min, 50.0, -1.0, 0.0)
    action = (0.0, 0.0)
    dt = 0.1
    x, y, vx, vy = P.physics_step(state, action, dt)
    # it would have reached x_min - 0.1, so it ends 0.1 inside the wall
    assert x == pytest.approx(x_min + 0.1)
    assert y == pytest.approx(50.0)
    assert vx == pytest.approx(1.0)
    assert vy == pytest.approx(0.0)


def assert_center_in_bounds(x: float, y: float) -> None:
    x0, x1, y0, y1 = P.center_bounds()
    assert x0 <= x <= x1
    assert y0 <= y <= y1


def test_physics_step_many_random_steps_stay_in_bounds() -> None:
    rng = random.Random(42)
    x0, x1, y0, y1 = P.center_bounds()
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    state = (cx, cy, 0.0, 0.0)
    dt = 0.05
    for _ in range(2000):
        fx = rng.uniform(-5.0, 5.0)
        fy = rng.uniform(-5.0, 5.0)
        state = P.physics_step(state, (fx, fy), dt)
        assert_center_in_bounds(state[0], state[1])
        for v in state[2:]:
            assert v == v  # not NaN
            assert abs(v) < 1e6


def test_sim_step_uses_dataset_dt() -> None:
    """The planner and the dataset must step the same physics."""
    state = (40.0, 60.0, 0.5, -0.5)
    action = (0.3, -0.7)
    assert sim.DT == 1
    assert sim.step(state, action) == P.physics_step(state, action, sim.DT)


def test_sim_reset_and_actions_in_range() -> None:
    rng = random.Random(0)
    for _ in range(100):
        x, y, vx, vy = sim.reset(rng)
        assert_center_in_bounds(x, y)
        assert -1 <= vx <= 1 and -1 <= vy <= 1
        fx, fy = sim.random_action(rng)
        assert -1 <= fx <= 1 and -1 <= fy <= 1


def test_sim_observe_is_chw_uint8() -> None:
    frame = sim.observe((40.0, 60.0, 0.0, 0.0))
    assert frame.shape == (3, int(P.ARENA_HEIGHT), int(P.ARENA_WIDTH))
    assert frame.dtype == np.uint8
