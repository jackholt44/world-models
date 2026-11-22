"""Off-screen pygame rendering: physics state -> RGB image for the encoder."""

from __future__ import annotations

import warnings

import numpy as np

# pygame 2.6.1 imports setuptools' pkg_resources, which warns on import.
# https://github.com/pygame/pygame/issues/4557
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API",
        category=UserWarning,
    )
    import pygame

from wm.env.physics import ARENA_HEIGHT, ARENA_WIDTH, BALL_RADIUS, State

_pg_inited = False


def _ensure_pg() -> None:
    global _pg_inited
    if not _pg_inited:
        pygame.init()
        _pg_inited = True


def render(state: State) -> np.ndarray:
    """
    Rasterize state to an RGB image.

    Returns:
        uint8 array of shape (H, W, 3), row-major (y, x, channel), suitable for JEPA obs.
    """
    _ensure_pg()
    w, h = int(ARENA_WIDTH), int(ARENA_HEIGHT)
    surface = pygame.Surface((w, h))
    surface.fill((255, 255, 255))
    pygame.draw.circle(
        surface,
        (0, 0, 0),
        (float(state[0]), float(state[1])),
        float(BALL_RADIUS),
    )
    # surfarray is (W, H, 3) indexed [x, y, c]; use (H, W, 3) for typical image tensors
    rgb_whc = pygame.surfarray.array3d(surface)
    rgb_hwc = np.transpose(rgb_whc, (1, 0, 2))
    return np.ascontiguousarray(rgb_hwc, dtype=np.uint8)
