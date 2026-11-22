"""The environment as the dataset and the planner both see it.

Episode generation and closed-loop planning must step the same physics and render the
same frames, so both go through these functions rather than calling physics_step and
render with their own settings.
"""

import random

import numpy as np

from wm.env.physics import Action, State, center_bounds, physics_step
from wm.env.render import render

# the time step the dataset was generated with; the model only knows this one
DT = 1
ACTION_BOUNDS = (-1, 1)


def reset(rng: random.Random) -> State:
    """A random start state: uniform position inside the walls, uniform velocity in [-1, 1]."""
    x0, x1, y0, y1 = center_bounds()
    x = rng.uniform(x0, x1)
    y = rng.uniform(y0, y1)
    vx = rng.uniform(-1, 1)
    vy = rng.uniform(-1, 1)
    return (x, y, vx, vy)


def random_action(rng: random.Random) -> Action:
    """An action from the training distribution: each force uniform in [-1, 1]."""
    force_x = rng.uniform(ACTION_BOUNDS[0], ACTION_BOUNDS[1])
    force_y = rng.uniform(ACTION_BOUNDS[0], ACTION_BOUNDS[1])
    return (force_x, force_y)


def step(state: State, action: Action) -> State:
    return physics_step(state, action, DT)


def rollout(state0: State, actions: np.ndarray) -> np.ndarray:
    """Final states after each action sequence from state0: (N, H, 2) -> (N, H, 4).

    `actions` (N, H, 2) must be on the host; the physics works on Python floats.
    """
    finals = []
    for seq in actions.tolist():
        state = tuple(state0)
        steps = []
        for action in seq:
            state = step(state, tuple(action))
            steps.append(state)
        finals.append(steps)
    return np.asarray(finals, dtype=np.float64)


def observe(state: State) -> np.ndarray:
    """The frame for a state: (3, H, W) uint8, the CHW layout stored in the dataset."""
    return render(state).transpose(2, 0, 1)
