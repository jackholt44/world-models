from typing import Tuple

# State is a tuple of (x, y, vx, vy) in meters and meters per second
State = Tuple[float, float, float, float]  # x, y, vx, vy
# Action is a tuple of (force_x, force_y) in newtons
Action = Tuple[float, float]


def _compute_next_state(state: State, action: Action, dt: float) -> State:
    """
    Step the physics forward by one time step.
    This is a simple physics engine that does not include gravity or friction.
    It is a simple Euler integration of the equations of motion.
    The equations of motion are:
    dv/dt = F/m
    dx/dt = v
    dy/dt = v
    The force is the applied force.

    Args:
        state: The current state of the system.
        action: The action to apply to the system.
        dt: The time step to use for the integration.

    Returns:
        The new state of the system.
    """
    x, y, vx, vy = state
    force_x, force_y = action
    scaled_force_x = force_x * ACTION_SCALE
    scaled_force_y = force_y * ACTION_SCALE
    vx += scaled_force_x * dt / MASS
    vy += scaled_force_y * dt / MASS
    x += vx * dt
    y += vy * dt
    return (x, y, vx, vy)


MASS = 1.0  # kg

# Arena outer size in world units (same units as x, y — often pixels once you match pygame).
ARENA_WIDTH = 128.0
ARENA_HEIGHT = 128.0
# State (x, y) is the **center** of the ball. Walls lie on the arena outer rectangle
# x in [0, ARENA_WIDTH], y in [0, ARENA_HEIGHT]. Inset center bounds by BALL_RADIUS.
BALL_RADIUS = 5.0

# Action scale factor to convert from [-1, 1] to force in newtons.
# with scale 10 we saw a lot of bounces and therefore more instances of destoryed infomation
# the model does not know if the ball bounced or was just moving slowly
ACTION_SCALE = 2.0


def center_bounds() -> Tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) for the ball **center**."""
    r = BALL_RADIUS
    return (r, ARENA_WIDTH - r, r, ARENA_HEIGHT - r)


def _resolve_wall_collision(
    state: State,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> State:
    """
    Resolve a collision with the wall.
    If the ball hits the wall, it will bounce off the wall.
    The ball will bounce off the wall with the same velocity, but in the opposite direction.
    We assume dt is small enough that the ball does not tunnel past a wall in one step.
    Args:
        state: The current state of the system.
        x_min, x_max, y_min, y_max: Legal range for the ball **center** (already inset by radius).

    Returns:
        The new state of the system.
    """
    x, y, vx, vy = state
    if x < x_min:
        # reflect the ball off the left wall
        # x_raw = x - vx * dt
        # x_min + (x_min - x_raw) = 2 * x_min - x_raw
        x = 2 * x_min - x
        vx = -vx
    elif x > x_max:
        x = 2 * x_max - x
        vx = -vx
    if y < y_min:
        y = 2 * y_min - y
        vy = -vy
    elif y > y_max:
        y = 2 * y_max - y
        vy = -vy
    return (x, y, vx, vy)


def physics_step(state: State, action: Action, dt: float) -> State:
    """
    Step the physics forward by one time step.
    """
    next_state = _compute_next_state(state, action, dt)
    x0, x1, y0, y1 = center_bounds()

    return _resolve_wall_collision(next_state, x0, x1, y0, y1)
