import time
import warnings

# pygame 2.6.1 imports setuptools' pkg_resources, which warns on import.
# https://github.com/pygame/pygame/issues/4557
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API",
        category=UserWarning,
    )
    import pygame

from wm.env import sim
from wm.env.physics import center_bounds
from wm.env.render import render

x0, x1, y0, y1 = center_bounds()
STARTING_STATE = ((x0 + x1) / 2, (y0 + y1) / 2, 0.0, 0.0)

ACTIONS = [
    (0.1, 0.0),
    (0.1, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
]


def loop() -> None:
    states = [STARTING_STATE]
    for action in ACTIONS:
        states.append(sim.step(states[-1], action))
        print(states[-1])
    frames = [render(state) for state in states]
    return frames


if __name__ == "__main__":
    frames = loop()
    for frame in frames:
        # Initialize the display
        pygame.init()
        # Set the display mode (width, height)
        screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
        # Set the caption
        pygame.display.set_caption("wm env — close window to quit")
        # Detach from numpy buffer so the array can be GC'd safely
        # Convert the frame to a pygame surface (width, height, channels)
        surf = pygame.image.frombuffer(
            frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB"
        ).convert()
        # Blit the surface to the screen (x, y)
        screen.blit(surf, (0, 0))
        # Update the display (swap the buffers)
        pygame.display.flip()
        # Sleep for 1 second
        time.sleep(1)
