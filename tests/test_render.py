import numpy as np

from wm.env import physics as P
from wm.env.render import render


def test_render_shape_dtype_and_not_blank() -> None:
    x0, x1, y0, y1 = P.center_bounds()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    frame = render((cx, cy, 0.0, 0.0))
    assert frame.shape == (int(P.ARENA_HEIGHT), int(P.ARENA_WIDTH), 3)
    assert frame.dtype == np.uint8
    # Ball center in image coords (row=y, col=x)
    ri, ci = int(round(cy)), int(round(cx))
    assert frame[ri, ci].max() < 128
    assert (frame < 255).any()
