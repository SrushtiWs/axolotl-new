"""
Surface-agnostic rendering core.

Everything in this package works on "a plane in front of a pinhole camera".
It knows nothing about floors or walls and MUST NOT import from
perspective_engine.surface -- the dependency runs one way:

    surface/floor/  ─┐
                     ├─> core/  ─> camera/, geometry/, lines/
    surface/wall/   ─┘

What differs between surfaces is injected: the plane constraint, the metric
anchor, the basis alignment, the rotation baseline and the grid anchor all
arrive as parameters or small objects supplied by a SurfaceProfile.
"""

from .options import SurfaceRenderOptions  # noqa: F401


def debug(msg):
    """
    Emit a debug line to the server terminal.

    flush=True rather than a bare print: under `uvicorn --reload` the app runs
    in a child process whose stdout is a pipe, not a TTY, and Python
    block-buffers a pipe at several kilobytes instead of flushing per line. In
    practice uvicorn's output still came through here without it, but a short
    debug line sitting in a buffer is an expensive thing to debug twice, and
    flushing costs nothing at this call rate.

    Routing through uvicorn's logger as well was tried and removed: it worked,
    but every line then appeared twice -- once raw, once with an INFO prefix.
    """
    print(msg, flush=True)


#: Print the per-render VP diagnostic blocks. Useful while tuning; noisy once
#: settled, so it is one flag to flip rather than a set of lines to delete.
VP_DEBUG_INFO = True
