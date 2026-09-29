"""
Surface-specific behaviour, one package per surface.

Import rule this package exists to enforce:

    surface/wall/ must never import from surface/floor/, and vice versa.

A wall problem is fixed inside surface/wall/. A floor problem is fixed inside
surface/floor/. Neither can regress the other, because neither can see the
other. The one thing they share is core/, which knows about neither.

(The wall's metric scale can OPTIONALLY consume a floor plane, but it receives
it as a plain (a,b,c,d) tuple from the pipeline -- it does not import the floor
package to get one. See surface/wall/scale.py.)
"""

from .base import GeometryEvidence, SurfaceInstance, SurfaceKind, SurfaceProfile  # noqa: F401


def get_profile(surface, opts=None) -> SurfaceProfile:
    """
    The single dispatch point. app.py and the pipelines call this rather than
    branching on strings themselves, so adding a surface touches one function.

    Imported lazily so that loading the floor path never pulls in wall code
    and vice versa.
    """
    kind = SurfaceKind.parse(surface)
    if kind is SurfaceKind.WALL:
        from ..pipeline.wall_pipeline import WallProfile
        return WallProfile(opts)
    from ..pipeline.floor_pipeline import FloorProfile
    return FloorProfile(opts)
