"""
Millimetres per plane unit, for a floor.

The anchor is the camera's height above the floor divided by the plane's own
camera-floor distance, so the depth map's per-image arbitrariness cancels.
This is camera/metric_scale.resolve_mm_per_unit, unchanged -- the function was
already surface-neutral in its maths, it is only the CHOICE of |d| as "camera
height" that is floor knowledge, and that choice is made here.

There is no tile_scale, no density slider and no tuning constant. The tile's
physical size is tile_width_mm x tile_height_mm and nothing rescales it; this
factor only converts plane units into millimetres so those dimensions mean
what they say.
"""

from ...camera.metric_scale import resolve_mm_per_unit


def resolve(plane, opts):
    """
    Returns (mm_per_unit, info). Raises MetricScaleError rather than inventing
    a number if no anchor applies -- see metric_scale.resolve_mm_per_unit.
    """
    return resolve_mm_per_unit(plane=plane, camera_height_mm=opts.camera_height_mm)
