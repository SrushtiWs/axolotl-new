"""
Floor-specific behaviour.

Every module here is a thin adapter over code that already worked. Nothing was
reimplemented: segment.py calls the existing generate_floor_mask, vp.py calls
the existing detect_floor_vanishing_points and extract_floor_boundary,
scale.py calls the existing resolve_mm_per_unit, and anchor.py is the
auto_anchor_to_wall block lifted out of the renderer unchanged. The floor
renders exactly as it did.
"""
