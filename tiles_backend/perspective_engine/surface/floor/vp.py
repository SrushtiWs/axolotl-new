"""
Floor vanishing-point resolution.

The detection itself is camera/vp_from_mask.detect_floor_vanishing_points and
camera/floor_boundary.extract_floor_boundary -- both unchanged. What lives
here is the accept/reject policy that used to sit inline in the renderer:
which VP is used for rotation, which height is used for pitch, and when a
detection is discarded in favour of the caller's manual point.
"""

from ...camera.floor_boundary import extract_floor_boundary
from ...camera.vp_from_mask import (
    DEPTH_SELECT_WEIGHTS as _VP_SELECT_WEIGHTS,
    detect_floor_vanishing_points,
)
from ...core import debug as _debug
from ..base import GeometryEvidence

# Plain-language reason per detector failure stage, so the terminal explains
# what went wrong instead of only naming it. Stage strings come from
# vp_from_mask.detect_floor_vanishing_points.
_VP_STAGE_REASONS = {
    "no-input": "no room image or floor mask was supplied",
    "too-few-floor-lines": "not enough line segments survived the floor filter",
    "no-two-directions": "floor lines did not split into two distinct directions",
    "too-few-stable-restarts": "too few RANSAC restarts produced a usable vanishing point",
    "unstable-horizon": "horizon moved too much between RANSAC restarts",
    "unstable-vp-direction": "vanishing-point direction moved too much between restarts",
    "horizon-out-of-range": "fitted horizon fell far outside the image",
    "low-confidence": "confidence fell below the detector's own gate",
}


def log_auto_vp_result(vp_info, auto_x, auto_y, auto_horizon,
                       confidence, threshold, accepted):
    """
    Print the automatic VP detector's own output, exactly as it came back.

    Reports the RAW detector values rather than the resolved vp_x/vp_y the
    renderer goes on to use: on a rejected detection those two differ, and
    logging the resolved pair would print the manual fallback under an
    "AUTO VP" heading.

    Every value is formatted defensively. A failed detection returns None for
    all three coordinates, and f"{None:.2f}" raises TypeError -- which would
    turn a soft VP failure that currently falls back and renders fine into a
    dead request. Logging must never be able to break a render.
    """
    def num(v, nd=2):
        try:
            return f"{float(v):.{nd}f}"
        except (TypeError, ValueError):
            return "None"

    def pt(v):
        if v is None:
            return "None"
        try:
            return f"({float(v[0]):.2f}, {float(v[1]):.2f})"
        except (TypeError, ValueError, IndexError):
            return str(v)

    stage = vp_info.get("stage", "unknown")

    # Depth/cross decision, broken down by the signal that drove it. Printed
    # first because a grid rotated off the room's axes is almost always this
    # choice going wrong, and the totals alone do not say which signal did it.
    det = vp_info.get("depth_select_detail")
    if det:
        _debug("=== VP DEBUG (depth/cross selection) ===")
        _debug(f"VP A: {pt(det['vp_a'])}   terms: {det['a_terms']}")
        _debug(f"VP B: {pt(det['vp_b'])}   terms: {det['b_terms']}")
        _debug(f"PICKED AS DEPTH: cluster {det['picked']}   margin: {det['margin']:.4f}")
        _debug(f"cluster sizes (A,B): {vp_info.get('cluster_sizes')}   "
               f"vertical fraction (A,B): {vp_info.get('cluster_vertical_fraction')}")
        w = ", ".join(f"{k}={v}" for k, v in _VP_SELECT_WEIGHTS.items())
        _debug(f"selection weights: {w}")

    _debug("=== AUTO VP RESULT ===")
    _debug(f"VP X: {num(auto_x)}")
    _debug(f"VP Y: {num(auto_y)}")
    _debug(f"Horizon Y: {num(auto_horizon)}")
    _debug(f"Confidence: {num(confidence, 3)}  (threshold {num(threshold, 3)})")
    _debug(f"VP1: {pt(vp_info.get('vp1'))}")
    _debug(f"VP2: {pt(vp_info.get('vp2'))}")
    _debug(f"Confidence Terms: {vp_info.get('confidence_terms')}")
    _debug(f"Stage: {stage}")
    _debug(f"Accepted: {'true' if accepted else 'false'}")
    if not accepted:
        if stage == "ok":
            reason = (f"confidence {num(confidence, 3)} < "
                      f"vp_confidence_threshold {num(threshold, 3)}")
        else:
            reason = _VP_STAGE_REASONS.get(stage, "detection did not complete")
        _debug(f"Reason: {reason}")
        _debug(f"Floor Lines: {vp_info.get('floor_lines')} of "
               f"{vp_info.get('total_lines')} detected, "
               f"{vp_info.get('boundary_lines')} boundary")
    _debug("======================")


def extract_boundary(floor_bool, room_shape, enabled: bool = True):
    """
    Silhouette of the segmented floor, reduced to a polygon: its corners and,
    more usefully, the floor-to-wall junction segments. Runs first because it
    needs nothing but the mask, and because everything downstream is better
    off with it -- VP detection gets line evidence a rug cannot fake, and the
    confidence score gets an honest measure of how much floor the frame
    actually shows.
    """
    if not enabled:
        return None
    boundary, boundary_info = extract_floor_boundary(floor_bool, room_shape)
    if boundary is not None:
        print(
            f"[boundary] {len(boundary['corners'])} corners, "
            f"{len(boundary['wall_lines'])} wall lines, "
            f"visible={boundary['visible_fraction']:.2f}, "
            f"coverage={boundary['floor_coverage']:.2f}"
        )
    else:
        print(f"[boundary] unavailable ({boundary_info['stage']})")
    return boundary


def resolve(room_bgr, floor_bool, opts, cx, cy, boundary=None) -> GeometryEvidence:
    """
    Detected from the room's own floor lines when possible, with the
    caller-supplied vanishing_point_x/y as the fallback. Only runs when
    something downstream will actually consume the result.

    Rotation uses the depth VP's real (x, y); pitch uses the horizon height.
    Mixing them describes a direction that is not in the scene and skews the
    grid off the room's axes. With a manual VP there is only the one point, so
    both fall back to it.

    A detection is accepted only if its confidence clears the threshold. A
    returned VP is not evidence that the camera angle was recovered -- see the
    confidence discussion in vp_from_mask.py -- and rendering from a
    low-confidence VP is worse than rendering from the manual one, because it
    looks deliberate.
    """
    ev = GeometryEvidence(
        vp_x=opts.vanishing_point_x,
        vp_y=opts.vanishing_point_y,
        vp_horizon_y=opts.vanishing_point_y,
        source="manual",
        boundary=boundary,
    )

    if not (opts.auto_detect_vp and
            (opts.use_vanishing_point or opts.align_plane_to_vanishing_point)):
        from ...core import VP_DEBUG_INFO
        if VP_DEBUG_INFO:
            # Say so explicitly when the detector never ran. Without this, "no
            # AUTO VP block in the terminal" has two very different meanings --
            # the detector is silent, or the detector was never called because
            # the request asked for manual VP -- and they are
            # indistinguishable.
            _debug("=== AUTO VP RESULT ===")
            _debug("Stage: not-run")
            _debug(f"Reason: auto_detect_vp={opts.auto_detect_vp}, "
                   f"use_vanishing_point={opts.use_vanishing_point}, "
                   f"align_plane_to_vanishing_point={opts.align_plane_to_vanishing_point} "
                   f"-- automatic detection requires auto_detect_vp=true AND at least "
                   f"one of the other two")
            _debug(f"Manual VP in use: ({opts.vanishing_point_x}, {opts.vanishing_point_y})")
            _debug("======================")
        return ev

    auto_x, auto_y, auto_horizon, vp_info = detect_floor_vanishing_points(
        room_bgr, floor_bool, principal_point=(cx, cy), boundary=boundary,
        use_boundary_lines=opts.use_floor_boundary,
    )
    ev.info = vp_info
    ev.confidence = vp_info.get("confidence", 0.0)
    vp_accepted = False

    if auto_x is not None and auto_y is not None:
        if ev.confidence >= opts.vp_confidence_threshold:
            ev.vp_x, ev.vp_y, ev.vp_horizon_y = auto_x, auto_y, auto_horizon
            ev.source = "auto"
            vp_accepted = True
            # Both raw VPs are kept for two-VP focal calibration; they are only
            # trustworthy for that when the VP itself is.
            ev.vp1_raw = vp_info.get("vp1")
            ev.vp2_raw = vp_info.get("vp2")
            print(
                f"[vp] auto vp=({ev.vp_x:.1f}, {ev.vp_y:.1f}) horizon_y={ev.vp_horizon_y:.1f} "
                f"confidence={ev.confidence:.2f} "
                f"from {vp_info['floor_lines']} floor + {vp_info['boundary_lines']} boundary lines "
                f"terms={vp_info['confidence_terms']}"
            )
        else:
            ev.source = "manual-low-confidence"
            print(
                f"[vp] REJECTED auto vp=({auto_x:.1f}, {auto_y:.1f}) -- "
                f"confidence={ev.confidence:.2f} < {opts.vp_confidence_threshold:.2f} "
                f"terms={vp_info['confidence_terms']} "
                f"-- falling back to manual vp=({ev.vp_x}, {ev.vp_y})"
            )
    else:
        print(
            f"[vp] auto-detect failed ({vp_info['stage']}, "
            f"{vp_info['floor_lines']}/{vp_info['total_lines']} floor lines) "
            f"-- falling back to manual vp=({ev.vp_x}, {ev.vp_y})"
        )

    # Printed after the accept/reject decision, because "Accepted" is not known
    # until the threshold has been applied. Purely a report of what the
    # detector produced -- it reads state and writes to stdout, and nothing
    # downstream depends on it.
    from ...core import VP_DEBUG_INFO
    if VP_DEBUG_INFO:
        log_auto_vp_result(
            vp_info, auto_x, auto_y, auto_horizon,
            ev.confidence, opts.vp_confidence_threshold, vp_accepted,
        )

    return ev
