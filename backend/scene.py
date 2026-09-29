"""
Calibrated scene assets.

The tile projection in engine.py is a metric raycast: it needs a camera
(K, R, t), a floor mask and a screeding mask that belong to the *same* room
photograph. Those come out of the GPU stages of the pipeline (Qwen empty room,
SAM2 masks, stage08g3 camera fit) and are committed under
test07/production_pipeline/.

This module loads that one calibrated scene. Nothing here infers geometry from
an arbitrary upload — see engine.MissingGeometryError for what happens then.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROD = PROJECT_ROOT / "runtime_assets"

EMPTY_ROOM_PATH = (
    PROD / "00_stage07_empty_room_candidate.png"
)

MASTER_INPUT_PATH = PROD / "00_master_input.png"

FLOOR_MASK_PATH = (
    PROD / "01_main_floor_mask.png"
)

SCREEDING_MASK_PATH = (
    PROD / "02_wall_screeding_mask.png"
)

CAMERA_STATE_PATH = (
    PROD / "00_stage08g3_room_box_state.json"
)

PROP_MASK_PATH = (
    PROD / "01_final_six_prop_union_mask.png"
)

REQUIRED_ASSETS = (
    EMPTY_ROOM_PATH,
    MASTER_INPUT_PATH,
    FLOOR_MASK_PATH,
    SCREEDING_MASK_PATH,
    CAMERA_STATE_PATH,
    PROP_MASK_PATH,
)


@dataclass(frozen=True)
class WallPlane:
    """
    One flat wall of the room box, and how a tile grid runs across it.

    A wall is a plane on which one world coordinate is constant — the left and
    right walls hold X, the back wall holds Z — plus the two directions a tile
    layout needs on it. `horizontal` is the world axis running across the wall
    as you look at it, signed so the coordinate grows left to right in the
    image; the vertical direction is always height above the floor, so it does
    not need naming.

    Keeping the parameterisation with the plane is what lets one projection
    routine serve every wall: the tile maths is identical once a hit point has
    been turned into (across, up), and only these numbers differ between the
    wall on your left and the one in front of you.
    """

    label: str
    axis: int  # world axis held constant: 0 = X, 1 = Y, 2 = Z
    value: float  # its value, in millimetres
    horizontal_axis: int  # world axis running across the wall
    horizontal_sign: float  # +1 or -1, so `across` increases with image x
    horizontal_origin: float  # world value of `across` = 0
    width_mm: float  # how far the wall runs across
    height_mm: float  # how far it runs up from the floor


@dataclass(frozen=True)
class Scene:
    """One fully calibrated room: pixels, masks and a metric camera."""

    empty_room: np.ndarray  # (H, W, 3) uint8 — props removed, lighting intact
    master_input: np.ndarray  # (H, W, 3) uint8 — original RGB, source of truth
    floor: np.ndarray  # (H, W) bool
    screeding: np.ndarray  # (H, W) bool — wall skirting band
    props: np.ndarray  # (H, W) bool — union of the six physical props
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    camera_center: np.ndarray
    back_x_image: float  # image x of the back floor corner; splits left/right wall
    room_u_mm: float
    room_v_mm: float
    room_height_mm: float

    # Full wall geometry, when the scene has it. The calibrated room leaves
    # these empty and keeps its committed screeding band, which is the wall
    # surface its own pipeline measured; an uploaded room fills them in from the
    # segmented wall and the estimated camera. `render` picks whichever the
    # scene carries, so neither path can disturb the other.
    wall: np.ndarray | None = None  # (H, W) bool — every visible wall pixel
    walls: tuple[WallPlane, ...] = ()

    # Soft coverage for the props, in [0, 1], when the scene has it.
    #
    # `props` is binary, and pasting objects back with it gives every one of
    # them a stair-stepped edge and a one-pixel rim of whatever was behind it —
    # the look of a cut-out dropped on a background. This is the same mask
    # matted against the photo's own gradients, so the restored object meets the
    # new tiles along its real boundary.
    #
    # Its presence also marks a scene whose `master_input` still contains the
    # objects, which is what makes the contact-shadow transfer possible. The
    # calibrated room leaves this empty: its empty-room render already carries
    # the room's lighting with the props removed, and nothing here should
    # disturb that.
    props_alpha: np.ndarray | None = None  # (H, W) float32

    # How `engine._relight` recovers this room's illumination from its empty
    # room. See `engine.ILLUM_POLY_ORDER` for why the two paths differ.
    #
    # "gaussian" is the calibrated room's behaviour and the default, so that
    # path stays byte-for-byte what the committed pipeline produced. An
    # uploaded room sets "polynomial".
    illumination: str = "gaussian"

    # How many pixels each wall actually received, filled in by the renderer.
    # A plain dict on a frozen dataclass: the binding never changes, only its
    # contents, which is what lets the render report per-wall coverage without
    # threading a second return value through the projection.
    wall_coverage: dict = field(default_factory=dict)

    @property
    def has_wall_planes(self) -> bool:
        return bool(self.walls) and self.wall is not None and bool(self.wall.any())

    @property
    def height(self) -> int:
        return int(self.empty_room.shape[0])

    @property
    def width(self) -> int:
        return int(self.empty_room.shape[1])


def missing_assets() -> list[Path]:
    return [path for path in REQUIRED_ASSETS if not path.exists()]


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


@lru_cache(maxsize=1)
def load_scene() -> Scene:
    """Load and cache the calibrated scene. Raises FileNotFoundError if incomplete."""
    for path in REQUIRED_ASSETS:
        if not path.exists():
            raise FileNotFoundError(path)

    state = json.loads(CAMERA_STATE_PATH.read_text(encoding="utf-8"))

    K = np.asarray(state["camera"]["K"], dtype=np.float64)
    R = np.asarray(state["camera"]["R"], dtype=np.float64)
    t = np.asarray(state["camera"]["t"], dtype=np.float64)

    dimensions = state["room_dimensions_mm"]

    return Scene(
        empty_room=_rgb(EMPTY_ROOM_PATH),
        master_input=_rgb(MASTER_INPUT_PATH),
        floor=_mask(FLOOR_MASK_PATH),
        screeding=_mask(SCREEDING_MASK_PATH),
        props=_mask(PROP_MASK_PATH),
        K=K,
        R=R,
        t=t,
        camera_center=-R.T @ t,
        back_x_image=float(state["evidence"]["back_floor_corner"][0]),
        room_u_mm=float(dimensions["width_2400_axis"]),
        room_v_mm=float(dimensions["length_1800_axis"]),
        room_height_mm=float(dimensions["height"]),
    )


def scene_signature() -> np.ndarray:
    """
    A 32x32 normalised grayscale fingerprint of the calibrated room photo.

    Used to decide whether an uploaded room image *is* the calibrated room, in
    which case the metric pipeline applies to it.
    """
    return image_signature(load_scene().master_input)


def image_signature(rgb: np.ndarray) -> np.ndarray:
    small = np.asarray(
        Image.fromarray(rgb).convert("L").resize((32, 32), Image.BILINEAR),
        dtype=np.float32,
    )

    small -= small.mean()

    norm = float(np.linalg.norm(small))

    return small / norm if norm > 1e-6 else small


def signature_match(rgb: np.ndarray) -> float:
    """Correlation in [-1, 1] between an uploaded room and the calibrated room."""
    return float((image_signature(rgb) * scene_signature()).sum())
