"""
The room's structural surfaces: wall, floor and ceiling.

This module used to be `segmentation.py`, and it used to be the object detector
— SegFormer's class map split into connected components, one cut-out per blob.
It is not that any more. Grounding DINO and SAM 2 own object detection now, and
what remains here is the one question SegFormer is genuinely better at than a
box-prompted segmenter: which pixels are the room itself.

That answer is needed in two places, and neither is object detection:

  * `extraction.filtering` rejects a detection whose box is overwhelmingly
    floor — which is how a polished floor coming back from DINO as "table" gets
    thrown out;
  * `live_scene` needs a floor mask to solve a camera for an uploaded photo, so
    that `/generate` can project tiles onto it.

Surfaces are a whole-class question with no instances in it: there is one floor,
however many objects stand on it. So there is no connected-component pass here
and no per-instance anything — just the class map, thresholded per surface.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

import cv2
import numpy as np

import matting
import weights

MODELS = weights.MODELS

MODEL_PATH = MODELS / "segformer_b4_ade.onnx"
CONFIG_PATH = MODELS / "segformer_b4_ade_config.json"

MODEL_URL = (
    "https://huggingface.co/Xenova/segformer-b4-finetuned-ade-512-512/resolve/main/onnx/model.onnx"
)
CONFIG_URL = (
    "https://huggingface.co/Xenova/segformer-b4-finetuned-ade-512-512/resolve/main/config.json"
)

# SegFormer's own preprocessing: 512x512, rescale to [0,1], ImageNet normalise.
INPUT_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The three ADE20K classes this module reports as surfaces.
SURFACE_CLASSES = ("wall", "floor", "ceiling")

# Fixed structure that is not one of the three surfaces but is just as much a
# part of the building: you cannot carry it out of the room.
#
# These were being detected as furniture and lifted into ALL_OBJECTS.png, which
# then removed them from the clean room — measured on an entrance hall where a
# staircase came back as "oven" at 82,401 px and its bottom step as "table" at
# 14,566 px, leaving a smeared hole where the stairs had been. Nothing in the
# surface rules could catch it, because the three room surfaces cover 0.1% of
# that mask.
#
# ADE20K names them precisely, and this model reads them well: on that same
# photo it labelled 81,845 px "stairway" and 14,828 px "stairs", and the
# staircase's own bounding box came back 58.6% stairway. The information was
# there all along and simply was not being asked for.
#
# Reported alongside the room surfaces so `filtering.structural_map` includes
# them, which is what lets a detection that is really the staircase be rejected
# before it ever becomes a cut-out. The *trim* is deliberately not extended to
# them — `SegmentationConfig.trim_surfaces` still names only floor, wall and
# ceiling — so no object's boundary is cut against these.
ARCHITECTURE_CLASSES = ("stairway", "stairs", "step", "railing", "bannister")

# ADE20K has a dedicated "mirror" class (27), kept distinct from "glass" (147)
# and "windowpane" (8). Only the first is a mirror: a glass partition and a
# window are not, however reflective they look in a photo.
#
# This is reported because it is the only *pixel-level* evidence available for
# whether something is a mirror. Grounding DINO's opinion is a text score, and
# measurement showed it cannot be trusted alone here — on a bedroom containing
# no mirror at all it called a wall shelf "mirror" at 0.266, more confidently
# than it called a genuine wall mirror a mirror (0.212) in another room. Neither
# the raw score nor its margin over the best non-mirror phrase separates those
# two cases. This class does: it matched the real mirror to within a few pixels
# and found nothing at all in the bedroom.
MIRROR_CLASS = "mirror"

# Background a mask can swallow that is not one of the three room surfaces.
#
# These are reported separately because they are trimmed differently. Wall,
# floor and ceiling are routinely misread *on furniture* — a beige headboard
# comes back as "wall" — so trimming them has to be cautious. These do not have
# that failure mode: glazing and sky appearing inside a plant's mask is the
# window behind the plant, seen between its leaves, and never part of the plant.
#
# Found by tracing a dark slab that survived every wall/floor guard in the
# plant's cut-out: 2,419 pixels of it were "windowpane", a class the surface map
# did not carry at all, so nothing could remove it.
BACKGROUND_CLASSES = ("windowpane", "glass", "sky")

# Glazing on its own, without sky.
#
# A mirrored wall panel is frequently read as "glass" or "windowpane" rather
# than "mirror" — the class map sees a flat reflective sheet and picks the
# commoner label. That makes these classes *supporting* evidence for a mirror,
# never sufficient evidence: they are only consulted when Grounding DINO has
# independently called the thing a mirror. A plain window is labelled the same
# way and must never be reclassified, which is exactly what requiring the
# detector's agreement prevents.
#
# Sky is deliberately excluded. It appears *through* glazing, so admitting it
# would let a view out of a window stand in for a reflective surface.
GLAZING_CLASSES = ("windowpane", "glass")

# Whether the wall/floor/ceiling masks are snapped onto the photograph's own
# edges before anyone uses them.
#
# SegFormer decides at a quarter of its 512 input — a 128x128 grid — and the
# logits are then interpolated up to the photo's size. Interpolation cannot
# invent the boundary it never had, so the floor/wall junction arrives a few
# pixels off wherever the real edge did not happen to fall on that grid. That
# matters twice: `live_scene` solves the camera pitch from the 2nd percentile of
# floor rows, so a boundary a few pixels out is a tile-scale error on every
# uploaded room; and `filtering.structural_map` compensates for the same
# uncertainty by eroding, which costs real object pixels.
#
# `matting.snap_to_edges` is the filter already used on SAM's masks for exactly
# this reason. Measured over six demo rooms, mean gradient magnitude under the
# mask boundary — higher means the boundary sits on a real edge in the photo:
#
#     floor   128.7 -> 171.8   (+33.5%)
#     wall     93.6 -> 138.7   (+48.2%)
#
# Positive in all twelve cases, and only 1.2% of mask pixels move: this corrects
# a boundary, it does not re-segment. Cost is about 0.25s per surface against
# SegFormer's own 1.3s.
#
# Two alternatives were measured and rejected. Feeding the model an
# aspect-preserving input instead of the squashed 512 square (the ONNX graph is
# fully dynamic, so the square is not a constraint) gained +0.0% on floor and
# +12% on wall only at a 768 short side, for 5.6x the time, and made one room
# distinctly worse. Horizontal-flip TTA gained +0.0% on floor and -2.0% on wall.
SNAP_TO_EDGES = True

# How far the snap may move a surface boundary, as a fraction of the photo's
# longest edge. Deliberately the same band `SegmentationConfig` uses for the
# surface trim: the two are describing the same uncertainty.
SNAP_BAND_FRACTION = 0.0075
SNAP_BAND_MIN_PX = 6

# How much of a surface the snap may remove before the result is discarded and
# the original mask kept.
#
# The filter moves a boundary; it is not entitled to delete a surface. On six
# real rooms it changes area by about 0.4%, so anything near this bar is not a
# refined boundary. The case it guards is a surface smaller than the band
# itself: a sliver of ceiling, or any mask whose confident interior vanishes
# under the erosion that defines "well inside", comes back empty. Same
# reasoning as `SegmentationConfig.min_surviving_after_trim`, which reverts the
# surface trim for the same reason.
SNAP_MIN_SURVIVING = 0.75

_lock = threading.Lock()
_session = None
_labels: dict[int, str] = {}


class SurfacesUnavailable(RuntimeError):
    """The ONNX model or runtime is not present, so surfaces cannot be found."""


# Kept for `live_scene`, which describes the floor it builds a camera from in
# these terms. It is a surface record, not an object instance.
@dataclass(frozen=True)
class Instance:
    """One surface: its label, its mask and where it sits."""

    label: str
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    pixels: int


def model_available() -> bool:
    return weights.present(MODEL_PATH, CONFIG_PATH)


def download_model() -> None:
    """Fetch the ONNX weights and label map. Called once, on demand."""
    weights.ensure(MODEL_URL, MODEL_PATH)
    weights.ensure(CONFIG_URL, CONFIG_PATH)


def _load():
    """Open the ONNX session once and cache it, with its label map."""
    global _session, _labels

    with _lock:
        if _session is not None:
            return _session, _labels

        if not model_available():
            download_model()

        try:
            import onnxruntime
        except ImportError as error:  # pragma: no cover - environment guard
            raise SurfacesUnavailable(
                "onnxruntime is not installed; run pip install -r backend/requirements.txt"
            ) from error

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        _session = onnxruntime.InferenceSession(
            str(MODEL_PATH), options, providers=["CPUExecutionProvider"]
        )

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

        _labels = {int(key): value.strip() for key, value in config["id2label"].items()}

        return _session, _labels


def _preprocess(rgb: np.ndarray) -> np.ndarray:
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    normalised = (resized.astype(np.float32) / 255.0 - MEAN) / STD

    return np.transpose(normalised, (2, 0, 1))[None, ...].astype(np.float32)


def _upsample_logits(logits: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize (C, h, w) logits to (C, height, width).

    cv2.resize only reads the trailing axis as channels while it has four or
    fewer, and ADE20K has 150 — so the planes go through in groups of four.
    """
    planes = np.transpose(logits, (1, 2, 0))

    resized = [
        cv2.resize(
            np.ascontiguousarray(planes[..., start : start + 4]),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        ).reshape(height, width, -1)
        for start in range(0, planes.shape[2], 4)
    ]

    return np.concatenate(resized, axis=2)


def label_map(rgb: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    """
    Run the model and return a per-pixel ADE20K class id map at full resolution.

    The network emits logits at a quarter of the input size; they are resized
    back before the argmax so class boundaries follow the original pixels.
    """
    session, labels = _load()

    logits = session.run(None, {session.get_inputs()[0].name: _preprocess(rgb)})[0][0]

    height, width = rgb.shape[:2]

    upsampled = _upsample_logits(logits, width, height)

    return upsampled.argmax(axis=2).astype(np.int32), labels


def confidence_maps(rgb: np.ndarray) -> dict[str, object]:
    """Return native-resolution SegFormer confidence data for diagnostics.

    This deliberately does not participate in `analyse`: the production
    surface masks still use the same argmax and snapping path. The returned
    probabilities stay at the model output resolution so a diagnostic can
    choose and document its own projection to the photo resolution.
    """
    session, labels = _load()

    raw = session.run(None, {session.get_inputs()[0].name: _preprocess(rgb)})[0][0]
    raw = raw.astype(np.float32)
    shifted = raw - raw.max(axis=0, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=0, keepdims=True)

    top_class = probabilities.argmax(axis=0).astype(np.int32)
    top_probability = probabilities.max(axis=0)
    if probabilities.shape[0] > 1:
        partitioned = np.partition(probabilities, -2, axis=0)
        second_probability = partitioned[-2]
    else:
        second_probability = np.zeros_like(top_probability)

    wall_id = next((index for index, label in labels.items() if label == "wall"), None)
    ceiling_id = next((index for index, label in labels.items() if label == "ceiling"), None)
    floor_id = next((index for index, label in labels.items() if label == "floor"), None)

    def selected(index):
        if index is None:
            return None
        return probabilities[index]

    wall_probability = selected(wall_id)
    wall_margin = (
        wall_probability - np.max(
            np.delete(probabilities, wall_id, axis=0), axis=0
        )
        if wall_id is not None
        else None
    )

    return {
        "probabilities": probabilities,
        "top_class": top_class,
        "top_probability": top_probability,
        "second_probability": second_probability,
        "wall_probability": wall_probability,
        "wall_margin": wall_margin,
        "ceiling_probability": selected(ceiling_id),
        "floor_probability": selected(floor_id),
        "labels": labels,
        "class_ids": {"wall": wall_id, "ceiling": ceiling_id, "floor": floor_id},
        "native_shape": [int(raw.shape[1]), int(raw.shape[2])],
    }


def snap(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """
    Pull one surface mask onto the photograph's own edges.

    A no-op when `SNAP_TO_EDGES` is off or the mask is empty, so a caller never
    has to test for either. See `SNAP_TO_EDGES` for what this buys and what was
    measured and rejected instead.
    """
    if not SNAP_TO_EDGES or not mask.any():
        return mask

    band = max(SNAP_BAND_MIN_PX, int(round(SNAP_BAND_FRACTION * max(rgb.shape[:2]))))

    snapped = matting.snap_to_edges(mask.astype(np.float32), rgb, radius=band, band=band)

    # Refusing the result is always safe: the unsnapped mask is what every
    # caller used before this existed.
    if snapped.sum() < SNAP_MIN_SURVIVING * mask.sum():
        return mask

    return snapped


def analyse(
    rgb: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    """
    Everything this module reports, from a single forward pass.

    Returns `(surfaces, mirror, background, glazing, classes, labels)`:

      * `surfaces` holds only the wall/floor/ceiling masks actually present,
        each snapped onto the photo's own edges — see `SNAP_TO_EDGES`;
      * `mirror` is the ADE20K mirror class, empty when the room has none;
      * `background` is glazing and sky — background an object mask can swallow
        that is not one of the three room surfaces;
      * `glazing` is the same without sky, kept apart because it is supporting
        evidence for a mirror and sky is not;
      * `classes` is the full per-pixel ADE20K class id map, and `labels` its
        id -> name lookup.

    All of them come from one `label_map` call because they are readings of the
    same class map, and running the network again for them would multiply the
    cost of the cheapest stage in the pipeline for no benefit.

    The last two are returned for `extraction.residual`. ADE20K has 150 classes
    and this module used to report five of them, discarding the rest — yet the
    discarded ones are precisely the evidence for an object the detector never
    boxed. Measured on one room where a floor lamp's arm and a plant's outer
    leaves survived into the clean room, the class map had already named them:
    6,526 px "plant", 2,004 px "lamp", 1,442 px "pot", none of it in any mask.
    The information was there all along and simply was not being asked for.
    """
    classes, labels = label_map(rgb)

    found: dict[str, np.ndarray] = {}

    mirror = np.zeros(classes.shape, dtype=bool)
    background = np.zeros(classes.shape, dtype=bool)
    glazing = np.zeros(classes.shape, dtype=bool)

    for class_id in np.unique(classes):
        label = labels.get(int(class_id), "")

        mask = classes == class_id

        if not mask.any():
            continue

        if label in SURFACE_CLASSES or label in ARCHITECTURE_CLASSES:
            # Only the three room surfaces are snapped. The architecture classes
            # are reported for structural *rejection*, where a few pixels of
            # boundary do not decide anything, and paying the filter for them
            # would be cost without a use.
            found[label] = snap(mask, rgb) if label in SURFACE_CLASSES else mask
        elif label == MIRROR_CLASS:
            mirror |= mask
        elif label in BACKGROUND_CLASSES:
            background |= mask

        if label in GLAZING_CLASSES:
            glazing |= mask

    return found, mirror, background, glazing, classes, labels


def detect(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """
    The wall / floor / ceiling split for a photo.

    Returns only the surfaces actually present, each as a full-frame boolean
    mask at the photo's own resolution.
    """
    return analyse(rgb)[0]


def instances(rgb: np.ndarray) -> list[Instance]:
    """The same surfaces, in the record shape `live_scene` reads."""
    found = []

    for label, mask in detect(rgb).items():
        ys, xs = np.nonzero(mask)

        if ys.size == 0:
            continue

        found.append(
            Instance(
                label=label,
                mask=mask,
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                pixels=int(mask.sum()),
            )
        )

    return found
