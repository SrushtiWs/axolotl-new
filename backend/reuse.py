"""
One extraction per room, shared by both endpoints.

`/segment` and `/generate` each used to run the whole pipeline from scratch —
Grounding DINO, SAM 2.1, SegFormer and LaMa, twice, on two slightly different
copies of the same photograph. That cost about ninety seconds a call and, worse,
meant the object boundaries a customer *saw* in `ALL_OBJECTS.png` were not the
boundaries used to put the objects back in the final render. Two independent
runs of a stochastic-in-practice pipeline do not agree pixel for pixel.

This module is the handoff. `/segment` does the inference once and writes what
the renderer needs beside the PNGs; `/generate` loads it and goes straight to
the tile projection. The same accepted masks then drive every output:

    ALL_OBJECTS.png   written from them
    MIRRORS_ONLY.png  written from them
    CLEAN_ROOM.png    inpainted inside their union
    the final render  tiles stop at their union, objects restored through it

Everything in a bundle is stored at the **render resolution** — the photo capped
to `MAX_RENDER_EDGE` — because that is the size the renderer works at, and
resizing a mask at load time would put the boundary back on a different grid
than the one it was decided on. `/segment` still writes its PNGs at the photo's
full size; those are for looking at, and they are cut from the same masks.

A bundle is keyed to the exact bytes of the photograph it came from. A caller
that uploads a different image gets no match and the full pipeline runs, which
is the old behaviour and always correct — the fast path can only ever be taken
when it is provably the same room.

Nothing here is a cache in the sense of "may be stale": a job directory is
written once and never updated, so a bundle either matches its upload or is not
used at all.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from surfaces import Instance

# Subdirectory of a job that holds the handoff.
FOLDER = "reuse"

ROOM = "room.png"
CLEAN = "clean.png"
MASKS = "masks.npz"
META = "meta.json"

# Bundle format. Bumped when the contents change shape, so a job written by an
# older build is ignored rather than misread.
VERSION = 1


@dataclass(frozen=True)
class Bundle:
    """Everything `/generate` needs that `/segment` already computed."""

    room: np.ndarray  # (H, W, 3) uint8 — the photo at render resolution
    clean: np.ndarray  # (H, W, 3) uint8 — objects removed, surfaces rebuilt
    props: np.ndarray  # (H, W) bool — union of every accepted object mask
    surfaces: list[Instance]  # floor/wall/ceiling, detected on `clean`
    meta: dict


def digest(data: bytes) -> str:
    """The identity of an uploaded photograph: a hash of its exact bytes."""
    return hashlib.sha256(data).hexdigest()


def _pack(mask: np.ndarray) -> np.ndarray:
    """A boolean mask as bits, which is 8x smaller on disk and exact."""
    return np.packbits(mask.astype(bool), axis=None)


def _unpack(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    count = int(shape[0]) * int(shape[1])

    return np.unpackbits(packed, count=count).astype(bool).reshape(shape)


def save(
    job_dir: Path,
    room: np.ndarray,
    clean: np.ndarray,
    props: np.ndarray,
    surfaces: list[Instance],
    meta: dict,
) -> Path:
    """
    Write the handoff for one job. Returns the folder it went into.

    `room`, `clean` and `props` must already be at the render resolution and
    the same shape as each other; this writes them as they are.
    """
    folder = job_dir / FOLDER

    folder.mkdir(parents=True, exist_ok=True)

    Image.fromarray(room, mode="RGB").save(folder / ROOM)
    Image.fromarray(clean, mode="RGB").save(folder / CLEAN)

    arrays = {"props": _pack(props)}

    for surface in surfaces:
        arrays[f"surface_{surface.label}"] = _pack(surface.mask)

    np.savez_compressed(folder / MASKS, **arrays)

    payload = {
        "version": VERSION,
        "shape": [int(props.shape[0]), int(props.shape[1])],
        "surfaces": [
            {
                "label": surface.label,
                "bbox": list(surface.bbox),
                "pixels": int(surface.pixels),
            }
            for surface in surfaces
        ],
        **meta,
    }

    (folder / META).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return folder


def load(job_dir: Path, source: str | None = None) -> Bundle | None:
    """
    Read a job's handoff, or `None` if there is nothing usable.

    `source` is the digest of the photograph the caller is holding. When it is
    given and does not match the one recorded, this returns `None`: a bundle
    belongs to one image and reusing it for another would tile a room using
    another room's masks.

    Every failure mode returns `None` rather than raising. The fast path is an
    optimisation, and the pipeline behind it is always available.
    """
    folder = job_dir / FOLDER

    try:
        meta = json.loads((folder / META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    if meta.get("version") != VERSION:
        return None

    if source is not None and meta.get("source") != source:
        return None

    try:
        room = np.array(Image.open(folder / ROOM).convert("RGB"))
        clean = np.array(Image.open(folder / CLEAN).convert("RGB"))

        with np.load(folder / MASKS) as arrays:
            shape = (int(meta["shape"][0]), int(meta["shape"][1]))

            props = _unpack(arrays["props"], shape)

            surfaces = [
                Instance(
                    label=entry["label"],
                    mask=_unpack(arrays[f"surface_{entry['label']}"], shape),
                    bbox=tuple(entry["bbox"]),
                    pixels=int(entry["pixels"]),
                )
                for entry in meta.get("surfaces", [])
                if f"surface_{entry['label']}" in arrays
            ]
    except (OSError, ValueError, KeyError):
        return None

    if room.shape[:2] != props.shape or clean.shape[:2] != props.shape:
        return None

    return Bundle(room=room, clean=clean, props=props, surfaces=surfaces, meta=meta)


def fit(rgb: np.ndarray, longest_edge: int) -> np.ndarray:
    """
    Downscale to the render resolution. No upscaling.

    Deliberately `cv2.INTER_AREA`, matching `extraction.extractor._fit_within`
    rather than the PIL LANCZOS path in `app._fit_within`: the masks in a bundle
    were decided on the image this produces, and the renderer has to be looking
    at the same pixels they were decided on.
    """
    height, width = rgb.shape[:2]

    scale = longest_edge / max(height, width)

    if scale >= 1.0:
        return rgb

    return cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
