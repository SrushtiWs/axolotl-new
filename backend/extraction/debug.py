"""
Debug renders of every stage.

When a cut-out comes out wrong the question is always *which stage* was wrong:
did Grounding DINO put the box in the wrong place, did SAM segment the wrong
thing inside a correct box, or did the cleanup eat the object? A final PNG
cannot answer that. These eight images can, and they are written only when
debug mode is on, because they cost a second or two and a few megabytes.

    01_dino_boxes.png         every raw detection, before filtering
    02_dino_labels.png        the detections that survived filtering, labelled
    03_sam_masks.png          SAM's masks, as returned, before cleanup
    04_clean_masks.png        the same masks after OpenCV cleanup
    05_mirror_masks.png       what was classed as a mirror
    06_object_overlay.png     every accepted object over the room
    07_all_objects_preview.png   ALL_OBJECTS.png on a checkerboard
    08_mirrors_preview.png       MIRRORS_ONLY.png on a checkerboard

Previews are drawn on a checkerboard rather than on white or black, because
against a flat colour a soft alpha edge and a hard one look identical — which
is precisely the thing being inspected.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from extraction.extractor import ExtractionResult, to_original

# A stable, well-separated palette, indexed by object number.
PALETTE = np.array(
    [
        [239, 71, 111],
        [255, 209, 102],
        [6, 214, 160],
        [17, 138, 178],
        [155, 93, 229],
        [241, 91, 181],
        [254, 228, 64],
        [0, 187, 249],
        [155, 246, 255],
        [253, 138, 138],
    ],
    dtype=np.uint8,
)

CHECKER = 16


def _colour(index: int) -> np.ndarray:
    return PALETTE[index % len(PALETTE)]


def _canvas(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()


def _save(path: Path, bgr: np.ndarray) -> None:
    cv2.imwrite(str(path), bgr)


def _label(canvas: np.ndarray, text: str, x: int, y: int, colour) -> None:
    """A label with a dark plate behind it, so it stays readable over any room."""
    scale = max(0.4, min(1.0, canvas.shape[1] / 1600.0))

    thickness = max(1, int(round(scale * 2)))

    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

    top = max(0, y - height - 6)

    cv2.rectangle(canvas, (x, top), (x + width + 6, top + height + 6), (0, 0, 0), cv2.FILLED)

    cv2.putText(
        canvas,
        text,
        (x + 3, top + height + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        tuple(int(c) for c in colour),
        thickness,
        cv2.LINE_AA,
    )


def _checkerboard(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape

    ys, xs = np.mgrid[0:height, 0:width]

    tiles = ((ys // CHECKER) + (xs // CHECKER)) % 2

    return np.where(tiles[..., None] == 0, 210, 165).astype(np.uint8).repeat(3, axis=2)


def _over_checkerboard(rgba: np.ndarray) -> np.ndarray:
    board = _checkerboard(rgba.shape[:2]).astype(np.float32)

    alpha = (rgba[..., 3:4].astype(np.float32)) / 255.0

    blended = rgba[..., :3].astype(np.float32) * alpha + board * (1.0 - alpha)

    return cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_RGB2BGR)


def _mask_overlay(rgb: np.ndarray, masks: list[tuple[str, np.ndarray]]) -> np.ndarray:
    canvas = _canvas(rgb).astype(np.float32)

    for index, (_, mask) in enumerate(masks):
        if not mask.any():
            continue

        tint = _colour(index)[::-1].astype(np.float32)  # RGB palette -> BGR canvas

        canvas[mask] = canvas[mask] * 0.45 + tint * 0.55

    result = canvas.astype(np.uint8)

    for index, (label, mask) in enumerate(masks):
        if not mask.any():
            continue

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        colour = tuple(int(c) for c in _colour(index)[::-1])

        cv2.drawContours(result, contours, -1, colour, 2)

        ys, xs = np.nonzero(mask)

        _label(result, label, int(xs.min()), int(ys.min()), colour)

    return result


def write(
    work_rgb: np.ndarray,
    result: ExtractionResult,
    destination: Path,
) -> list[str]:
    """
    Render every debug image. `work_rgb` is the working-resolution photo, which
    is the resolution every mask in `result` is at.

    Never raises: a debug render failing must not lose the extraction it was
    meant to explain.
    """
    destination.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    def record(name: str) -> None:
        written.append(name)

    trace = result.trace

    try:
        # 01 — every raw detection, before anything was filtered out.
        canvas = _canvas(work_rgb)

        for index, detection in enumerate(trace.get("raw_detections", [])):
            x0, y0, x1, y1 = detection.box

            cv2.rectangle(canvas, (x0, y0), (x1, y1), tuple(int(c) for c in _colour(index)[::-1]), 1)

        _save(destination / "01_dino_boxes.png", canvas)
        record("01_dino_boxes.png")

        # 02 — what survived filtering, with labels and confidences.
        canvas = _canvas(work_rgb)

        for index, detection in enumerate(trace.get("kept_detections", [])):
            x0, y0, x1, y1 = detection.box

            colour = tuple(int(c) for c in _colour(index)[::-1])

            cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)

            _label(canvas, f"{detection.label} {detection.confidence:.2f}", x0, y0, colour)

        _save(destination / "02_dino_labels.png", canvas)
        record("02_dino_labels.png")

        # 03 — SAM's masks as returned, before cleanup.
        _save(
            destination / "03_sam_masks.png",
            _mask_overlay(work_rgb, list(trace.get("sam_masks", []))),
        )
        record("03_sam_masks.png")

        # 04 — the same masks after the OpenCV chain.
        _save(
            destination / "04_clean_masks.png",
            _mask_overlay(work_rgb, [(item.label, item.mask) for item in result.objects]),
        )
        record("04_clean_masks.png")

        # 05 — mirrors only, so a misclassified window is obvious.
        mirrors = [(item.label, item.mask) for item in result.objects if item.is_mirror]

        _save(
            destination / "05_mirror_masks.png",
            _mask_overlay(work_rgb, mirrors) if mirrors else _canvas(work_rgb),
        )
        record("05_mirror_masks.png")

        # 06 — accepted objects with their ids, the summary view.
        _save(
            destination / "06_object_overlay.png",
            _mask_overlay(
                work_rgb,
                [(f"{item.identifier} {item.label}", item.mask) for item in result.objects],
            ),
        )
        record("06_object_overlay.png")
    except Exception:  # pragma: no cover - debug output is never load-bearing
        pass

    return written


def write_layer_previews(destination: Path, layer_dir: Path, filenames: tuple[str, str]) -> list[str]:
    """
    Render the two finished layers over a checkerboard.

    Separate from `write` because these read the files that were actually
    saved — so what is inspected is the bytes on disk, not an in-memory array
    that might differ from them.
    """
    destination.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    targets = (
        (filenames[0], "07_all_objects_preview.png"),
        (filenames[1], "08_mirrors_preview.png"),
    )

    for source_name, preview_name in targets:
        source = layer_dir / source_name

        if not source.exists():
            continue

        try:
            rgba = np.asarray(Image.open(source).convert("RGBA"), dtype=np.uint8)

            _save(destination / preview_name, _over_checkerboard(rgba))

            written.append(preview_name)
        except Exception:  # pragma: no cover - debug output is never load-bearing
            continue

    return written
