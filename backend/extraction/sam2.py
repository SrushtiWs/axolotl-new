"""
SAM 2.1 (Hiera-L) through onnxruntime, on the CPU.

This replaces the SAM ViT-B export the backend used to run. That export took
point prompts only — it has no `input_boxes` at all — which is why the old
pipeline had to reduce a region to a handful of interior points and hope SAM
grew them back into the right object. This one takes boxes directly:

    vision_encoder          pixel_values (1, 3, 1024, 1024)
                                -> image_embeddings.0  (1, 32, 256, 256)
                                -> image_embeddings.1  (1, 64, 128, 128)
                                -> image_embeddings.2  (1, 256, 64, 64)

    prompt_encoder_mask_decoder
                            input_boxes (1, N, 4)  + the three embeddings
                                -> pred_masks           (1, N, 3, h, w)
                                -> iou_scores           (1, N, 3)
                                -> object_score_logits  (1, N, 1)

So a Grounding DINO box becomes a SAM 2 box prompt with no reinterpretation in
between, which is the relationship the pipeline is specified around.

The encoder runs **once per photo** and every box reuses its embeddings; the
decoder runs once for a batch of boxes. Two inference calls for a whole room,
not two per object.

SAM 2 resizes to a square 1024 without preserving aspect ratio, so mapping
coordinates is a plain per-axis scale in both directions and there is no
letterbox padding to crop back off.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

import weights

MODEL_DIR = weights.MODELS / "sam2_large"

# These filenames are load-bearing: each graph stores its weights in a sibling
# `.onnx_data` file and refers to it by this exact name, so the pair cannot be
# renamed independently.
ENCODER_PATH = MODEL_DIR / "vision_encoder.onnx"
ENCODER_DATA_PATH = MODEL_DIR / "vision_encoder.onnx_data"
DECODER_PATH = MODEL_DIR / "prompt_encoder_mask_decoder.onnx"
DECODER_DATA_PATH = MODEL_DIR / "prompt_encoder_mask_decoder.onnx_data"

_BASE = "https://huggingface.co/onnx-community/sam2.1-hiera-large-ONNX/resolve/main/onnx"

DOWNLOADS = (
    (f"{_BASE}/vision_encoder.onnx", ENCODER_PATH),
    (f"{_BASE}/vision_encoder.onnx_data", ENCODER_DATA_PATH),
    (f"{_BASE}/prompt_encoder_mask_decoder.onnx", DECODER_PATH),
    (f"{_BASE}/prompt_encoder_mask_decoder.onnx_data", DECODER_DATA_PATH),
)

IMAGE_SIZE = 1024

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# How many boxes go through the decoder in one call. The decoder is cheap
# relative to the encoder, and batching keeps the intermediate mask tensor to a
# size that comfortably fits in memory for a busy room.
DECODE_BATCH = 8

_lock = threading.Lock()
_encoder = None
_decoder = None


class Sam2Unavailable(RuntimeError):
    """The SAM 2 weights or onnxruntime are not present."""


@dataclass(frozen=True)
class Embedding:
    """One encoded photo. Every box prompt for that photo reuses this."""

    features: tuple[np.ndarray, np.ndarray, np.ndarray]
    shape: tuple[int, int]  # (height, width) of the original photo

    @property
    def scale_x(self) -> float:
        return IMAGE_SIZE / float(self.shape[1])

    @property
    def scale_y(self) -> float:
        return IMAGE_SIZE / float(self.shape[0])


def available() -> bool:
    return weights.present(ENCODER_PATH, ENCODER_DATA_PATH, DECODER_PATH, DECODER_DATA_PATH)


def download() -> None:
    for url, destination in DOWNLOADS:
        weights.ensure(url, destination)


def _load():
    """Open both sessions once per process and cache them."""
    global _encoder, _decoder

    with _lock:
        if _encoder is not None:
            return _encoder, _decoder

        if not available():
            download()

        try:
            import onnxruntime
        except ImportError as error:  # pragma: no cover - environment guard
            raise Sam2Unavailable(
                "onnxruntime is not installed; run pip install -r backend/requirements.txt"
            ) from error

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            _encoder = onnxruntime.InferenceSession(
                str(ENCODER_PATH), options, providers=["CPUExecutionProvider"]
            )
            _decoder = onnxruntime.InferenceSession(
                str(DECODER_PATH), options, providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise Sam2Unavailable(f"SAM 2 weights could not be loaded: {error}") from error

        return _encoder, _decoder


def encode(rgb: np.ndarray) -> Embedding:
    """Run the vision encoder once for a photo."""
    encoder, _ = _load()

    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    normalised = (resized.astype(np.float32) / 255.0 - MEAN) / STD

    pixel_values = np.transpose(normalised, (2, 0, 1))[None].astype(np.float32)

    outputs = encoder.run(None, {"pixel_values": pixel_values})

    return Embedding(features=tuple(outputs[:3]), shape=rgb.shape[:2])


def _feeds(decoder, embedding: Embedding, boxes: np.ndarray) -> dict:
    """
    Assemble the decoder inputs for a batch of boxes.

    Every graph input has to be supplied, including the point prompts this call
    is not using, so those go in empty — a zero-length points axis, which the
    prompt encoder reads as "no point prompts" rather than as a point at the
    origin.
    """
    names = {item.name for item in decoder.get_inputs()}

    feeds = {
        "image_embeddings.0": embedding.features[0],
        "image_embeddings.1": embedding.features[1],
        "image_embeddings.2": embedding.features[2],
        "input_boxes": boxes[None].astype(np.float32),
    }

    if "input_points" in names:
        feeds["input_points"] = np.zeros((1, 1, 0, 2), dtype=np.float32)

    if "input_labels" in names:
        feeds["input_labels"] = np.zeros((1, 1, 0), dtype=np.int64)

    return {name: value for name, value in feeds.items() if name in names}


def masks_for_boxes(
    embedding: Embedding, boxes: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment every box against one encoded photo.

    `boxes` are in the photo's own pixel coordinates. Returns

        logits  (N, 3, h, w) float32  raw mask logits, at the decoder's own
                                      resolution — not yet the photo's
        scores  (N, 3)       float32  SAM's predicted IoU per candidate
        objects (N,)         float32  SAM's object-presence logit per box

    The masks are deliberately left small and un-thresholded. Selection happens
    first, and only the one chosen candidate per box is ever resized up to the
    photo's resolution, which for a 4000 px photo is the difference between
    resizing three masks per object and resizing one.
    """
    _, decoder = _load()

    if not boxes:
        empty = np.zeros((0, 3, 1, 1), dtype=np.float32)

        return empty, np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    scaled = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    scaled[:, [0, 2]] *= embedding.scale_x
    scaled[:, [1, 3]] *= embedding.scale_y

    all_logits: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_objects: list[np.ndarray] = []

    for start in range(0, len(scaled), DECODE_BATCH):
        batch = scaled[start : start + DECODE_BATCH]

        iou_scores, pred_masks, object_logits = decoder.run(
            None, _feeds(decoder, embedding, batch)
        )

        all_logits.append(np.asarray(pred_masks[0], dtype=np.float32))
        all_scores.append(np.asarray(iou_scores[0], dtype=np.float32))
        all_objects.append(np.asarray(object_logits[0], dtype=np.float32).reshape(-1))

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_objects, axis=0),
    )


def logits_at_full_resolution(logits: np.ndarray, embedding: Embedding) -> np.ndarray:
    """
    One candidate's logit grid, resized to the photo's exact size.

    The *magnitude* is kept, not just the sign. A logit barely above zero is a
    pixel SAM is unsure about, and that is where a mask bleeds into the wall or
    floor behind an object; deep inside the object the logits are large. Mask
    refinement uses that difference to tell contamination from real object, so
    the values have to survive the resize rather than being thresholded away
    first.
    """
    height, width = embedding.shape

    return cv2.resize(
        logits.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )


def to_full_resolution(logits: np.ndarray, embedding: Embedding, threshold: float) -> np.ndarray:
    """
    One candidate's logit grid -> a boolean mask at the photo's exact size.

    Resizing the logits and thresholding afterwards keeps the boundary where
    the model put it; thresholding first and resizing the binary mask would
    quantise the edge to the decoder's grid and then interpolate the staircase.
    """
    return logits_at_full_resolution(logits, embedding) > threshold
