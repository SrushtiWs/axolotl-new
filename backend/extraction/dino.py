"""
Grounding DINO through onnxruntime, on the CPU.

This is the live detector: it runs on every uploaded photo and answers the two
questions the rest of the pipeline is built on — *what objects are present* and
*where are they*. Nothing here is precomputed, cached from a previous run, or
read back from a committed JSON. The only inputs are the photograph and the
configured prompt list.

The export is `onnx-community/grounding-dino-tiny-ONNX`:

    pixel_values     (1, 3, 800, 800)   float32, ImageNet-normalised
    pixel_mask       (1, 800, 800)      int64, all ones — the square is full
    input_ids        (1, T)             int64, BERT word-piece ids
    token_type_ids   (1, T)             int64, zeros
    attention_mask   (1, T)             int64, ones
        ->
    logits           (1, 900, 256)      raw, per query per text position
    pred_boxes       (1, 900, 4)        cx, cy, w, h, normalised to [0, 1]

The image goes in as a plain 800x800 resize — the processor config says
`{"width": 800, "height": 800}`, not shortest-edge — so aspect ratio is not
preserved going in, and the normalised boxes coming out map straight back onto
the original photo's own width and height. That is what keeps coordinates exact
without any letterbox arithmetic.

Scoring follows the reference post-processing: sigmoid the logits, and score a
query against a phrase by the strongest text position belonging to that phrase.
A query's label is the phrase it scores highest on.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import weights
from extraction.config import ExtractionConfig

MODEL_PATH = weights.MODELS / "grounding_dino_tiny.onnx"
TOKENIZER_PATH = weights.MODELS / "grounding_dino_tiny_tokenizer.json"

_BASE = "https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main"

MODEL_URL = f"{_BASE}/onnx/model.onnx"
TOKENIZER_URL = f"{_BASE}/tokenizer.json"

# The export's fixed spatial input.
INPUT_SIZE = 800

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Text positions the model attends over. A prompt list longer than this is run
# in several passes rather than silently truncated — losing the tail of the list
# would mean quietly never looking for the objects at the end of it.
MAX_TEXT_TOKENS = 256

_lock = threading.Lock()
_session = None
_tokenizer = None


class DinoUnavailable(RuntimeError):
    """The Grounding DINO weights, tokenizer or runtime are not present."""


@dataclass(frozen=True)
class Detection:
    """One thing Grounding DINO found, in the photo's own pixel coordinates."""

    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2

    # Other phrases that fired on this same object and were merged into it.
    # Carried rather than discarded because the winning label is the *strongest*
    # phrase, not necessarily the most informative one: a mirrored door scores
    # "door" and "mirror door" separately, and if "door" wins, only the alias
    # still says the thing is a mirror.
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """Every phrase that fired on this object, winner first."""
        return (self.label, *self.aliases)

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.box
        return max(0, x1 - x0) * max(0, y1 - y0)


def available() -> bool:
    return weights.present(MODEL_PATH, TOKENIZER_PATH)


def download() -> None:
    weights.ensure(MODEL_URL, MODEL_PATH)
    weights.ensure(TOKENIZER_URL, TOKENIZER_PATH)


def _load():
    """Open the session and tokenizer once per process, and cache both."""
    global _session, _tokenizer

    with _lock:
        if _session is not None:
            return _session, _tokenizer

        if not available():
            download()

        try:
            import onnxruntime
        except ImportError as error:  # pragma: no cover - environment guard
            raise DinoUnavailable(
                "onnxruntime is not installed; run pip install -r backend/requirements.txt"
            ) from error

        try:
            from tokenizers import Tokenizer
        except ImportError as error:  # pragma: no cover - environment guard
            raise DinoUnavailable(
                "tokenizers is not installed; run pip install -r backend/requirements.txt"
            ) from error

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            _session = onnxruntime.InferenceSession(
                str(MODEL_PATH), options, providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise DinoUnavailable(f"Grounding DINO weights could not be loaded: {error}") from error

        _tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

        return _session, _tokenizer


def _preprocess(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Photo -> the fixed 800x800 normalised tensor, plus its all-ones mask."""
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    normalised = (resized.astype(np.float32) / 255.0 - MEAN) / STD

    pixel_values = np.transpose(normalised, (2, 0, 1))[None].astype(np.float32)

    pixel_mask = np.ones((1, INPUT_SIZE, INPUT_SIZE), dtype=np.int64)

    return pixel_values, pixel_mask


def _phrase_spans(phrases: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """
    Build the prompt string and record each phrase's character span in it.

    Character spans, not token indices, because the tokenizer reports offsets
    back into this same string — so mapping a token to the phrase it came from
    is a range check and needs no assumption about how word-pieces split.
    """
    text = ""
    spans: list[tuple[int, int]] = []

    for phrase in phrases:
        start = len(text)

        text += phrase

        spans.append((start, len(text)))

        text += ". "

    return text.strip(), spans


def _token_phrase_index(encoding, spans: list[tuple[int, int]], count: int) -> np.ndarray:
    """
    For every token position, which phrase it belongs to — or -1 for none.

    Separators, [CLS] and [SEP] map to -1, so punctuation can never carry a
    score into a phrase.
    """
    owner = np.full(count, -1, dtype=np.int32)

    for position, (start, end) in enumerate(encoding.offsets):
        if position >= count:
            break

        if end <= start:  # special tokens report an empty span
            continue

        for index, (phrase_start, phrase_end) in enumerate(spans):
            if start >= phrase_start and end <= phrase_end:
                owner[position] = index
                break

    return owner


def _chunk(phrases: tuple[str, ...], tokenizer, limit: int) -> list[list[str]]:
    """
    Split the prompt list into passes.

    Two bounds apply, and the second is the one that matters.

    The text window is a hard limit: more than `MAX_TEXT_TOKENS` positions and
    the tail of the list would be silently truncated, so the system would simply
    never look for the objects at the end of it.

    `limit` is a much tighter *accuracy* bound. Because this export has no text
    self-attention mask, every phrase in a pass dilutes every other one, and a
    long pass drives real objects below the box threshold — a lamp measured at
    0.696 in a short pass scored under 0.15 with 64 prompts sharing the
    sequence. Splitting the list is what recovers them. See
    `DetectionConfig.max_phrases_per_pass` for the measurements and the cost.
    """
    limit = max(1, int(limit))

    chunks: list[list[str]] = []

    current: list[str] = []

    for phrase in phrases:
        candidate = current + [phrase]

        text, _ = _phrase_spans(candidate)

        too_long = len(tokenizer.encode(text).ids) > MAX_TEXT_TOKENS

        if current and (too_long or len(candidate) > limit):
            chunks.append(current)
            current = [phrase]
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def _run_pass(
    session,
    tokenizer,
    pixel_values: np.ndarray,
    pixel_mask: np.ndarray,
    phrases: list[str],
    shape: tuple[int, int],
    config: ExtractionConfig,
) -> list[Detection]:
    """One forward pass over one chunk of the prompt list."""
    height, width = shape

    text, spans = _phrase_spans(phrases)

    encoding = tokenizer.encode(text)

    ids = np.asarray(encoding.ids, dtype=np.int64)[None]

    feeds = {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "input_ids": ids,
        "token_type_ids": np.zeros_like(ids),
        "attention_mask": np.ones_like(ids),
    }

    supplied = {item.name for item in session.get_inputs()}

    logits, boxes = session.run(None, {k: v for k, v in feeds.items() if k in supplied})

    # (queries, text_positions) in [0, 1].
    scores = 1.0 / (1.0 + np.exp(-logits[0].astype(np.float64)))

    owner = _token_phrase_index(encoding, spans, scores.shape[1])

    detections: list[Detection] = []

    for index in range(len(phrases)):
        columns = np.nonzero(owner == index)[0]

        if columns.size == 0:
            continue

        span = scores[:, columns]

        # A phrase scores by the *mean* of its word pieces, not the strongest
        # one. The reference implementation can take the max because it feeds a
        # text self-attention mask that isolates each phrase; this export has no
        # such input, so every phrase shares one sequence and a word shared
        # between phrases carries its score into all of them. Under max, "table
        # lamp" inherits "table" exactly and the two become indistinguishable —
        # measured on a real photo, both scored 0.212. The mean requires every
        # word of a phrase to be supported, which separated them to 0.212 and
        # 0.111.
        phrase_scores = span.mean(axis=1)

        # The text threshold then acts as a floor on the phrase's best word, so
        # a phrase whose words are all weakly present cannot accumulate its way
        # past the box threshold.
        phrase_scores = np.where(
            span.max(axis=1) >= config.detection.text_threshold, phrase_scores, 0.0
        )

        for query in np.nonzero(phrase_scores >= config.detection.box_threshold)[0]:
            cx, cy, bw, bh = boxes[0, query].astype(np.float64)

            x0 = int(round((cx - bw / 2.0) * width))
            y0 = int(round((cy - bh / 2.0) * height))
            x1 = int(round((cx + bw / 2.0) * width))
            y1 = int(round((cy + bh / 2.0) * height))

            x0 = max(0, min(width - 1, x0))
            y0 = max(0, min(height - 1, y0))
            x1 = max(x0 + 1, min(width, x1))
            y1 = max(y0 + 1, min(height, y1))

            detections.append(
                Detection(
                    label=phrases[index],
                    confidence=float(phrase_scores[query]),
                    box=(x0, y0, x1, y1),
                )
            )

    return detections


def _scan(
    session,
    tokenizer,
    rgb: np.ndarray,
    config: ExtractionConfig,
    chunks: list[list[str]],
) -> list[Detection]:
    """Every prompt chunk against one image, in that image's own coordinates."""
    pixel_values, pixel_mask = _preprocess(rgb)

    shape = rgb.shape[:2]

    found: list[Detection] = []

    for phrases in chunks:
        found.extend(
            _run_pass(session, tokenizer, pixel_values, pixel_mask, phrases, shape, config)
        )

    return found


def _slices(
    height: int, width: int, rows: int, columns: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """
    Overlapping crops covering the frame.

    Steps are shortened by `overlap` so neighbouring slices share a margin: an
    object lying across a seam is cut in both slices but whole in neither unless
    they overlap, and a half object is exactly what the acceptance checks throw
    away. The last slice in each direction is pinned to the far edge so the
    frame is covered even when the step does not divide evenly.
    """
    rows = max(1, int(rows))
    columns = max(1, int(columns))

    overlap = float(np.clip(overlap, 0.0, 0.9))

    tile_h = int(round(height / (rows - overlap * (rows - 1)))) if rows > 1 else height
    tile_w = int(round(width / (columns - overlap * (columns - 1)))) if columns > 1 else width

    tile_h = max(1, min(height, tile_h))
    tile_w = max(1, min(width, tile_w))

    step_y = max(1, int(round(tile_h * (1.0 - overlap))))
    step_x = max(1, int(round(tile_w * (1.0 - overlap))))

    boxes: list[tuple[int, int, int, int]] = []

    # The final row and column are pinned to the far edge rather than stepped to
    # it, so integer rounding cannot leave an uncovered strip along the bottom
    # or the right-hand side of the frame.
    for row in range(rows):
        y0 = max(0, height - tile_h) if row == rows - 1 else min(row * step_y, max(0, height - tile_h))

        for column in range(columns):
            x0 = (
                max(0, width - tile_w)
                if column == columns - 1
                else min(column * step_x, max(0, width - tile_w))
            )

            box = (x0, y0, min(width, x0 + tile_w), min(height, y0 + tile_h))

            if box not in boxes:
                boxes.append(box)

    return boxes


def detect(rgb: np.ndarray, config: ExtractionConfig) -> list[Detection]:
    """
    Run Grounding DINO over the photo and return every detection above
    threshold, in the photo's own pixel coordinates.

    The prompt list is asked in several short passes rather than one long one.
    The photo is preprocessed once and every pass sees the identical tensor, so
    the boxes all land in the same coordinate space and the passes differ only
    in which words were on offer. `filtering.deduplicate` then collapses the
    phrases that fired on one object, exactly as it already did for phrases
    within a single pass.

    Duplicate handling, structural rejection and size sanity are deliberately
    not done here — this function's whole job is "what did the model say".
    `filtering.py` decides what survives.
    """
    session, tokenizer = _load()

    rules = config.detection

    chunks = _chunk(config.prompts, tokenizer, rules.max_phrases_per_pass)

    # The whole frame first. This is the pass that finds the large objects, and
    # on a small photo it is the only one that runs.
    detections = _scan(session, tokenizer, rgb, config, chunks)

    height, width = rgb.shape[:2]

    if max(height, width) < rules.sahi_min_edge:
        detections.sort(key=lambda item: item.confidence, reverse=True)

        return detections

    # Sliced inference. Each crop is scanned at its own scale, so an object that
    # the 800x800 resize had shrunk below the threshold is seen at something
    # closer to its true size. Boxes come back in crop coordinates and are
    # offset into the frame's.
    for x0, y0, x1, y1 in _slices(
        height, width, rules.sahi_rows, rules.sahi_columns, rules.sahi_overlap
    ):
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])

        tile_area = max(1, (x1 - x0) * (y1 - y0))

        for found in _scan(session, tokenizer, crop, config, chunks):
            if found.area / tile_area >= rules.sahi_max_box_tile_fraction:
                # The crop's own border, not an object's.
                continue

            bx0, by0, bx1, by1 = found.box

            detections.append(
                Detection(
                    label=found.label,
                    confidence=found.confidence,
                    box=(bx0 + x0, by0 + y0, bx1 + x0, by1 + y0),
                    aliases=found.aliases,
                )
            )

    detections.sort(key=lambda item: item.confidence, reverse=True)

    return detections
