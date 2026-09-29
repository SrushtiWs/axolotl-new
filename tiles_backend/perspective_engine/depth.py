"""
Depth for the wall pipeline: MiDaS DPT-Hybrid-384, on onnxruntime.

The reference project ran MiDaS DPT-Hybrid-384 under PyTorch
(`segmentation/depth_module.estimate_depth_heatmap`) and its `/depth`
endpoint turned the result into a Turbo-coloured PNG, which the renderer then
decoded back into [0, 1] with `core.composite.decode_depth_array`. PyTorch
cannot run on this machine, so this module runs the SAME network — the
`Intel/dpt-hybrid-midas` weights, exported to ONNX as `Xenova/dpt-hybrid-midas`
— and reproduces every step around it, including the colour round trip, so the
wall pipeline sees the depth it was written against.

Reproduced exactly from the reference project:

  * preprocessing (`midas/model_loader.py`, model_type "dpt_hybrid_384"):
    bicubic resize, NormalizeImage(mean 0.5, std 0.5), HWC -> CHW float32.

    Including one quirk: `estimate_depth_heatmap` hands the transform the
    uint8 RGB image, not the [0, 1] float image MiDaS normally expects, so the
    normalisation produces values in [-1, 509] rather than [-1, 1]. That is
    what the reference project's walls were tuned on, so it is kept.

  * postprocessing: bicubic resize to the image size, min-max normalise,
    `(depth * 255).astype(uint8)` (truncation, not rounding), COLORMAP_TURBO.

  * the per-surface re-normalisation `/depth` applied when the frontend sent a
    surface mask (which it always did): decode, re-spread inside the mask,
    re-encode with Turbo.

One deliberate difference, forced by the export: its position embeddings are
fixed at 24x24 patches, so it only accepts a 384x384 input, where the
reference project kept the aspect ratio and interpolated the embeddings (a
1600x1200 room ran at 512x384). Measured on the reference project's own
input/output pair (`input_vp/room5.png` -> `depth_heatmap.png`), the decoded
depth this module produces differs from the reference by 0.0043 on average
(p95 0.0138) on a 0..1 scale, correlation 0.99983. Feeding the textbook [0, 1]
input instead of the uint8 quirk doubles that error (0.0096), which is what
confirms the quirk is what the reference project really ran.

Both pipelines need it. The wall splits into separate walls and picks each
wall's direction from depth. The floor takes its plane from the vanishing
point's horizon when that is accepted — there depth only sets the plane's
distance, which cancels out — but falls back to a RANSAC fit on depth when the
vanishing point is rejected, which is common.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import numpy as np

from .core.composite import decode_depth_array, normalise_depth_within_mask

MODEL_FILENAME = "midas_dpt_hybrid_384.onnx"

MODEL_URL = "https://huggingface.co/Xenova/dpt-hybrid-midas/resolve/main/onnx/model.onnx"

# The export's fixed input size. See the module docstring.
NET_SIZE = 384

_session = None
_lock = threading.Lock()


class DepthUnavailable(RuntimeError):
    """The depth model cannot run. Carries a reason fit to show a user."""


def _models_dir() -> Path:
    """The backend's shared weights folder, so every model lives in one place."""
    return Path(__file__).resolve().parents[2] / "backend" / "models"


def model_path() -> Path:
    return _models_dir() / MODEL_FILENAME


def available() -> bool:
    return model_path().exists()


def _load():
    global _session

    with _lock:
        if _session is not None:
            return _session

        path = model_path()

        if not path.exists():
            # Reuse the backend's downloader when it is importable, so this
            # model is fetched exactly like every other one.
            try:
                backend = str(_models_dir().parent)

                if backend not in sys.path:
                    sys.path.insert(0, backend)

                import weights

                weights.ensure(MODEL_URL, path)
            except Exception as error:
                raise DepthUnavailable(
                    f"MiDaS depth weights are missing ({path}) and could not be "
                    f"downloaded: {error}. Wall tiling needs them; floor tiling does not."
                ) from error

        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        _session = onnxruntime.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )

        return _session


def _preprocess(rgb: np.ndarray) -> np.ndarray:
    resized = cv2.resize(rgb, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_CUBIC)

    # NormalizeImage(mean=0.5, std=0.5) applied to uint8 values, as the
    # reference project did. See the module docstring.
    normalised = (resized - 0.5) / 0.5

    return np.ascontiguousarray(np.transpose(normalised, (2, 0, 1))).astype(np.float32)[None]


def estimate_depth_heatmap(rgb: np.ndarray) -> np.ndarray:
    """
    The reference project's `estimate_depth_heatmap`: an RGB image in, a
    Turbo-coloured BGR depth heatmap out, the size of the input.
    """
    session = _load()

    prediction = session.run(None, {session.get_inputs()[0].name: _preprocess(rgb)})[0]

    prediction = np.squeeze(prediction).astype(np.float32)

    height, width = rgb.shape[:2]

    depth = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_CUBIC)

    depth = (depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-12)

    return cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def surface_depth_heatmap(heatmap_bgr: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """
    The reference `/depth` endpoint's per-surface map: the whole-frame
    heatmap, decoded, re-spread across the surface mask, re-encoded.

    The frontend sent the surface mask with every depth request, so this is
    the map the reference renderer actually received. Pixels outside the mask
    are left as they are (`mask_outside` defaulted to false).
    """
    if mask_bool is None or not np.any(mask_bool):
        return heatmap_bgr

    depth_val = decode_depth_array(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB))

    depth_val = normalise_depth_within_mask(depth_val, mask_bool)

    return cv2.applyColorMap(
        (np.clip(depth_val, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
