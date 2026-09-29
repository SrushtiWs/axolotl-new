"""
On-demand model weight downloads.

Every ONNX file this backend runs is fetched once, on first use, into
backend/models/ and reused from there. Downloads go to a .part file and are
renamed only once complete, so an interrupted fetch never leaves a truncated
model that onnxruntime would fail on later in a much more confusing way.
"""

from __future__ import annotations

import threading
from pathlib import Path
from urllib.request import urlopen

MODELS = Path(__file__).resolve().parent / "models"

_lock = threading.Lock()


def ensure(url: str, destination: Path) -> Path:
    """Download `url` to `destination` unless it is already there."""
    if destination.exists():
        return destination

    with _lock:
        # Another thread may have finished it while this one waited.
        if destination.exists():
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)

        partial = destination.with_suffix(destination.suffix + ".part")

        with urlopen(url) as response, partial.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)

        partial.replace(destination)

    return destination


def present(*paths: Path) -> bool:
    return all(path.exists() for path in paths)
