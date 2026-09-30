"""
Per-room results that a tile change must not recompute.

A render changes with the tile (size, artwork, rotation, grout), but the depth
map, the detected lines and vanishing points, the EXIF focal length and the
estimated camera depend only on the room photograph and its masks. `room_memo`
remembers such a result by a hash of the function's inputs -- arrays by their
bytes, shape and dtype -- so a second render of the same room with another
tile reuses it instead of detecting again.

Exact by construction: the key is the full input content, the functions it
wraps are deterministic (fixed seeds), and every hit returns a deep copy, so a
caller can never modify what the next caller gets. An input it cannot hash
(e.g. a caller-supplied random generator) bypasses the memo and computes as
before. `stats()` reports hits and misses per function.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import threading
from collections import OrderedDict

import numpy as np

#: Entries kept per function: a few rooms, so switching between two open rooms
#: stays cached without holding every photo ever rendered.
MAXSIZE = 6

_STATS: dict[str, dict[str, int]] = {}


class _Unhashable(Exception):
    pass


def _feed(h, value) -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        h.update(repr((type(value).__name__, value)).encode())
    elif isinstance(value, bytes):
        h.update(b"bytes"); h.update(hashlib.sha1(value).digest())
    elif isinstance(value, np.ndarray):
        h.update(repr(("nd", value.shape, value.dtype.str)).encode())
        h.update(hashlib.sha1(np.ascontiguousarray(value).tobytes()).digest())
    elif isinstance(value, np.generic):
        _feed(h, value.item())
    elif isinstance(value, (list, tuple)):
        h.update(repr((type(value).__name__, len(value))).encode())
        for item in value:
            _feed(h, item)
    elif isinstance(value, dict):
        h.update(repr(("dict", len(value))).encode())
        for key in sorted(value, key=repr):
            _feed(h, key); _feed(h, value[key])
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        h.update(type(value).__qualname__.encode())
        for field in dataclasses.fields(value):
            _feed(h, field.name); _feed(h, getattr(value, field.name))
    else:
        raise _Unhashable(type(value).__name__)


def room_memo(fn):
    """Memoise a deterministic per-room function by the content of its inputs."""
    name = f"{fn.__module__}.{fn.__qualname__}"
    cache: OrderedDict[str, object] = OrderedDict()
    lock = threading.Lock()
    stats = _STATS.setdefault(name, {"hits": 0, "misses": 0, "bypassed": 0})

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        h = hashlib.sha1(name.encode())
        try:
            _feed(h, args); _feed(h, kwargs)
        except _Unhashable:
            with lock:
                stats["bypassed"] += 1
            return fn(*args, **kwargs)
        key = h.hexdigest()
        with lock:
            if key in cache:
                cache.move_to_end(key)
                stats["hits"] += 1
                return copy.deepcopy(cache[key])
        result = fn(*args, **kwargs)
        with lock:
            stats["misses"] += 1
            cache[key] = copy.deepcopy(result)
            while len(cache) > MAXSIZE:
                cache.popitem(last=False)
        return result

    wrapper.cache_clear = lambda: cache.clear()  # type: ignore[attr-defined]
    return wrapper


def stats() -> dict[str, dict[str, int]]:
    """Hits / misses / bypasses per memoised function since start (or reset)."""
    return {k: dict(v) for k, v in _STATS.items()}


def reset_stats() -> None:
    for v in _STATS.values():
        v.update(hits=0, misses=0, bypassed=0)
