"""
Cut-to-cut room object extraction.

    room photo
        -> Grounding DINO      what objects, and where (boxes + confidence)
        -> SAM 2.1             which exact pixels belong to each box
        -> OpenCV              deterministic mask cleanup
        -> matting             a boundary that composites without a rim
        -> mask union          ALL_OBJECTS and MIRRORS_ONLY
        -> RGBA PNGs           two full-canvas layers, no per-object files

Each stage has one job and is replaceable on its own. Nothing in here invents
pixels: every RGB value in every output is a pixel of the uploaded photograph,
and the only thing computed is how much of each pixel is object.
"""

__all__ = ["ExtractionError", "ExtractedObject", "extract"]


def __getattr__(name: str):
    """
    Re-export the pipeline entry points without importing them at package
    import time. `config` is read by callers that never run inference, and
    loading the heavy modules for that would pull onnxruntime in for nothing.
    """
    if name in __all__:
        from extraction import extractor

        return getattr(extractor, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
