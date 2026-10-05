"""
GeoCalib, run OUTSIDE the backend's environment (a learned focal / gravity cross-check).

    ~/geocalib-env/bin/python backend/geocalib_runner.py <image> [<image> ...]

Prints one JSON line per image: focal_px (for the image as given), hfov_deg,
pitch_deg, roll_deg, their uncertainties, and the seconds it took. The backend
never imports this file (it needs PyTorch, which the backend does not have);
it calls it once per room as a subprocess.
"""

from __future__ import annotations

import json
import math
import sys
import time


def main() -> int:
    import torch
    from geocalib import GeoCalib

    torch.set_grad_enabled(False)
    t0 = time.perf_counter()
    model = GeoCalib()
    load_s = time.perf_counter() - t0
    for path in sys.argv[1:]:
        t = time.perf_counter()
        image = model.load_image(path)                       # (3, H, W) in [0, 1], at the file's size
        h, w = image.shape[-2:]
        result = model.calibrate(image)
        camera, gravity = result["camera"], result["gravity"]
        f = camera.f.reshape(-1).tolist()
        roll, pitch = (math.degrees(float(v)) for v in gravity.rp.reshape(-1).tolist())

        def unc(key, deg=False):
            v = result.get(key)
            if v is None:
                return None
            v = float(v.reshape(-1)[0])
            return round(math.degrees(v) if deg else v, 3)

        fx = float(f[0])
        print(json.dumps({
            "image": path, "size": [int(w), int(h)],
            "focal_px": round(fx, 1),
            "hfov_deg": round(math.degrees(2 * math.atan(w / (2 * fx))), 1),
            "pitch_deg": round(pitch, 2), "roll_deg": round(roll, 2),
            "focal_uncertainty_px": unc("focal_uncertainty"),
            "vfov_uncertainty_deg": unc("vfov_uncertainty", deg=True),
            "pitch_uncertainty_deg": unc("pitch_uncertainty", deg=True),
            "roll_uncertainty_deg": unc("roll_uncertainty", deg=True),
            "seconds": round(time.perf_counter() - t, 2), "model_load_seconds": round(load_s, 2),
        }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
