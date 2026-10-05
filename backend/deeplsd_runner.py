"""
DeepLSD line segments, run OUTSIDE the backend's environment.

    ~/geocalib-env/bin/python backend/deeplsd_runner.py <image> <out.json> [weights.tar]

Writes {"lines": [[x1, y1, x2, y2], ...], "seconds": ...} for the image at its
own size. The backend never imports this (DeepLSD needs PyTorch).
"""

from __future__ import annotations

import json
import os
import sys
import time


def main() -> int:
    import cv2
    import numpy as np
    import torch
    from deeplsd.models.deeplsd_inference import DeepLSD

    image, out = sys.argv[1], sys.argv[2]
    weights = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/geocalib-env/weights/deeplsd_md.tar")
    t = time.perf_counter()
    conf = {"detect_lines": True,
            "line_detection_params": {"merge": False, "filtering": True, "grad_thresh": 3, "grad_nfa": True}}
    net = DeepLSD(conf)
    net.load_state_dict(torch.load(weights, map_location="cpu")["model"])
    net.eval()
    gray = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    with torch.no_grad():
        pred = net({"image": torch.tensor(gray, dtype=torch.float)[None, None] / 255.0})
    lines = np.asarray(pred["lines"][0]).reshape(-1, 4).tolist()      # [[x1, y1, x2, y2], ...]
    json.dump({"image": image, "size": [int(gray.shape[1]), int(gray.shape[0])], "lines": lines,
               "seconds": round(time.perf_counter() - t, 2)}, open(out, "w"))
    print(json.dumps({"lines": len(lines), "seconds": round(time.perf_counter() - t, 2)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
