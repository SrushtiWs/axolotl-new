
from pathlib import Path
from PIL import Image, ImageDraw
import argparse
import json
import gc
import torch

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)

MODEL_ID = "IDEA-Research/grounding-dino-base"

BOX_THRESHOLD = 0.18
TEXT_THRESHOLD = 0.15

REGION_3X3 = {
    "top_left":      (0, 0),
    "top_center":    (1, 0),
    "top_right":     (2, 0),
    "middle_left":   (0, 1),
    "middle_center": (1, 1),
    "middle_right":  (2, 1),
    "bottom_left":   (0, 2),
    "bottom_center": (1, 2),
    "bottom_right":  (2, 2),
}


def get_3x3_bbox(region, W, H):

    col, row = REGION_3X3[region]

    base_w = W / 3.0
    base_h = H / 3.0

    mx = int(base_w * 0.18)
    my = int(base_h * 0.18)

    x1 = max(
        0,
        int(col * base_w) - mx
    )

    y1 = max(
        0,
        int(row * base_h) - my
    )

    x2 = min(
        W,
        int((col + 1) * base_w) + mx
    )

    y2 = min(
        H,
        int((row + 1) * base_h) + my
    )

    return [x1, y1, x2, y2]


def get_source_bbox(
    obj,
    W,
    H,
    bbox4
):

    existing = obj.get(
        "source_bbox"
    )

    if existing is not None:

        return [
            int(round(float(v)))
            for v in existing
        ]

    region = str(
        obj.get(
            "source_region",
            ""
        )
    )

    pass_name = str(
        obj.get(
            "discovery_pass",
            ""
        )
    )

    if (
        pass_name == "3x3"
        and
        region in REGION_3X3
    ):

        return get_3x3_bbox(
            region,
            W,
            H
        )

    if (
        pass_name == "4x4"
        and
        region in bbox4
    ):

        return [
            int(round(float(v)))
            for v in bbox4[region]
        ]

    return None


def run(
    master_path,
    inventory_json,
    manifest_4x4,
    output_dir,
    model_cache
):

    master_path = Path(master_path)
    inventory_json = Path(inventory_json)
    manifest_4x4 = Path(manifest_4x4)
    output_dir = Path(output_dir)
    model_cache = Path(model_cache)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for p in [
        master_path,
        inventory_json,
        manifest_4x4,
        model_cache
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    master = Image.open(
        master_path
    ).convert("RGB")

    W, H = master.size

    inventory = json.loads(
        inventory_json.read_text()
    )

    objects = inventory.get(
        "objects",
        []
    )

    if not isinstance(
        objects,
        list
    ):
        raise RuntimeError(
            'inventory["objects"] must be a list'
        )

    manifest4 = json.loads(
        manifest_4x4.read_text()
    )

    bbox4 = {
        x["region"]: x["bbox"]
        for x in manifest4
    }

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 110)
    print("TEST07 - STAGE03 V4.5")
    print("FULL SOURCE-CROP CONSTRAINED GROUNDING")
    print("=" * 110)

    print("MASTER      :", master_path)
    print("RESOLUTION  :", W, "x", H)
    print("OBSERVATIONS:", len(objects))
    print("DEVICE      :", device)
    print("MODEL       :", MODEL_ID)
    print("DTYPE       : float32")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=str(model_cache)
    )

    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(
            MODEL_ID,
            cache_dir=str(model_cache)
        )
        .to(device)
    )

    model.eval()

    print(
        "MODEL DTYPE :",
        next(model.parameters()).dtype
    )

    results = []

    preview = master.copy()

    draw = ImageDraw.Draw(
        preview
    )

    observations_with_candidates = 0
    observations_without_candidates = 0

    for index, obj in enumerate(
        objects,
        start=1
    ):

        oid = int(
            obj.get(
                "id",
                index
            )
        )

        name = str(
            obj.get(
                "name",
                "object"
            )
        ).strip()

        source_bbox = get_source_bbox(
            obj,
            W,
            H,
            bbox4
        )

        print()
        print("-" * 110)

        print(
            f"[{index}/{len(objects)}] "
            f"ID={oid:03d} | "
            f"{obj.get('discovery_pass')} | "
            f"{obj.get('source_region')} | "
            f"{name}"
        )

        if source_bbox is None:

            print(
                "NO SOURCE BBOX — SKIPPED"
            )

            results.append({
                "inventory_id":
                    oid,

                "name":
                    name,

                "grounding_phrase":
                    obj.get(
                        "grounding_phrase",
                        name
                    ),

                "confidence":
                    obj.get(
                        "confidence",
                        "low"
                    ),

                "discovery_pass":
                    obj.get(
                        "discovery_pass"
                    ),

                "source_region":
                    obj.get(
                        "source_region"
                    ),

                "source_bbox":
                    None,

                "query":
                    name,

                "candidate_count":
                    0,

                "candidates":
                    [],

                "status":
                    "NO_SOURCE_BBOX"
            })

            observations_without_candidates += 1

            continue

        sx1, sy1, sx2, sy2 = (
            source_bbox
        )

        crop = master.crop(
            (
                sx1,
                sy1,
                sx2,
                sy2
            )
        )

        query = (
            name
            .rstrip(".")
            +
            "."
        )

        inputs = processor(
            images=crop,
            text=query,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():

            outputs = model(
                **inputs
            )

        target_sizes = torch.tensor(
            [
                [
                    crop.height,
                    crop.width
                ]
            ],
            device=device
        )

        processed = (
            processor
            .post_process_grounded_object_detection(
                outputs,
                input_ids=
                    inputs["input_ids"],
                threshold=
                    BOX_THRESHOLD,
                text_threshold=
                    TEXT_THRESHOLD,
                target_sizes=
                    target_sizes
            )[0]
        )

        boxes = (
            processed["boxes"]
            .detach()
            .cpu()
            .tolist()
        )

        scores = (
            processed["scores"]
            .detach()
            .cpu()
            .tolist()
        )

        text_labels = processed.get(
            "text_labels",
            processed.get(
                "labels",
                []
            )
        )

        candidates = []

        for rank, (
            box,
            score
        ) in enumerate(
            zip(
                boxes,
                scores
            ),
            start=1
        ):

            cx1, cy1, cx2, cy2 = [
                float(v)
                for v in box
            ]

            master_bbox = [
                cx1 + sx1,
                cy1 + sy1,
                cx2 + sx1,
                cy2 + sy1
            ]

            label = (
                str(
                    text_labels[
                        rank - 1
                    ]
                )
                if rank - 1 < len(text_labels)
                else name
            )

            candidates.append({
                "rank":
                    rank,

                "score":
                    float(score),

                "crop_bbox":
                    [
                        cx1,
                        cy1,
                        cx2,
                        cy2
                    ],

                "bbox":
                    master_bbox,

                "dino_label":
                    label
            })

        candidates = sorted(
            candidates,
            key=lambda x:
                x["score"],
            reverse=True
        )

        # Re-rank after score sorting.
        for rank, candidate in enumerate(
            candidates,
            start=1
        ):
            candidate["rank"] = rank

        if candidates:
            observations_with_candidates += 1
        else:
            observations_without_candidates += 1

        results.append({
            "inventory_id":
                oid,

            "name":
                name,

            "grounding_phrase":
                obj.get(
                    "grounding_phrase",
                    name
                ),

            "confidence":
                obj.get(
                    "confidence",
                    "low"
                ),

            "discovery_pass":
                obj.get(
                    "discovery_pass"
                ),

            "source_region":
                obj.get(
                    "source_region"
                ),

            "source_bbox":
                source_bbox,

            "query":
                query,

            "candidate_count":
                len(candidates),

            "candidates":
                candidates,

            "status":
                (
                    "GROUNDED"
                    if candidates
                    else
                    "NO_DINO_CANDIDATE"
                )
        })

        print(
            "SOURCE BBOX:",
            source_bbox
        )

        print(
            "QUERY      :",
            query
        )

        print(
            "CANDIDATES :",
            len(candidates)
        )

        for c in candidates[:3]:

            print(
                f'   #{c["rank"]} '
                f'score={c["score"]:.4f} '
                f'bbox='
                f'{[round(v,1) for v in c["bbox"]]}'
            )

        # ====================================================
        # PREVIEW
        #
        # Blue = discovery crop
        # Red  = top DINO box
        # ====================================================

        draw.rectangle(
            [
                sx1,
                sy1,
                sx2,
                sy2
            ],
            outline="blue",
            width=1
        )

        if candidates:

            x1, y1, x2, y2 = [
                int(round(v))
                for v in candidates[0]["bbox"]
            ]

            draw.rectangle(
                [
                    x1,
                    y1,
                    x2,
                    y2
                ],
                outline="red",
                width=2
            )

            draw.text(
                (
                    x1 + 1,
                    max(
                        0,
                        y1 - 11
                    )
                ),
                f"{oid}:{name}",
                fill="red"
            )

        del inputs
        del outputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # FLATTEN CANDIDATES
    # ========================================================

    flat_candidates = []

    for row in results:

        for candidate in row[
            "candidates"
        ]:

            flat_candidates.append({
                "inventory_id":
                    row["inventory_id"],

                "inventory_name":
                    row["name"],

                "grounding_phrase":
                    row[
                        "grounding_phrase"
                    ],

                "qwen_confidence":
                    row[
                        "confidence"
                    ],

                "discovery_pass":
                    row[
                        "discovery_pass"
                    ],

                "source_region":
                    row[
                        "source_region"
                    ],

                "source_bbox":
                    row[
                        "source_bbox"
                    ],

                "rank_for_observation":
                    candidate[
                        "rank"
                    ],

                "score":
                    candidate[
                        "score"
                    ],

                "bbox":
                    candidate[
                        "bbox"
                    ],

                "dino_label":
                    candidate[
                        "dino_label"
                    ]
            })

    # ========================================================
    # SAVE
    # ========================================================

    result_path = (
        output_dir /
        "01_source_crop_grounding_by_observation.json"
    )

    flat_path = (
        output_dir /
        "02_source_crop_dino_candidates_flat.json"
    )

    preview_path = (
        output_dir /
        "03_source_crop_grounding_preview.png"
    )

    report_path = (
        output_dir /
        "stage03_v45_report.json"
    )

    result_path.write_text(
        json.dumps(
            results,
            indent=2
        )
    )

    flat_path.write_text(
        json.dumps(
            flat_candidates,
            indent=2
        )
    )

    preview.save(
        preview_path
    )

    report = {
        "stage":
            "TEST07_STAGE03_V45_FULL_SOURCE_CROP_GROUNDING",

        "method":
            "Qwen multiscale observations grounded only inside their discovery crop using concise object-name Grounding DINO query",

        "master":
            str(master_path),

        "inventory_json":
            str(inventory_json),

        "manifest_4x4":
            str(manifest_4x4),

        "model":
            MODEL_ID,

        "box_threshold":
            BOX_THRESHOLD,

        "text_threshold":
            TEXT_THRESHOLD,

        "observation_count":
            len(objects),

        "observations_with_candidates":
            observations_with_candidates,

        "observations_without_candidates":
            observations_without_candidates,

        "total_dino_candidates":
            len(flat_candidates),

        "guarantees": [
            "3x3 and 4x4 provenance preserved",
            "DINO searches only original discovery crop",
            "DINO query uses concise physical object identity",
            "DINO crop coordinates mapped back to master coordinates",
            "no cross-observation deduplication",
            "no cross-scale deduplication",
            "no SAM2",
            "no mask modification"
        ],

        "outputs": {
            "per_observation":
                str(result_path),

            "flat_candidates":
                str(flat_path),

            "preview":
                str(preview_path)
        }
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2
        )
    )

    print()
    print("=" * 110)
    print("STAGE03 V4.5 SUMMARY")
    print("=" * 110)

    print(
        "OBSERVATIONS              :",
        len(objects)
    )

    print(
        "WITH DINO CANDIDATES      :",
        observations_with_candidates
    )

    print(
        "WITHOUT DINO CANDIDATES   :",
        observations_without_candidates
    )

    print(
        "TOTAL DINO CANDIDATES     :",
        len(flat_candidates)
    )

    print()
    print(
        "RESULT:",
        result_path
    )

    print(
        "FLAT  :",
        flat_path
    )

    print(
        "PREVIEW:",
        preview_path
    )

    print(
        "REPORT:",
        report_path
    )

    del model
    del processor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--master",
        required=True
    )

    parser.add_argument(
        "--inventory-json",
        required=True
    )

    parser.add_argument(
        "--manifest-4x4",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "--model-cache",
        default=
            "/workspace/data/huggingface-cache"
    )

    args = parser.parse_args()

    run(
        master_path=
            args.master,

        inventory_json=
            args.inventory_json,

        manifest_4x4=
            args.manifest_4x4,

        output_dir=
            args.output_dir,

        model_cache=
            args.model_cache
    )
