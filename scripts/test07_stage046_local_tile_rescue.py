
from pathlib import Path
import argparse
import json

import torch
from PIL import Image, ImageDraw

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


MODEL_ID = "IDEA-Research/grounding-dino-base"

BOX_THRESHOLD = 0.18
TEXT_THRESHOLD = 0.15
TOP_K = 5


def run(
    master_path,
    inventory_json,
    existence_json,
    tiled_candidates_json,
    tile_dir,
    output_dir,
    model_cache,
):

    master_path = Path(master_path)
    inventory_json = Path(inventory_json)
    existence_json = Path(existence_json)
    tiled_candidates_json = Path(tiled_candidates_json)
    tile_dir = Path(tile_dir)
    output_dir = Path(output_dir)
    model_cache = Path(model_cache)

    output_dir.mkdir(parents=True, exist_ok=True)

    crop_out = output_dir / "rescue_crops"
    crop_out.mkdir(parents=True, exist_ok=True)

    master = Image.open(master_path).convert("RGB")
    W, H = master.size

    inventory = json.loads(
        inventory_json.read_text()
    )["objects"]

    existence = json.loads(
        existence_json.read_text()
    )

    tiled_candidates = json.loads(
        tiled_candidates_json.read_text()
    )

    inventory_by_id = {
        int(x["id"]): x
        for x in inventory
    }

    # ========================================================
    # ONLY Stage04.5 PRESENT INSTANCES
    # ========================================================

    present_ids = {
        int(x["inventory_id"])
        for x in existence
        if x["decision"] == "PRESENT"
    }

    print("=" * 90)
    print("TEST07 STAGE04.6")
    print("LOCAL TILE DINO RESCUE")
    print("=" * 90)

    print("MASTER:", master_path)
    print("PRESENT RESCUE IDS:", sorted(present_ids))

    # ========================================================
    # REGION METADATA
    # ========================================================

    region_bbox = {}

    for row in tiled_candidates:

        region = row.get(
            "source_region"
        )

        bbox = row.get(
            "source_bbox"
        )

        if region and bbox:
            region_bbox[
                region
            ] = bbox

    # ========================================================
    # LOAD DINO
    # ========================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Loading Grounding DINO...")

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID,
            cache_dir=str(model_cache)
        )
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

    print("✅ DINO READY")

    all_candidates = []

    # ========================================================
    # RESCUE EACH PRESENT INSTANCE
    # ========================================================

    for iid in sorted(present_ids):

        obj = inventory_by_id[iid]

        regions = obj.get(
            "source_regions",
            []
        )

        if isinstance(
            regions,
            str
        ):
            regions = [regions]

        if not regions:

            sr = obj.get(
                "source_region",
                ""
            )

            if sr:
                regions = [sr]

        print()
        print("-" * 90)

        print(
            f'{iid:02d}. '
            f'{obj["name"]} | '
            f'{obj["grounding_phrase"]}'
        )

        print(
            "regions:",
            regions
        )

        instance_candidates = []

        for region in regions:

            if region not in region_bbox:

                print(
                    "❌ Missing bbox metadata:",
                    region
                )

                continue

            tile_path = (
                tile_dir
                / f"{region}.png"
            )

            if not tile_path.exists():

                print(
                    "❌ Missing tile:",
                    tile_path
                )

                continue

            tile = (
                Image.open(
                    tile_path
                )
                .convert("RGB")
            )

            tw, th = tile.size

            phrase = (
                obj[
                    "grounding_phrase"
                ]
                .strip()
                .lower()
            )

            # Grounding DINO convention benefits
            # from terminating phrase with period.
            if not phrase.endswith("."):
                phrase += "."

            inputs = processor(
                images=tile,
                text=phrase,
                return_tensors="pt"
            ).to(device)

            with torch.inference_mode():

                outputs = model(
                    **inputs
                )

            processed = (
                processor
                .post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    target_sizes=[
                        (
                            th,
                            tw
                        )
                    ]
                )[0]
            )

            boxes = processed.get(
                "boxes",
                []
            )

            scores = processed.get(
                "scores",
                []
            )

            labels = processed.get(
                "text_labels",
                processed.get(
                    "labels",
                    []
                )
            )

            if torch.is_tensor(boxes):
                boxes = (
                    boxes
                    .detach()
                    .cpu()
                    .numpy()
                )

            if torch.is_tensor(scores):
                scores = (
                    scores
                    .detach()
                    .cpu()
                    .numpy()
                )

            x_offset, y_offset, _, _ = (
                region_bbox[
                    region
                ]
            )

            rows = []

            for j in range(
                len(boxes)
            ):

                local_box = [
                    float(v)
                    for v in boxes[j]
                ]

                lx1, ly1, lx2, ly2 = (
                    local_box
                )

                global_box = [
                    lx1 + x_offset,
                    ly1 + y_offset,
                    lx2 + x_offset,
                    ly2 + y_offset,
                ]

                score = float(
                    scores[j]
                )

                label = (
                    str(labels[j])
                    if j < len(labels)
                    else ""
                )

                row = {
                    "inventory_id":
                        iid,

                    "inventory_name":
                        obj["name"],

                    "grounding_phrase":
                        obj[
                            "grounding_phrase"
                        ],

                    "source_region":
                        region,

                    "source_bbox":
                        region_bbox[
                            region
                        ],

                    "local_bbox":
                        local_box,

                    "bbox":
                        global_box,

                    "score":
                        score,

                    "dino_label":
                        label,

                    "rescue_source":
                        "stage04_5_present_local_tile_dino",
                }

                rows.append(row)

            rows.sort(
                key=lambda r:
                    r["score"],
                reverse=True
            )

            rows = rows[:TOP_K]

            print(
                region,
                "→",
                len(rows),
                "candidate(s)"
            )

            for row in rows:

                print(
                    "  score:",
                    round(
                        row["score"],
                        3
                    ),
                    "| bbox:",
                    [
                        round(v, 1)
                        for v in row["bbox"]
                    ],
                    "| label:",
                    row["dino_label"]
                )

            instance_candidates.extend(
                rows
            )

        instance_candidates.sort(
            key=lambda r:
                r["score"],
            reverse=True
        )

        all_candidates.extend(
            instance_candidates
        )

        # ====================================================
        # BEST-CANDIDATE DIAGNOSTIC CROP
        # ====================================================

        if instance_candidates:

            best = instance_candidates[0]

            x1, y1, x2, y2 = [
                int(round(v))
                for v in best["bbox"]
            ]

            margin = 12

            cx1 = max(
                0,
                x1 - margin
            )

            cy1 = max(
                0,
                y1 - margin
            )

            cx2 = min(
                W,
                x2 + margin
            )

            cy2 = min(
                H,
                y2 + margin
            )

            crop = master.crop(
                (
                    cx1,
                    cy1,
                    cx2,
                    cy2
                )
            )

            draw = ImageDraw.Draw(
                crop
            )

            draw.rectangle(
                [
                    x1 - cx1,
                    y1 - cy1,
                    x2 - cx1,
                    y2 - cy1,
                ],
                outline="red",
                width=2
            )

            crop.save(
                crop_out
                / (
                    f"{iid:02d}_"
                    f'{obj["name"].replace(" ", "_")}'
                    "_best.png"
                )
            )

    # ========================================================
    # SAVE ALL RESCUE PROPOSALS
    # ========================================================

    candidate_path = (
        output_dir
        / "00_local_tile_rescue_candidates.json"
    )

    candidate_path.write_text(
        json.dumps(
            all_candidates,
            indent=2
        )
    )

    # ========================================================
    # PREVIEW
    # ========================================================

    preview = master.copy()

    draw = ImageDraw.Draw(
        preview
    )

    for row in all_candidates:

        x1, y1, x2, y2 = (
            row["bbox"]
        )

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
                x1,
                max(
                    0,
                    y1 - 12
                )
            ),
            (
                f'{row["inventory_id"]}:'
                f'{row["inventory_name"]}'
                f' {row["score"]:.2f}'
            ),
            fill="red"
        )

    preview_path = (
        output_dir
        / "01_local_tile_rescue_preview.png"
    )

    preview.save(
        preview_path
    )

    # ========================================================
    # REPORT
    # ========================================================

    per_instance = {}

    for iid in sorted(present_ids):

        rows = [
            r
            for r in all_candidates
            if int(
                r["inventory_id"]
            ) == iid
        ]

        per_instance[
            str(iid)
        ] = {
            "name":
                inventory_by_id[
                    iid
                ]["name"],

            "candidate_count":
                len(rows),

            "best_score":
                (
                    max(
                        r["score"]
                        for r in rows
                    )
                    if rows
                    else None
                ),
        }

    report = {
        "stage":
            "TEST07_STAGE046",

        "method":
            "Stage04.5 PRESENT instance local-tile Grounding DINO rescue",

        "present_ids":
            sorted(
                present_ids
            ),

        "candidate_count":
            len(
                all_candidates
            ),

        "per_instance":
            per_instance,

        "outputs": {
            "candidate_json":
                str(
                    candidate_path
                ),

            "preview":
                str(
                    preview_path
                ),

            "crop_dir":
                str(
                    crop_out
                ),
        }
    }

    (
        output_dir
        / "stage046_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2
        )
    )

    print()
    print("=" * 90)
    print("TEST07 STAGE04.6 RESULT")
    print("=" * 90)

    print(
        "PRESENT INPUT:",
        len(
            present_ids
        )
    )

    print(
        "RESCUE CANDIDATES:",
        len(
            all_candidates
        )
    )

    print()

    for iid in sorted(
        present_ids
    ):

        info = per_instance[
            str(iid)
        ]

        print(
            f'{iid:02d}. '
            f'{info["name"]} → '
            f'{info["candidate_count"]} candidate(s)'
            f' | best='
            f'{info["best_score"]}'
        )

    print()
    print("CANDIDATES:")
    print(candidate_path)

    print()
    print("PREVIEW:")
    print(preview_path)

    print()
    print(
        "IMPORTANT: RESCUE PROPOSALS ONLY."
    )

    print(
        "NOT YET ADDED TO STAGE04 VALID SET."
    )


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
        "--existence-json",
        required=True
    )

    parser.add_argument(
        "--tiled-candidates-json",
        required=True
    )

    parser.add_argument(
        "--tile-dir",
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

        existence_json=
            args.existence_json,

        tiled_candidates_json=
            args.tiled_candidates_json,

        tile_dir=
            args.tile_dir,

        output_dir=
            args.output_dir,

        model_cache=
            args.model_cache,
    )
