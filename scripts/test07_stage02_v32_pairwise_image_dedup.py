
from pathlib import Path
import argparse
import json
import re
from itertools import combinations

import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info


# ============================================================
# TEST07 — STAGE02-V3.2
#
# PAIRWISE IMAGE-GROUNDED DUPLICATE AUDIT
#
# INPUT:
#   Existing tiled Stage02-V3 candidate inventory
#
# PURPOSE:
#   Determine which records from overlapping tiles describe
#   the SAME physical instance.
#
# IMPORTANT:
#   Qwen NEVER rewrites the entire inventory.
#
#   It only answers:
#
#       SAME
#       DIFFERENT
#
#   Python performs the actual merge.
#
# CONSEQUENCE:
#   No candidate can disappear because of generation truncation.
# ============================================================


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

BATCH_SIZE = 16

MAX_NEW_TOKENS = 260

MIN_TEXT_SIMILARITY = 0.22


# ============================================================
# REGION GRID
# ============================================================

REGION_POSITIONS = {
    "top_left":       (0, 0),
    "top_center":     (1, 0),
    "top_right":      (2, 0),

    "middle_left":    (0, 1),
    "middle_center":  (1, 1),
    "middle_right":   (2, 1),

    "bottom_left":    (0, 2),
    "bottom_center":  (1, 2),
    "bottom_right":   (2, 2),
}


# ============================================================
# PROMPT
# ============================================================

PAIR_PROMPT = """
You are auditing duplicate object candidates from overlapping
image crops of ONE real scene.

You are shown the FULL scene image.

Each numbered PAIR contains two candidate descriptions.

For EACH pair, decide whether candidate A and candidate B refer
to the SAME physical instance visible in the image.

Rules:

1. SAME means both records describe one single physical object
   seen from overlapping crops.

2. DIFFERENT means they describe two separately visible physical
   objects.

3. Objects may have the same generic class and still be DIFFERENT.
   Example: two separate pots must remain different.

4. Different wording may still describe the SAME object.
   Example:
   "pendant light" and "brass hanging lamp"
   may refer to one fixture.

5. Use the full image as the primary evidence.

6. Use source-region names as supporting evidence.

7. Do NOT invent objects.

8. Do NOT merge merely because names are similar.

9. Be conservative:
   if there is not enough visual evidence that they are the same
   physical instance, return DIFFERENT.

Return ONLY:

pair_id || SAME_or_DIFFERENT || confidence

confidence must be:
high
medium
or
low

Example:

1 || SAME || high
2 || DIFFERENT || high

No JSON.
No explanation.
No markdown.
""".strip()


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):

    text = str(text).lower().strip()

    text = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def token_set(text):

    return set(
        normalize_text(text).split()
    )


def text_similarity(a, b):

    ta = token_set(
        f'{a.get("name", "")} '
        f'{a.get("grounding_phrase", "")}'
    )

    tb = token_set(
        f'{b.get("name", "")} '
        f'{b.get("grounding_phrase", "")}'
    )

    if not ta or not tb:
        return 0.0

    intersection = len(
        ta & tb
    )

    union = len(
        ta | tb
    )

    if union == 0:
        return 0.0

    return (
        intersection
        /
        union
    )


# ============================================================
# REGION RELATION
# ============================================================

def neighboring_regions(
    region_a,
    region_b
):

    if (
        region_a not in REGION_POSITIONS
        or
        region_b not in REGION_POSITIONS
    ):
        return False

    if region_a == region_b:
        return False

    ax, ay = REGION_POSITIONS[
        region_a
    ]

    bx, by = REGION_POSITIONS[
        region_b
    ]

    dx = abs(
        ax - bx
    )

    dy = abs(
        ay - by
    )

    # Same row/column neighbors or diagonally touching crops.
    return (
        dx <= 1
        and
        dy <= 1
    )


# ============================================================
# PAIR GENERATION
# ============================================================

def build_candidate_pairs(
    candidates
):

    pairs = []

    pair_id = 1

    for ia, ib in combinations(
        range(len(candidates)),
        2
    ):

        a = candidates[ia]
        b = candidates[ib]

        region_a = a.get(
            "source_region",
            ""
        )

        region_b = b.get(
            "source_region",
            ""
        )

        # Duplicate candidates are most plausible when crops
        # overlap spatially.
        if not neighboring_regions(
            region_a,
            region_b
        ):
            continue

        name_a = normalize_text(
            a.get(
                "name",
                ""
            )
        )

        name_b = normalize_text(
            b.get(
                "name",
                ""
            )
        )

        similarity = text_similarity(
            a,
            b
        )

        same_name = (
            name_a
            and
            name_a == name_b
        )

        # Shortlist only plausible duplicate pairs.
        if (
            not same_name
            and
            similarity < MIN_TEXT_SIMILARITY
        ):
            continue

        pairs.append({
            "pair_id":
                pair_id,

            "index_a":
                ia,

            "index_b":
                ib,

            "similarity":
                round(
                    similarity,
                    4
                ),

            "candidate_a":
                a,

            "candidate_b":
                b,
        })

        pair_id += 1

    return pairs


# ============================================================
# QWEN OUTPUT PARSER
# ============================================================

def parse_pair_decisions(
    raw
):

    decisions = {}

    for line in raw.splitlines():

        line = line.strip()

        if not line:
            continue

        if "||" not in line:
            continue

        parts = [
            p.strip()
            for p in line.split("||")
        ]

        if len(parts) != 3:
            continue

        raw_id, decision, confidence = parts

        match = re.search(
            r"\d+",
            raw_id
        )

        if not match:
            continue

        pair_id = int(
            match.group()
        )

        decision = (
            decision
            .upper()
            .strip()
        )

        if decision not in {
            "SAME",
            "DIFFERENT"
        }:
            continue

        confidence = (
            confidence
            .lower()
            .strip()
        )

        if confidence not in {
            "high",
            "medium",
            "low"
        }:
            confidence = "medium"

        decisions[
            pair_id
        ] = {
            "decision":
                decision,

            "confidence":
                confidence
        }

    return decisions


# ============================================================
# QWEN BATCH
# ============================================================

def run_qwen_pair_batch(
    model,
    processor,
    master_path,
    pairs
):

    lines = []

    for pair in pairs:

        a = pair[
            "candidate_a"
        ]

        b = pair[
            "candidate_b"
        ]

        lines.append(
            (
                f'PAIR {pair["pair_id"]}\n'
                f'A: {a.get("name", "")} | '
                f'{a.get("grounding_phrase", "")} | '
                f'region={a.get("source_region", "")}\n'
                f'B: {b.get("name", "")} | '
                f'{b.get("grounding_phrase", "")} | '
                f'region={b.get("source_region", "")}\n'
            )
        )

    batch_text = (
        PAIR_PROMPT
        +
        "\n\nPAIRS:\n"
        +
        "\n".join(
            lines
        )
    )

    messages = [
        {
            "role":
                "user",

            "content": [
                {
                    "type":
                        "image",

                    "image":
                        str(
                            master_path
                        )
                },
                {
                    "type":
                        "text",

                    "text":
                        batch_text
                }
            ]
        }
    ]

    chat_text = (
        processor
        .apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    )

    image_inputs, video_inputs = (
        process_vision_info(
            messages
        )
    )

    inputs = processor(
        text=[
            chat_text
        ],
        images=
            image_inputs,
        videos=
            video_inputs,
        padding=
            True,
        return_tensors=
            "pt"
    )

    inputs = inputs.to(
        model.device
    )

    with torch.inference_mode():

        generated = model.generate(
            **inputs,
            max_new_tokens=
                MAX_NEW_TOKENS,
            do_sample=
                False,
            repetition_penalty=
                1.05
        )

    trimmed = [
        output_ids[
            len(input_ids):
        ]
        for input_ids, output_ids
        in zip(
            inputs.input_ids,
            generated
        )
    ]

    raw = (
        processor
        .batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    )

    del inputs
    del generated
    del trimmed

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return raw


# ============================================================
# UNION FIND
# ============================================================

class UnionFind:

    def __init__(
        self,
        n
    ):

        self.parent = list(
            range(n)
        )

        self.rank = [
            0
        ] * n


    def find(
        self,
        x
    ):

        while (
            self.parent[x]
            != x
        ):

            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )

            x = self.parent[x]

        return x


    def union(
        self,
        a,
        b
    ):

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if (
            self.rank[ra]
            <
            self.rank[rb]
        ):

            ra, rb = rb, ra

        self.parent[
            rb
        ] = ra

        if (
            self.rank[ra]
            ==
            self.rank[rb]
        ):

            self.rank[
                ra
            ] += 1


# ============================================================
# MERGED RECORD
# ============================================================

def confidence_rank(
    confidence
):

    return {
        "high":
            3,

        "medium":
            2,

        "low":
            1,
    }.get(
        str(
            confidence
        ).lower(),
        0
    )


def choose_representative(
    group
):

    # Prefer:
    # 1. higher confidence
    # 2. more descriptive grounding phrase

    ranked = sorted(
        group,
        key=lambda row: (
            confidence_rank(
                row.get(
                    "confidence",
                    ""
                )
            ),
            len(
                str(
                    row.get(
                        "grounding_phrase",
                        ""
                    )
                )
            )
        ),
        reverse=True
    )

    best = dict(
        ranked[0]
    )

    regions = sorted(
        set(
            row.get(
                "source_region",
                ""
            )
            for row in group
            if row.get(
                "source_region",
                ""
            )
        )
    )

    best[
        "source_regions"
    ] = regions

    best[
        "merged_candidate_count"
    ] = len(
        group
    )

    # We no longer need single-source-only metadata.
    best.pop(
        "source_region",
        None
    )

    best.pop(
        "source_bbox",
        None
    )

    return best


# ============================================================
# MAIN
# ============================================================

def run(
    master_path,
    candidate_json,
    output_dir,
    model_cache
):

    master_path = Path(
        master_path
    )

    candidate_json = Path(
        candidate_json
    )

    output_dir = Path(
        output_dir
    )

    model_cache = Path(
        model_cache
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_cache.mkdir(
        parents=True,
        exist_ok=True
    )

    if not master_path.exists():
        raise FileNotFoundError(
            master_path
        )

    if not candidate_json.exists():
        raise FileNotFoundError(
            candidate_json
        )

    candidates = json.loads(
        candidate_json.read_text()
    )

    if not isinstance(
        candidates,
        list
    ):
        raise RuntimeError(
            "Candidate JSON must contain a list."
        )

    print("=" * 90)
    print("TEST07 STAGE02-V3.2")
    print("PAIRWISE IMAGE-GROUNDED DEDUP")
    print("=" * 90)

    print(
        "INPUT CANDIDATES:",
        len(
            candidates
        )
    )

    pairs = build_candidate_pairs(
        candidates
    )

    print(
        "PLAUSIBLE DUPLICATE PAIRS:",
        len(
            pairs
        )
    )

    (
        output_dir
        / "00_candidate_pairs.json"
    ).write_text(
        json.dumps(
            pairs,
            indent=2
        )
    )

    if not pairs:

        print()
        print(
            "No duplicate pairs require audit."
        )

        final_objects = [
            dict(row)
            for row in candidates
        ]

        for i, row in enumerate(
            final_objects,
            start=1
        ):
            row["id"] = i

    else:

        dtype = (
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )

        print()
        print(
            "Loading Qwen 7B..."
        )

        model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                MODEL_ID,
                torch_dtype=
                    dtype,
                device_map=
                    "auto",
                cache_dir=
                    str(
                        model_cache
                    )
            )
        )

        model.eval()

        processor = (
            AutoProcessor
            .from_pretrained(
                MODEL_ID,
                cache_dir=
                    str(
                        model_cache
                    )
            )
        )

        print(
            "✅ QWEN READY"
        )

        all_decisions = {}

        raw_dir = (
            output_dir
            / "pair_batches"
        )

        raw_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        total_batches = (
            len(pairs)
            +
            BATCH_SIZE
            -
            1
        ) // BATCH_SIZE

        for batch_index in range(
            total_batches
        ):

            start = (
                batch_index
                *
                BATCH_SIZE
            )

            end = min(
                len(pairs),
                start
                +
                BATCH_SIZE
            )

            batch = pairs[
                start:end
            ]

            print()
            print(
                f"BATCH {batch_index + 1}"
                f"/{total_batches}"
                f" | pairs {start + 1}-{end}"
            )

            raw = run_qwen_pair_batch(
                model,
                processor,
                master_path,
                batch
            )

            (
                raw_dir
                /
                f"batch_{batch_index + 1:03d}.txt"
            ).write_text(
                raw,
                encoding="utf-8"
            )

            decisions = (
                parse_pair_decisions(
                    raw
                )
            )

            print(
                "Decisions parsed:",
                len(
                    decisions
                ),
                "/",
                len(
                    batch
                )
            )

            all_decisions.update(
                decisions
            )

        # ----------------------------------------------------
        # Union SAME pairs
        # ----------------------------------------------------

        uf = UnionFind(
            len(
                candidates
            )
        )

        audit_rows = []

        same_count = 0
        different_count = 0
        unresolved_count = 0

        for pair in pairs:

            pid = pair[
                "pair_id"
            ]

            decision_info = (
                all_decisions.get(
                    pid
                )
            )

            if decision_info is None:

                decision = (
                    "UNRESOLVED"
                )

                confidence = (
                    "unknown"
                )

                unresolved_count += 1

            else:

                decision = (
                    decision_info[
                        "decision"
                    ]
                )

                confidence = (
                    decision_info[
                        "confidence"
                    ]
                )

                if (
                    decision
                    ==
                    "SAME"
                ):

                    same_count += 1

                    uf.union(
                        pair[
                            "index_a"
                        ],
                        pair[
                            "index_b"
                        ]
                    )

                else:

                    different_count += 1

            audit_rows.append({
                "pair_id":
                    pid,

                "index_a":
                    pair[
                        "index_a"
                    ],

                "index_b":
                    pair[
                        "index_b"
                    ],

                "similarity":
                    pair[
                        "similarity"
                    ],

                "decision":
                    decision,

                "confidence":
                    confidence,

                "candidate_a":
                    pair[
                        "candidate_a"
                    ],

                "candidate_b":
                    pair[
                        "candidate_b"
                    ],
            })

        (
            output_dir
            / "01_pairwise_audit.json"
        ).write_text(
            json.dumps(
                audit_rows,
                indent=2
            )
        )

        # ----------------------------------------------------
        # Build instance groups
        # ----------------------------------------------------

        groups = {}

        for index, row in enumerate(
            candidates
        ):

            root = uf.find(
                index
            )

            groups.setdefault(
                root,
                []
            ).append(
                row
            )

        final_objects = []

        for root in sorted(
            groups.keys()
        ):

            final_objects.append(
                choose_representative(
                    groups[
                        root
                    ]
                )
            )

        for i, row in enumerate(
            final_objects,
            start=1
        ):

            row[
                "id"
            ] = i

    # ========================================================
    # SAVE FINAL
    # ========================================================

    inventory = {
        "method":
            "pairwise image-grounded duplicate audit",

        "model":
            MODEL_ID,

        "input_candidate_count":
            len(
                candidates
            ),

        "plausible_pair_count":
            len(
                pairs
            ),

        "final_inventory_count":
            len(
                final_objects
            ),

        "objects":
            final_objects,
    }

    inventory_path = (
        output_dir
        / "02_pairwise_verified_inventory.json"
    )

    inventory_path.write_text(
        json.dumps(
            inventory,
            indent=2
        )
    )

    report_lines = [
        "=" * 90,
        "TEST07 STAGE02-V3.2 RESULT",
        "=" * 90,
        f"INPUT CANDIDATES: {len(candidates)}",
        f"PLAUSIBLE DUPLICATE PAIRS: {len(pairs)}",
        f"FINAL INVENTORY: {len(final_objects)}",
        "",
        "OBJECT LIST:"
    ]

    for row in final_objects:

        regions = row.get(
            "source_regions",
            [
                row.get(
                    "source_region",
                    ""
                )
            ]
        )

        report_lines.append(
            f'{row["id"]:03d}. '
            f'{row.get("name", "")} | '
            f'{row.get("confidence", "")} | '
            f'{row.get("grounding_phrase", "")} | '
            f'sources={regions} | '
            f'merged={row.get("merged_candidate_count", 1)}'
        )

    report_path = (
        output_dir
        / "03_inventory_report.txt"
    )

    report_path.write_text(
        "\n".join(
            report_lines
        )
    )

    report = {
        "stage":
            "TEST07_STAGE02_V32",

        "master":
            str(
                master_path
            ),

        "candidate_source":
            str(
                candidate_json
            ),

        "input_candidates":
            len(
                candidates
            ),

        "plausible_duplicate_pairs":
            len(
                pairs
            ),

        "final_inventory":
            len(
                final_objects
            ),

        "inventory":
            str(
                inventory_path
            ),

        "report":
            str(
                report_path
            ),
    }

    (
        output_dir
        / "stage02_v32_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2
        )
    )

    print()
    print("=" * 90)
    print("TEST07 STAGE02-V3.2 RESULT")
    print("=" * 90)

    print(
        "INPUT CANDIDATES:",
        len(
            candidates
        )
    )

    print(
        "PLAUSIBLE DUPLICATE PAIRS:",
        len(
            pairs
        )
    )

    print(
        "FINAL INVENTORY:",
        len(
            final_objects
        )
    )

    print()
    print(
        "INVENTORY:"
    )

    print(
        inventory_path
    )

    print()
    print(
        "REPORT:"
    )

    print(
        report_path
    )

    print()
    print(
        "OBJECT LIST:"
    )

    for row in final_objects:

        regions = row.get(
            "source_regions",
            [
                row.get(
                    "source_region",
                    ""
                )
            ]
        )

        print(
            f'{row["id"]:03d}. '
            f'{row.get("name", "")} | '
            f'{row.get("confidence", "")} | '
            f'{row.get("grounding_phrase", "")} | '
            f'sources={regions} | '
            f'merged={row.get("merged_candidate_count", 1)}'
        )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--master",
        required=True
    )

    parser.add_argument(
        "--candidate-json",
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

        candidate_json=
            args.candidate_json,

        output_dir=
            args.output_dir,

        model_cache=
            args.model_cache
    )
