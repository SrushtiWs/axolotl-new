
from pathlib import Path
import argparse
import json
import re
import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

VERIFY_TOKENS = 120


PROMPT_TEMPLATE = """
You are verifying whether ONE candidate physical object is actually visible
in this full temporary clean-room image.

Important image state:

- The mirror was intentionally removed before this image was created.
- The shower glass partition was intentionally removed before this image was created.
- Wall and floor surfaces were intentionally simplified for computer vision.
- Wall, floor and ceiling are architectural background surfaces and are NOT objects.
- Temporary plain wall/floor regions are NOT objects.
- Reflections and transparency artifacts are NOT objects.

Candidate:
NAME: {name}
GROUNDING PHRASE: {phrase}
SOURCE REGION: {source_region}
ORIGINAL CONFIDENCE: {confidence}

Question:
Is this candidate a real discrete physical object or fixture whose actual
visible pixels are supported somewhere in the full image?

Return PRESENT only when there is direct visual evidence.

Return ABSENT when:
- the candidate is wall, floor, ceiling or another structural surface
- the candidate is a mirror, because the mirror has been intentionally removed
- the candidate is glass / glass partition / window-like transparency artifact
  created only from the removed shower glass
- the candidate appears to be a preprocessing artifact
- the candidate is inferred from context rather than visibly supported
- the candidate is only a speculative component of another object
- no actual visible physical pixels support it

Return UNCERTAIN only when genuine visual evidence exists but is too ambiguous
to confidently decide.

Be conservative. Precision is more important than recall at this verification step.

Return ONLY:

decision || confidence || short_reason

decision must be:
PRESENT
ABSENT
UNCERTAIN

confidence must be:
high
medium
low

No JSON.
No markdown.
No explanation.
""".strip()


def run_qwen_image(
    model,
    processor,
    image_path,
    prompt,
    max_tokens
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path)
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to(
        model.device
    )

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids
        in zip(inputs.input_ids, generated)
    ]

    raw = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    del inputs
    del generated
    del trimmed

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return raw.strip()


def parse_result(raw):

    text = raw.strip()

    text = re.sub(
        r"^\s*```(?:text)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text
    )

    lines = [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]

    for line in lines:

        parts = [
            x.strip()
            for x in line.split("||")
        ]

        if len(parts) < 3:
            continue

        decision = parts[0].upper()
        confidence = parts[1].lower()

        reason = " || ".join(
            parts[2:]
        ).strip()

        if decision not in {
            "PRESENT",
            "ABSENT",
            "UNCERTAIN"
        }:
            continue

        if confidence not in {
            "high",
            "medium",
            "low"
        }:
            confidence = "low"

        return {
            "decision":
                decision,

            "verification_confidence":
                confidence,

            "reason":
                reason,

            "raw":
                raw
        }

    return {
        "decision":
            "UNCERTAIN",

        "verification_confidence":
            "low",

        "reason":
            "could_not_parse_verifier_output",

        "raw":
            raw
    }


def main(
    master,
    candidate_json,
    output_dir,
    model_cache=None
):

    master = Path(master)
    candidate_json = Path(candidate_json)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not master.exists():
        raise FileNotFoundError(master)

    if not candidate_json.exists():
        raise FileNotFoundError(candidate_json)

    candidates = json.loads(
        candidate_json.read_text()
    )

    if not isinstance(candidates, list):
        raise RuntimeError(
            "candidate-json must contain a list"
        )

    if model_cache is None:
        model_cache = (
            output_dir /
            "model_cache"
        )

    model_cache = Path(
        model_cache
    )

    model_cache.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 100)
    print("TEST07 - STAGE02 V4.1")
    print("FULL-IMAGE EXISTENCE VERIFICATION")
    print("=" * 100)

    print("MASTER:", master)
    print("CANDIDATES:", len(candidates))
    print("MODEL:", MODEL_ID)

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    print()
    print("Loading Qwen 7B...")

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=str(model_cache)
        )
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=str(model_cache)
    )

    results = []

    for index, row in enumerate(
        candidates,
        start=1
    ):

        name = str(
            row.get("name", "")
        ).strip()

        phrase = str(
            row.get(
                "grounding_phrase",
                name
            )
        ).strip()

        source_region = str(
            row.get(
                "source_region",
                row.get(
                    "source_regions",
                    ""
                )
            )
        ).strip()

        confidence = str(
            row.get(
                "confidence",
                "low"
            )
        ).strip()

        prompt = PROMPT_TEMPLATE.format(
            name=name,
            phrase=phrase,
            source_region=source_region,
            confidence=confidence
        )

        print()
        print("-" * 100)
        print(
            f"{index:03d}/{len(candidates):03d} |",
            name,
            "|",
            phrase
        )

        raw = run_qwen_image(
            model,
            processor,
            master,
            prompt,
            VERIFY_TOKENS
        )

        parsed = parse_result(
            raw
        )

        result = dict(
            row
        )

        result.update(
            parsed
        )

        result[
            "verification_index"
        ] = index

        results.append(
            result
        )

        print(
            parsed["decision"],
            "|",
            parsed["verification_confidence"],
            "|",
            parsed["reason"]
        )

    present = [
        x
        for x in results
        if x["decision"] == "PRESENT"
    ]

    absent = [
        x
        for x in results
        if x["decision"] == "ABSENT"
    ]

    uncertain = [
        x
        for x in results
        if x["decision"] == "UNCERTAIN"
    ]

    all_path = (
        output_dir /
        "01_all_existence_decisions.json"
    )

    present_path = (
        output_dir /
        "02_present_candidates.json"
    )

    absent_path = (
        output_dir /
        "03_absent_candidates.json"
    )

    uncertain_path = (
        output_dir /
        "04_uncertain_candidates.json"
    )

    all_path.write_text(
        json.dumps(
            results,
            indent=2
        )
    )

    present_path.write_text(
        json.dumps(
            present,
            indent=2
        )
    )

    absent_path.write_text(
        json.dumps(
            absent,
            indent=2
        )
    )

    uncertain_path.write_text(
        json.dumps(
            uncertain,
            indent=2
        )
    )

    report = {
        "stage":
            "TEST07_STAGE02_V41_FULL_IMAGE_EXISTENCE_VERIFICATION",

        "master":
            str(master),

        "candidate_json":
            str(candidate_json),

        "input_candidates":
            len(candidates),

        "present":
            len(present),

        "absent":
            len(absent),

        "uncertain":
            len(uncertain),

        "outputs": {
            "all":
                str(all_path),

            "present":
                str(present_path),

            "absent":
                str(absent_path),

            "uncertain":
                str(uncertain_path)
        }
    }

    report_path = (
        output_dir /
        "00_stage02_v41_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2
        )
    )

    print()
    print("=" * 100)
    print("V4.1 SUMMARY")
    print("=" * 100)

    print("INPUT     :", len(candidates))
    print("PRESENT   :", len(present))
    print("ABSENT    :", len(absent))
    print("UNCERTAIN :", len(uncertain))

    print()
    print("PRESENT OBJECTS:")

    for r in present:
        print(
            "-",
            r.get("name"),
            "|",
            r.get("grounding_phrase"),
            "| source=",
            r.get(
                "source_region",
                r.get("source_regions", "")
            )
        )

    print()
    print("REPORT:", report_path)


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
        default=None
    )

    args = parser.parse_args()

    main(
        master=
            args.master,

        candidate_json=
            args.candidate_json,

        output_dir=
            args.output_dir,

        model_cache=
            args.model_cache
    )
