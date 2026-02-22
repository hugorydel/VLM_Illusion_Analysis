#!/usr/bin/env python3
"""
batch_vlm.py - Submit and retrieve OpenAI Batch API jobs (Phase 2, async variant)

The Batch API is 3.4× cheaper than real-time querying for GPT-5.2, at the cost of a
delayed output.  This module handles three distinct phases:

  submit    Scan the output directory for existing participant files, compute
            which participant IDs are missing up to --n-participants, build a
            batch JSONL request file, upload it, and submit the batch job.
            Saves a state file (batch_state.json) so the other commands know
            which batch to poll.

  status    Check the current status of the submitted batch.

  download  Once the batch completes, download the results and write one
            participant_XX.jsonl per participant into the output directory,
            in the same format produced by query_vlm.py.

Gap-filling behaviour
---------------------
Only participant IDs that are genuinely absent from the output directory are
queued, regardless of the numbering of existing files.  Example:

    Existing files : participant_076.jsonl … participant_100.jsonl  (25 files)
    --n-participants 100
    → Missing IDs  : 1 … 75  (submitted to the batch)
    → NOT          : 101 … 125  (that would exceed the target)

This means you can freely mix real-time (query_vlm.py) and batch runs without
creating duplicate participant files.

Batch request format
--------------------
Each line in the uploaded JSONL is a /v1/chat/completions request.  Note that
the Batch API does not support /v1/responses (the newer endpoint used by
query_vlm.py), so this module uses chat.completions instead — the model output
and structured JSON schema behaviour are identical.

Custom ID encoding
------------------
    custom_id = "p{participant_id:03d}|{image_id}"

The pipe (|) is used as a separator because image IDs already contain
underscores.  The download phase splits on the first | to recover both fields.

State file
----------
    results/batch_state.json   (location overridable with --state-file)

Stores: batch_id, input_file_id, submitted_at, participant_ids,
n_participants_target, output_dir.

Usage:
    python batch_vlm.py submit   --n-participants 100
    python batch_vlm.py status
    python batch_vlm.py download

Options (all subcommands):
    --output-dir PATH     Participant JSONL directory  (default: ./results/synthetic_participants)
    --state-file PATH     Batch state JSON             (default: ./results/batch_state.json)

Options (submit only):
    --image-dir PATH      Stimulus PNG directory       (default: ./stimuli)
    --n-participants N    Target total participants    (required)
    --model NAME          OpenAI model                 (default: from parameters.py)
    --temperature FLOAT   Sampling temperature         (default: 0.5)
    --max-dimension N     Resize images above this     (default: 1024)
    --jpeg-quality N      JPEG quality 1-100           (default: 90)
    --batch-dir PATH      Staging dir for batch JSONL  (default: ./results/batch_staging)
    --dry-run             Show plan without uploading
"""

import argparse
import getpass
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from model_parameters import (
    MAX_BATCH_BYTES,
    MAX_DIMENSIONS,
    MAX_TOKENS,
    OPENAI_MODEL,
    TEMPERATURE,
)
from openai import OpenAI
from PIL import Image
from query_vlm import compute_correct, discover_images, parse_image_id, preprocess_image
from response_schema import response_schema
from vlm_prompt import vlm_prompt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_MODEL = OPENAI_MODEL
STATE_FILENAME = "batch_state.json"

# Translate response_schema (responses-API format) to chat.completions format
_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": response_schema["name"],
        "strict": response_schema["strict"],
        "schema": response_schema["schema"],
    },
}

# ============================================================================
# GAP-FILLING: which (participant, image) pairs are missing?
# ============================================================================


def get_missing_requests(
    output_dir: Path, n_participants: int, all_images: list[str]
) -> dict[int, list[str]]:
    """
    Scan output_dir and return {participant_id: [missing_image_ids]}.

    Checks inside each file to find which images are already recorded,
    so adding new stimuli to the grid is handled automatically.
    Only participants with at least one missing image are included.
    """
    requests: dict[int, list[str]] = {}
    for pid in range(1, n_participants + 1):
        path = output_dir / f"participant_{pid:02d}.jsonl"
        done: set[str] = set()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            done.add(json.loads(line)["image_id"])
                        except Exception:
                            pass
        missing = [img for img in all_images if img not in done]
        if missing:
            requests[pid] = missing
    return requests


# ============================================================================
# BATCH JSONL PREPARATION
# ============================================================================


def make_custom_id(participant_id: int, image_id: str) -> str:
    """Encode participant + image into a single batch custom_id string."""
    return f"p{participant_id:03d}|{image_id}"


def parse_custom_id(custom_id: str) -> tuple[int, str]:
    """Decode a custom_id back into (participant_id, image_id)."""
    pid_str, image_id = custom_id.split("|", 1)
    return int(pid_str[1:]), image_id  # strip leading 'p'


def build_single_request(
    participant_id: int,
    image_id: str,
    image_base64: str,
    model: str,
    temperature: float,
) -> dict:
    """
    Build one /v1/chat/completions batch request dict for a single image.

    The Batch API does NOT support /v1/responses, so we use chat.completions
    format here.  Output is structurally identical to query_vlm.py.
    """
    return {
        "custom_id": make_custom_id(participant_id, image_id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": MAX_TOKENS,
            "response_format": _RESPONSE_FORMAT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vlm_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
        },
    }


def prepare_batch_files(
    requests_to_make: dict[int, list[str]],
    image_dir: Path,
    batch_dir: Path,
    model: str,
    temperature: float,
    max_dimension: int,
    jpeg_quality: int,
) -> list[tuple[Path, list[int]]]:
    """
    Build one or more batch input JSONLs, splitting at participant boundaries
    whenever the running size would exceed MAX_BATCH_BYTES.

    Accepts a per-participant dict of missing image IDs so that participants
    with partially-complete files only contribute their outstanding images.
    Returns a list of (path, participant_ids) tuples, one per sub-batch.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    total_requests = sum(len(imgs) for imgs in requests_to_make.values())

    print(
        f"\nPreparing batch JSONL(s): {total_requests} total requests across "
        f"{len(requests_to_make)} participants"
    )
    print(f"  Size limit per file : {MAX_BATCH_BYTES // 1024 // 1024} MB")

    # Encode only the unique images actually needed
    needed_images: set[str] = set()
    for imgs in requests_to_make.values():
        needed_images.update(imgs)

    print("  Pre-encoding images...")
    encoded: dict[str, str] = {}
    for image_id in needed_images:
        encoded[image_id] = preprocess_image(
            image_dir / f"{image_id}.png",
            max_dimension=max_dimension,
            jpeg_quality=jpeg_quality,
        )
    print(f"  {len(encoded)} unique images encoded")

    sub_batches: list[tuple[Path, list[int]]] = []
    file_index = 1
    current_ids: list[int] = []
    current_bytes = 0
    current_path = None
    current_fh = None

    def _open_new():
        nonlocal file_index, current_path, current_fh
        current_path = batch_dir / f"batch_input_{timestamp}_{file_index:02d}.jsonl"
        file_index += 1
        print(f"  Opening: {current_path.name}")
        current_fh = open(current_path, "w", encoding="utf-8")

    _open_new()

    for pid, img_list in requests_to_make.items():
        pid_lines = [
            json.dumps(
                build_single_request(pid, img_id, encoded[img_id], model, temperature)
            )
            + "\n"
            for img_id in img_list
        ]
        pid_bytes = sum(len(l.encode()) for l in pid_lines)

        # Roll over if this participant would push us over the limit
        if current_ids and (current_bytes + pid_bytes) > MAX_BATCH_BYTES:
            current_fh.close()
            print(
                f"    -> Closed with {len(current_ids)} participants, "
                f"{current_bytes / 1024 / 1024:.1f} MB"
            )
            sub_batches.append((current_path, list(current_ids)))
            current_ids = []
            current_bytes = 0
            _open_new()

        for line in pid_lines:
            current_fh.write(line)
        current_ids.append(pid)
        current_bytes += pid_bytes

    current_fh.close()
    print(
        f"    -> Closed with {len(current_ids)} participants, "
        f"{current_bytes / 1024 / 1024:.1f} MB"
    )
    sub_batches.append((current_path, list(current_ids)))

    print(f"  {len(sub_batches)} sub-batch file(s) ready")
    return sub_batches


# ============================================================================
# STATE FILE
# ============================================================================


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"  ✓ State saved to {state_path}")


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        print(f"Error: State file not found: {state_path}")
        print("Run 'python batch_vlm.py submit' first.")
        sys.exit(1)
    with open(state_path, "r") as f:
        return json.load(f)


# ============================================================================
# SUBCOMMAND: submit
# ============================================================================


def cmd_submit(args) -> None:
    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)
    batch_dir = Path(args.batch_dir)
    state_path = Path(args.state_file)

    # -- Discover stimuli -----------------------------------------------------
    try:
        all_images = discover_images(image_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # -- Compute missing (participant, image) pairs ---------------------------
    requests_to_make = get_missing_requests(output_dir, args.n_participants, all_images)
    total_requests = sum(len(imgs) for imgs in requests_to_make.values())

    print("\n" + "=" * 60)
    print("BATCH SUBMISSION PLAN")
    print("=" * 60)
    print(f"  Target participants         : {args.n_participants}")
    print(f"  Participants needing updates: {len(requests_to_make)}")
    print(f"  Total missing requests      : {total_requests}")
    print(f"  Images available            : {len(all_images)}")
    print(f"  Model                       : {args.model}")
    print(f"  Temperature                 : {args.temperature}")
    print("=" * 60)

    if not requests_to_make:
        print("\n[OK] All participants already complete -- nothing to submit.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would submit the above batch. Exiting.")
        return

    # -- Build batch JSONL(s) — auto-split if > 190 MB -----------------------
    sub_batch_files = prepare_batch_files(
        requests_to_make=requests_to_make,
        image_dir=image_dir,
        batch_dir=batch_dir,
        model=args.model,
        temperature=args.temperature,
        max_dimension=args.max_dimension,
        jpeg_quality=args.jpeg_quality,
    )

    # -- API key --------------------------------------------------------------
    print()
    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ")
    if not api_key.strip():
        print("Error: No API key provided.")
        sys.exit(1)
    client = OpenAI(api_key=api_key.strip())

    # -- Upload and submit each sub-batch -------------------------------------
    submitted_batches = []
    for i, (batch_path, pids) in enumerate(sub_batch_files, 1):
        print(
            f"\nSub-batch {i}/{len(sub_batch_files)}: "
            f"{len(pids)} participants (IDs {pids[0]}-{pids[-1]})"
        )

        print(f"  Uploading {batch_path.name}...")
        with open(batch_path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        print(f"  File uploaded: {uploaded.id}")

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": (
                    f"VLM illusion study -- participants {pids[0]}-{pids[-1]} "
                    f"(sub-batch {i}/{len(sub_batch_files)})"
                )
            },
        )
        print(f"  Batch submitted: {batch.id}  status={batch.status}")
        submitted_batches.append(
            {
                "batch_id": batch.id,
                "input_file_id": uploaded.id,
                "participant_ids": pids,
                "status": batch.status,
                "output_file_id": None,
            }
        )

    # -- Save state -----------------------------------------------------------
    state = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "n_participants_target": args.n_participants,
        "output_dir": str(output_dir),
        "model": args.model,
        "batches": submitted_batches,
    }
    save_state(state_path, state)

    print(f"\nNext steps:")
    print(f"  Check status : python batch_vlm.py status")
    print(
        f"  Download     : python batch_vlm.py download  (once all batches = completed)"
    )


# ============================================================================
# SUBCOMMAND: status
# ============================================================================


def cmd_status(args) -> None:
    state_path = Path(args.state_file)
    state = load_state(state_path)

    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ")
    if not api_key.strip():
        print("Error: No API key provided.")
        sys.exit(1)
    client = OpenAI(api_key=api_key.strip())

    batches = state.get("batches", [])
    print("\n" + "=" * 60)
    print(f"BATCH STATUS  ({len(batches)} sub-batch(es))")
    print("=" * 60)
    print(f"  Submitted at : {state['submitted_at']}")
    print(f"  Model        : {state.get('model', 'unknown')}")

    all_complete = True
    for i, b in enumerate(batches, 1):
        batch = client.batches.retrieve(b["batch_id"])
        b["status"] = batch.status
        if batch.output_file_id:
            b["output_file_id"] = batch.output_file_id
        if batch.status != "completed":
            all_complete = False

        print(f"\n  Sub-batch {i}/{len(batches)}:")
        print(f"    Batch ID     : {batch.id}")
        print(f"    Status       : {batch.status}")
        print(f"    Participants : {b['participant_ids']}")
        if batch.request_counts:
            rc = batch.request_counts
            print(
                f"    Requests     : {rc.completed}/{rc.total} complete, {rc.failed} failed"
            )
        if batch.output_file_id:
            print(f"    Output file  : {batch.output_file_id}")
        if batch.status == "failed":
            print(f"    [FAILED] Check the OpenAI dashboard for details.")

    print("\n" + "=" * 60)
    n_done = sum(1 for b in batches if b["status"] == "completed")
    if all_complete:
        print("\n[OK] All sub-batches complete -- run 'python batch_vlm.py download'.")
    else:
        print(f"\n  {n_done}/{len(batches)} sub-batches complete. Check again later.")

    save_state(state_path, state)


# ============================================================================
# SUBCOMMAND: download
# ============================================================================


def parse_batch_response(line: str) -> Optional[dict]:
    """
    Parse one line of the batch output JSONL and return a participant record,
    or None if the request failed.

    Each output line has the shape:
        {
            "custom_id": "p001|MullerLyer_str+049_diff+0.46000",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": "{\"image_id\":...,\"response\":...}"}}]
                }
            },
            "error": null
        }
    """
    obj = json.loads(line)

    if obj.get("error") is not None:
        return None

    response_body = obj.get("response", {}).get("body", {})
    if response_body.get("error"):
        return None

    choices = response_body.get("choices", [])
    if not choices:
        return None

    content = choices[0]["message"]["content"]
    parsed = json.loads(content)

    participant_id, image_id = parse_custom_id(obj["custom_id"])
    strength, true_diff = parse_image_id(image_id)
    correct = compute_correct(parsed["response"], true_diff)

    return {
        "participant_id": participant_id,
        "image_id": image_id,
        "illusion_strength": strength,
        "true_diff": true_diff,
        "response": parsed["response"],
        "correct": correct,
    }


def cmd_download(args) -> None:
    state_path = Path(args.state_file)
    state = load_state(state_path)
    output_dir = Path(args.output_dir or state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = state.get("batches", [])
    if not batches:
        print("Error: No batches found in state file. Re-run 'submit' first.")
        sys.exit(1)

    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ")
    if not api_key.strip():
        print("Error: No API key provided.")
        sys.exit(1)
    client = OpenAI(api_key=api_key.strip())

    records_by_participant: dict[int, list[dict]] = {}
    total_ok = 0
    total_fail = 0

    for i, b in enumerate(batches, 1):
        print(f"\nSub-batch {i}/{len(batches)}: {b['batch_id']}")
        batch = client.batches.retrieve(b["batch_id"])

        if batch.status != "completed":
            print(f"  [SKIP] Status is '{batch.status}' -- not yet complete.")
            continue

        print(f"  Downloading (file: {batch.output_file_id})...")
        raw = client.files.content(batch.output_file_id).text
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        print(f"  {len(lines)} response lines received")

        n_ok = n_fail = 0
        for line in lines:
            record = parse_batch_response(line)
            if record is None:
                n_fail += 1
                continue
            records_by_participant.setdefault(record["participant_id"], []).append(
                record
            )
            n_ok += 1
        print(f"  Parsed: {n_ok} successful, {n_fail} failed")
        total_ok += n_ok
        total_fail += n_fail

    # -- Write participant files (merge with existing to avoid overwriting) ---
    print(f"\nWriting participant files to {output_dir}/")
    for pid in sorted(records_by_participant):
        out_path = output_dir / f"participant_{pid:02d}.jsonl"

        # Load existing records keyed by image_id to deduplicate
        existing: dict[str, dict] = {}
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            existing[rec["image_id"]] = rec
                        except Exception:
                            pass

        # Merge: new records overwrite old ones for the same image_id
        for rec in records_by_participant[pid]:
            existing[rec["image_id"]] = rec

        all_records = sorted(
            existing.values(),
            key=lambda r: (r["illusion_strength"], r["true_diff"]),
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec) + "\n")
        n_new = len(records_by_participant[pid])
        print(
            f"  participant_{pid:02d}.jsonl  ({n_new} new -> {len(all_records)} total)"
        )

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Participants written : {len(records_by_participant)}")
    print(f"  Total successful     : {total_ok}")
    print(f"  Total failed         : {total_fail}")
    print(f"  Output directory     : {output_dir}/")
    print("=" * 60)
    print("\nNext step: run fit_psychometrics.py to aggregate and fit PSEs.")

    state["status"] = "downloaded"
    save_state(state_path, state)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI Batch API wrapper for VLM illusion study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── Shared options ─────────────────────────────────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--output-dir",
        type=str,
        default="./results/synthetic_participants",
        help="Participant JSONL directory (default: ./results/synthetic_participants)",
    )
    shared.add_argument(
        "--state-file",
        type=str,
        default="./results/batch_state.json",
        help="Batch state file (default: ./results/batch_state.json)",
    )

    # ── submit ─────────────────────────────────────────────────────────────
    p_submit = subparsers.add_parser(
        "submit",
        parents=[shared],
        help="Prepare and submit a batch job for missing participants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_submit.add_argument(
        "--n-participants",
        type=int,
        required=True,
        help="Target total number of participants (fills gaps up to this number)",
    )
    p_submit.add_argument(
        "--image-dir",
        type=str,
        default="./stimuli",
        help="Directory of stimulus PNGs (default: ./stimuli)",
    )
    p_submit.add_argument(
        "--batch-dir",
        type=str,
        default="./results/batch_staging",
        help="Staging directory for batch JSONL files (default: ./results/batch_staging)",
    )
    p_submit.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL})",
    )
    p_submit.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})",
    )
    p_submit.add_argument(
        "--max-dimension",
        type=int,
        default=MAX_DIMENSIONS,
        help="Max image dimension before resizing",
    )
    p_submit.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for API upload (default: 90)",
    )
    p_submit.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without uploading or submitting",
    )

    # ── status ─────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "status",
        parents=[shared],
        help="Check the status of the submitted batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── download ───────────────────────────────────────────────────────────
    subparsers.add_parser(
        "download",
        parents=[shared],
        help="Download completed batch results and write participant files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "download":
        cmd_download(args)


if __name__ == "__main__":
    main()
