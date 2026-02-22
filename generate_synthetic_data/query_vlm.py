#!/usr/bin/env python3
"""
query_vlm.py - Query VLMs on Müller-Lyer stimulus grid (Phase 2)

Sends each stimulus image to the OpenAI Vision API as a strict forced-choice
question ("Which line looks longer — Bottom or Top?") and logs results to JSONL.

Runs N_PARTICIPANTS independent passes over the full stimulus grid, each saved
to its own file so that proportion-correct is computed over genuine independent
samples rather than a single deterministic response.

Each output record contains:
    participant_id   : zero-padded integer (01–20)
    image_id         : stem of the image filename
    illusion_strength: parsed from filename
    true_diff        : parsed from filename (top − bottom physical length)
    response         : "Bottom" or "Top" (model's answer)
    correct          : 1/0/null — whether response matches physical truth
                       (null when true_diff == 0, i.e. lines are equal)

Output layout:
    results/synthetic_participants/participant_01.jsonl
    results/synthetic_participants/participant_02.jsonl
    ...
    results/synthetic_participants/participant_20.jsonl

Usage:
    python query_vlm.py
    python query_vlm.py --image-dir ./stimuli --output-dir ./results/synthetic_participants
    python query_vlm.py --n-participants 5 --temperature 1.0

Options:
    --image-dir PATH        Directory of .png stimuli   (default: ./stimuli)
    --output-dir PATH       Participant output directory (default: ./results/synthetic_participants)
    --errors-dir PATH       Error log directory         (default: ./results/errors)
    --n-participants N       Number of synthetic participants to run
    --max-images N          Max images per participant  (default: 0 = all)
    --max-concurrency N     Concurrent API requests
    --max-dimension N       Resize images above this    (default: 1024)
    --jpeg-quality N        JPEG quality 1–100          (default: 90)
    --model NAME            OpenAI model                (default: from parameters.py)
    --force-reprocess       Ignore existing results and reprocess everything
    --dry-run               Print plan without making any API calls

Requirements:
    pip install openai pillow aiofiles
"""

import argparse
import asyncio
import base64
import getpass
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiofiles
from model_parameters import (
    MAX_DIMENSIONS,
    MAX_TOKENS,
    N_PARTICIPANTS,
    OPENAI_MODEL,
    TEMPERATURE,
)
from openai import AsyncOpenAI
from PIL import Image
from response_schema import response_schema
from vlm_prompt import vlm_prompt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MODEL = OPENAI_MODEL
VLM_PROMPT = vlm_prompt
RESPONSE_SCHEMA = response_schema

# ============================================================================
# FILENAME PARSING
# ============================================================================


def parse_image_id(image_id: str) -> tuple[int, float]:
    """
    Extract illusion_strength and true_diff from a stimulus filename stem.

    Example:
        parse_image_id('MullerLyer_str+050_diff+0.20000') → (50, 0.20)
        parse_image_id('MullerLyer_str-025_diff-0.10000') → (-25, -0.10)
    """
    parts = image_id.split("_")
    strength = float(parts[1].replace("str", ""))
    diff = float(parts[2].replace("diff", ""))
    return strength, diff


def compute_correct(response: str, true_diff: float) -> Optional[int]:
    """
    Determine whether the model's response matches physical ground truth.

    Convention: true_diff = top_length − bottom_length.
      true_diff > 0 → top is longer    → correct answer is "Top"
      true_diff < 0 → bottom is longer → correct answer is "Bottom"
      true_diff = 0 → lines are equal  → no correct answer (returns None)
    """
    if true_diff > 0:
        return 1 if response == "Top" else 0
    elif true_diff < 0:
        return 1 if response == "Bottom" else 0
    else:
        return None


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================


def preprocess_image(
    image_path: Path, max_dimension: int = 1024, jpeg_quality: int = 90
) -> str:
    """
    Load, resize if needed, and encode image as base64 JPEG.

    Args:
        image_path:    Path to stimulus PNG.
        max_dimension: Max pixel dimension before resizing.
        jpeg_quality:  JPEG quality for encoding.

    Returns:
        Base64-encoded JPEG string.
    """
    img = Image.open(image_path)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    width, height = img.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        img = img.resize(
            (int(width * scale), int(height * scale)),
            Image.Resampling.LANCZOS,
        )

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ============================================================================
# IMAGE DISCOVERY
# ============================================================================


def discover_images(image_dir: Path) -> List[str]:
    """
    Discover all Müller-Lyer stimulus PNGs in image_dir.

    Returns:
        List of image ID stems (filenames without extension), sorted by
        illusion_strength then true_diff.
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    png_files = list(image_dir.glob("MullerLyer_str*_diff*.png"))

    if not png_files:
        raise FileNotFoundError(
            f"No Müller-Lyer stimuli (MullerLyer_str*_diff*.png) found in {image_dir}. "
            "Run generate_stimuli.py first."
        )

    image_ids = [f.stem for f in png_files]
    image_ids.sort(key=lambda s: parse_image_id(s))
    return image_ids


# ============================================================================
# RESULT MANAGEMENT
# ============================================================================


def load_existing_results(output_path: Path) -> Set[str]:
    """Load image IDs that have already been successfully processed."""
    processed_ids: Set[str] = set()
    if not output_path.exists():
        return processed_ids
    try:
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "image_id" in record:
                            processed_ids.add(record["image_id"])
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
    return processed_ids


def load_errored_images(errors_path: Path) -> Set[str]:
    """Load image IDs that previously errored (so they get retried)."""
    errored_ids: Set[str] = set()
    if not errors_path.exists():
        return errored_ids
    try:
        with open(errors_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "image_id" in record:
                            errored_ids.add(record["image_id"])
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Warning: Could not load error log: {e}")
    return errored_ids


# ============================================================================
# API CLIENT WITH RETRY LOGIC
# ============================================================================


class VLMQuerier:
    """Async OpenAI client with exponential backoff retry."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = TEMPERATURE,
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

    async def query_image(self, image_id: str, image_base64: str) -> Dict[str, Any]:
        """
        Query OpenAI API on a single stimulus with exponential backoff retry.

        Returns:
            Dict with keys: image_id, illusion_strength, true_diff, response, correct.

        Raises:
            Exception if all retries are exhausted.
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.client.responses.create(
                    model=self.model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": VLM_PROMPT},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                                },
                            ],
                        }
                    ],
                    text={"format": RESPONSE_SCHEMA},
                    max_output_tokens=MAX_TOKENS,
                    temperature=self.temperature,
                )

                result = json.loads(response.output_text)

                strength, true_diff = parse_image_id(image_id)
                correct = compute_correct(result["response"], true_diff)

                return {
                    "image_id": image_id,
                    "illusion_strength": strength,
                    "true_diff": true_diff,
                    "response": result["response"],
                    "correct": correct,
                }

            except Exception as e:
                error_msg = str(e)
                is_rate_limit = "rate_limit" in error_msg.lower() or "429" in error_msg

                if attempt < self.max_retries - 1:
                    delay = self.initial_retry_delay * (2**attempt)
                    if is_rate_limit:
                        delay *= 2

                    safe_error = error_msg.encode("ascii", "backslashreplace").decode(
                        "ascii"
                    )
                    print(
                        f"  Retry {attempt + 1}/{self.max_retries} for {image_id} "
                        f"(waiting {delay:.1f}s): {safe_error[:100]}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise Exception(f"All retries exhausted: {error_msg}")


# ============================================================================
# BATCH PROCESSOR
# ============================================================================


class BatchProcessor:
    """Processes a batch of stimulus images concurrently with a semaphore."""

    def __init__(
        self,
        querier: VLMQuerier,
        image_dir: Path,
        output_path: Path,
        errors_path: Path,
        participant_id: int,
        max_concurrency: int = 20,
        max_dimension: int = 1024,
        jpeg_quality: int = 90,
    ):
        self.querier = querier
        self.image_dir = image_dir
        self.output_path = output_path
        self.errors_path = errors_path
        self.participant_id = participant_id
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality

        self.processed = 0
        self.errors = 0
        self.results_lock = asyncio.Lock()
        self.errors_lock = asyncio.Lock()

    async def process_single_image(self, image_id: str) -> None:
        """Load, encode, query, and log result for one stimulus."""
        async with self.semaphore:
            try:
                image_path = self.image_dir / f"{image_id}.png"
                image_base64 = preprocess_image(
                    image_path,
                    max_dimension=self.max_dimension,
                    jpeg_quality=self.jpeg_quality,
                )

                result = await self.querier.query_image(image_id, image_base64)
                # Tag each record with its participant ID for traceability
                result["participant_id"] = self.participant_id

                async with self.results_lock:
                    async with aiofiles.open(self.output_path, "a") as f:
                        await f.write(json.dumps(result) + "\n")

                self.processed += 1
                strength = result["illusion_strength"]
                diff = result["true_diff"]
                resp = result["response"]
                correct = result["correct"]
                correct_str = "✓" if correct == 1 else ("✗" if correct == 0 else "–")
                print(
                    f"  [P{self.participant_id:02d} | {self.processed:03d}] "
                    f"str={strength:+.0f} diff={diff:+.2f} → {resp:<6} {correct_str}"
                )

            except Exception as e:
                error_record = {
                    "participant_id": self.participant_id,
                    "image_id": image_id,
                    "error": str(e),
                }

                async with self.errors_lock:
                    async with aiofiles.open(self.errors_path, "a") as f:
                        await f.write(json.dumps(error_record) + "\n")

                self.errors += 1
                print(
                    f"  ✗ ERROR  [P{self.participant_id:02d}] {image_id}: {str(e)[:100]}"
                )

    async def process_batch(self, image_ids: List[str]) -> None:
        """Process all images concurrently (bounded by semaphore)."""
        tasks = [self.process_single_image(img_id) for img_id in image_ids]
        await asyncio.gather(*tasks)


# ============================================================================
# PARTICIPANT RUN
# ============================================================================


async def run_participant(
    participant_id: int,
    image_ids: List[str],
    image_dir: Path,
    output_dir: Path,
    errors_dir: Path,
    querier: VLMQuerier,
    args,
    force_reprocess: bool,
) -> tuple[int, int]:
    """
    Run one synthetic participant over the full stimulus list.

    Args:
        participant_id: 1-based integer identifier for this participant.
        image_ids:      Full list of discovered stimulus IDs.
        output_dir:     Directory to write participant_XX.jsonl into.
        errors_dir:     Directory to write participant_XX_errors.jsonl into.
        querier:        Shared VLMQuerier instance.
        args:           Parsed CLI arguments.
        force_reprocess: Re-query images already present in the output file.

    Returns:
        (n_processed, n_errors) for this participant run.
    """
    output_path = output_dir / f"participant_{participant_id:02d}.jsonl"
    errors_path = errors_dir / f"participant_{participant_id:02d}_errors.jsonl"

    # Initialise files if absent
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        output_path.write_text("")
    if not errors_path.exists():
        errors_path.write_text("")

    # Skip already-done images unless --force-reprocess
    if force_reprocess:
        to_process = image_ids
    else:
        already_done = load_existing_results(output_path)
        to_process = [img for img in image_ids if img not in already_done]
        if already_done:
            print(
                f"  [P{participant_id:02d}] Skipping {len(already_done)} already-processed images"
            )

    # Apply per-run image limit
    if args.max_images > 0:
        to_process = to_process[: args.max_images]

    if not to_process:
        print(f"  [P{participant_id:02d}] ✓ Already complete — skipping.")
        return 0, 0

    print(f"\n{'─' * 60}")
    print(
        f"  Participant {participant_id:02d} / {args.n_participants}  ({len(to_process)} images)"
    )
    print(f"{'─' * 60}")

    processor = BatchProcessor(
        querier=querier,
        image_dir=image_dir,
        output_path=output_path,
        errors_path=errors_path,
        participant_id=participant_id,
        max_concurrency=args.max_concurrency,
        max_dimension=args.max_dimension,
        jpeg_quality=args.jpeg_quality,
    )

    await processor.process_batch(to_process)
    return processor.processed, processor.errors


# ============================================================================
# MAIN
# ============================================================================


async def main_async(args) -> None:
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    errors_dir = Path(args.errors_dir)

    # Discover stimuli
    try:
        all_images = discover_images(image_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("VLM FORCED-CHOICE QUERYING — SYNTHETIC PARTICIPANTS")
    print("=" * 60)
    print(f"  Image directory   : {image_dir}")
    print(f"  Total stimuli     : {len(all_images)}")
    print(f"  Participants      : {args.n_participants}")
    print(
        f"  Total API calls   : {len(all_images) * args.n_participants} (may be less if data already exists)"
    )
    print(f"  Model             : {args.model}")
    print(f"  Temperature       : {args.temperature}")
    print(f"  Max concurrency   : {args.max_concurrency}")
    print(f"  Output directory  : {output_dir}/")
    print(f"  Errors directory  : {errors_dir}/")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for p in range(1, args.n_participants + 1):
            print(f"  participant_{p:02d}.jsonl  ← {len(all_images)} images")
        return

    # Prompt for API key once, reuse across all participants
    print()
    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ")
    if not api_key or not api_key.strip():
        print("Error: No API key provided.")
        sys.exit(1)
    print("  ✓ API key received\n")

    querier = VLMQuerier(
        api_key=api_key.strip(),
        model=args.model,
        temperature=args.temperature,
    )

    total_processed = 0
    total_errors = 0

    for participant_id in range(1, args.n_participants + 1):
        n_ok, n_err = await run_participant(
            participant_id=participant_id,
            image_ids=all_images,
            image_dir=image_dir,
            output_dir=output_dir,
            errors_dir=errors_dir,
            querier=querier,
            args=args,
            force_reprocess=args.force_reprocess,
        )
        total_processed += n_ok
        total_errors += n_err

    # Final summary
    print("\n" + "=" * 60)
    print("ALL PARTICIPANTS COMPLETE")
    print("=" * 60)
    print(f"  Total responses   : {total_processed}")
    print(f"  Total errors      : {total_errors}")
    print(f"  Output directory  : {output_dir}/")
    if total_errors:
        print(f"  Error logs        : {errors_dir}/")
    print("=" * 60)
    print("\nNext step: run fit_psychometrics.py to aggregate and fit PSEs.")


def main():
    parser = argparse.ArgumentParser(
        description="Query VLM on Müller-Lyer stimuli — synthetic participant paradigm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="./stimuli",
        help="Directory of stimulus PNGs (default: ./stimuli)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/synthetic_participants",
        help="Directory for participant JSONL files (default: ./results/synthetic_participants)",
    )
    parser.add_argument(
        "--errors-dir",
        type=str,
        default="./results/errors",
        help="Directory for error logs (default: ./results/errors)",
    )
    parser.add_argument(
        "--n-participants",
        type=int,
        default=N_PARTICIPANTS,
        help=f"Number of synthetic participants to get up to (default: {N_PARTICIPANTS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"API sampling temperature (default: {TEMPERATURE})",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max images per participant (default: 0 = all)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=240,
        help="Concurrent API requests per participant",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=MAX_DIMENSIONS,
        help="Max image dimension before resizing",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for API upload (default: 90)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Reprocess all images, ignoring existing results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without making any API calls",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
