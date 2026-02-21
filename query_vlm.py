#!/usr/bin/env python3
"""
query_vlm.py - Query GPT-4o on Müller-Lyer stimulus grid (Phase 2)

Sends each stimulus image to the OpenAI Vision API as a strict forced-choice
question ("Which line looks longer — Bottom or Top?") and logs results to JSONL.

Each output record contains:
    image_id         : stem of the image filename
    illusion_strength: parsed from filename (0–100)
    true_diff        : parsed from filename (top − bottom physical length)
    response         : "Bottom" or "Top" (model's answer)
    correct          : 1/0/null — whether response matches physical truth
                       (null when true_diff == 0, i.e. lines are equal)

Usage:
    python query_vlm.py
    python query_vlm.py --image-dir ./stimuli --output ./results/raw_responses.jsonl

Options:
    --image-dir PATH        Directory of .png stimuli (default: ./stimuli)
    --output PATH           Output JSONL file        (default: ./results/raw_responses.jsonl)
    --errors PATH           Error log JSONL file     (default: ./results/errors.jsonl)
    --max-images N          Max images to process this run (default: 0 = all)
    --max-concurrency N     Concurrent API requests  (default: 5)
    --max-dimension N       Resize images above this (default: 1024)
    --jpeg-quality N        JPEG quality 1–100       (default: 90)
    --model NAME            OpenAI model             (default: gpt-4o)
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
from openai import AsyncOpenAI
from PIL import Image

from parameters import OPENAI_MODEL
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
        parse_image_id('MullerLyer_str050_diff+0.20') → (50, 0.20)
        parse_image_id('MullerLyer_str025_diff-0.10') → (25, -0.10)
    """
    parts = image_id.split("_")
    strength = int(parts[1].replace("str", ""))
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
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

    async def query_image(self, image_id: str, image_base64: str) -> Dict[str, Any]:
        """
        Query OpeAI API on a single stimulus with exponential backoff retry.

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
                    max_output_tokens=500,
                    reasoning={"effort": "none"},
                    temperature=1,
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
        max_concurrency: int = 5,
        max_dimension: int = 1024,
        jpeg_quality: int = 90,
    ):
        self.querier = querier
        self.image_dir = image_dir
        self.output_path = output_path
        self.errors_path = errors_path
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
                    f"  [{self.processed:02d}] str={strength:3d} diff={diff:+.2f} "
                    f"→ {resp:<5} {correct_str}"
                )

            except Exception as e:
                error_record = {"image_id": image_id, "error": str(e)}

                async with self.errors_lock:
                    async with aiofiles.open(self.errors_path, "a") as f:
                        await f.write(json.dumps(error_record) + "\n")

                self.errors += 1
                print(f"  ✗ ERROR  {image_id}: {str(e)[:100]}")

    async def process_batch(self, image_ids: List[str]) -> None:
        """Process all images concurrently (bounded by semaphore)."""
        tasks = [self.process_single_image(img_id) for img_id in image_ids]
        await asyncio.gather(*tasks)


# ============================================================================
# MAIN
# ============================================================================


async def main_async(args) -> None:
    image_dir = Path(args.image_dir)
    output_path = Path(args.output)
    errors_path = Path(args.errors)

    # Discover stimuli
    try:
        all_images = discover_images(image_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load existing results to skip already-processed images
    processed_images: Set[str] = set()
    errored_images: Set[str] = set()

    if not args.force_reprocess:
        print("Checking for existing results...")
        processed_images = load_existing_results(output_path)
        errored_images = load_errored_images(errors_path)
        if processed_images:
            print(f"  ✓ Found {len(processed_images)} already-processed images")
        if errored_images:
            print(f"  ⚠️  Found {len(errored_images)} previously errored (will retry)")

    # Filter out already-processed images; always retry errored ones
    if args.force_reprocess:
        unprocessed = all_images
        skipped_count = 0
    else:
        unprocessed = [img for img in all_images if img not in processed_images]
        skipped_count = len(all_images) - len(unprocessed)

    # Apply per-run limit
    if args.max_images > 0:
        image_ids = unprocessed[: args.max_images]
        remaining = len(unprocessed) - args.max_images
    else:
        image_ids = unprocessed
        remaining = 0

    print("\n" + "=" * 60)
    print("VLM FORCED-CHOICE QUERYING")
    print("=" * 60)
    print(f"  Image directory  : {image_dir}")
    print(f"  Total stimuli    : {len(all_images)}")
    if skipped_count:
        print(f"  Already done     : {skipped_count}")
    print(f"  To process now   : {len(image_ids)}")
    print(f"  Model            : {args.model}")
    print(f"  Max concurrency  : {args.max_concurrency}")
    print(f"  Temperature      : 0 (deterministic)")
    print(f"  Output           : {output_path}")
    print(f"  Errors           : {errors_path}")
    print("=" * 60)

    if len(image_ids) == 0:
        print("\n✓ All images already processed. Use --force-reprocess to redo.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for img_id in image_ids:
            print(f"  {img_id}")
        return

    # Prompt for API key securely
    print()
    api_key = getpass.getpass("Enter your OpenAI API key (input hidden): ")
    if not api_key or not api_key.strip():
        print("Error: No API key provided.")
        sys.exit(1)
    print("  ✓ API key received\n")

    # Ensure output dirs exist and files are initialised
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        output_path.write_text("")
    if not errors_path.exists():
        errors_path.write_text("")

    # Run
    querier = VLMQuerier(api_key=api_key.strip(), model=args.model)
    processor = BatchProcessor(
        querier=querier,
        image_dir=image_dir,
        output_path=output_path,
        errors_path=errors_path,
        max_concurrency=args.max_concurrency,
        max_dimension=args.max_dimension,
        jpeg_quality=args.jpeg_quality,
    )

    print(f"Processing {len(image_ids)} images...\n")
    await processor.process_batch(image_ids)

    # Summary
    total_done = len(processed_images) + processor.processed
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  This run   : {processor.processed} ok, {processor.errors} errors")
    print(f"  Total done : {total_done} / {len(all_images)}")
    if remaining > 0:
        print(f"\n  💡 {remaining} remaining — run again to continue.")
    print(f"\n  📄 Results : {output_path}")
    if processor.errors:
        print(f"  ⚠️  Errors  : {errors_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Query GPT-4o on Müller-Lyer stimuli",
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
        "--output",
        type=str,
        default="./results/raw_responses.jsonl",
        help="Output JSONL file (default: ./results/raw_responses.jsonl)",
    )
    parser.add_argument(
        "--errors",
        type=str,
        default="./results/errors.jsonl",
        help="Error log JSONL file (default: ./results/errors.jsonl)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max images to process this run (default: 0 = all)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Concurrent API requests (default: 5)",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1024,
        help="Max image dimension before resizing (default: 1024)",
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
