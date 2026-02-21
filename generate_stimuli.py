#!/usr/bin/env python3
"""
generate_stimuli.py - Generate Müller-Lyer illusion stimulus grid

Generates a 2D grid of PNG images varying across:
  - illusion_strength: controls the angle/size of the arrow fins (0 = no illusion)
  - difference:        physical length difference between the two lines (right - left)

Grid defaults:
  - illusion_strength: [0, 25, 50, 75, 100]
  - difference:        [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
  → 35 images total

Filenames encode parameters for downstream parsing:
    ML_str050_diff+0.20.png
    ML_str025_diff-0.10.png

Usage:
    python generate_stimuli.py [--output-dir PATH]

Requirements:
    pip install pyllusion pillow
"""

import argparse
from pathlib import Path

import pyllusion

# ============================================================================
# GRID DEFINITION
# ============================================================================

ILLUSION_STRENGTHS = [0, 25, 50, 75, 100]

# Physical difference between right and left lines (right - left).
# Positive → right line is physically longer.
# Negative → left line is physically longer.
DIFFERENCES = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]


# ============================================================================
# HELPERS
# ============================================================================


def make_filename(strength: int, diff: float) -> str:
    """
    Encode illusion parameters into a filename for downstream parsing.

    Examples:
        make_filename(50,  0.2)  → 'ML_str050_diff+0.20.png'
        make_filename(25, -0.1)  → 'ML_str025_diff-0.10.png'
    """
    sign = "+" if diff >= 0 else ""
    return f"ML_str{strength:03d}_diff{sign}{diff:.2f}.png"


def parse_filename(stem: str) -> tuple[int, float]:
    """
    Reverse of make_filename — extract (illusion_strength, true_diff) from a stem.

    Example:
        parse_filename('ML_str050_diff+0.20') → (50, 0.20)
    """
    parts = stem.split("_")
    strength = int(parts[1].replace("str", ""))
    diff = float(parts[2].replace("diff", ""))
    return strength, diff


# ============================================================================
# GENERATION
# ============================================================================


def generate_grid(output_dir: Path, force: bool = False) -> None:
    """
    Generate the full stimulus grid and save images to output_dir.

    Args:
        output_dir: Directory to save images into.
        force:      If True, regenerate images that already exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(ILLUSION_STRENGTHS) * len(DIFFERENCES)

    print("=" * 60)
    print("MÜLLER-LYER STIMULUS GRID GENERATION")
    print("=" * 60)
    print(f"Illusion strengths : {ILLUSION_STRENGTHS}")
    print(f"Differences        : {DIFFERENCES}")
    print(f"Total images       : {total}")
    print(f"Output directory   : {output_dir}/")
    print("=" * 60 + "\n")

    generated = 0
    skipped = 0

    for strength in ILLUSION_STRENGTHS:
        for diff in DIFFERENCES:
            filename = make_filename(strength, diff)
            out_path = output_dir / filename

            if out_path.exists() and not force:
                print(f"  [skip]  {filename}")
                skipped += 1
                continue

            illusion = pyllusion.MullerLyer(
                illusion_strength=strength,
                difference=diff,
            )
            img = illusion.to_image()
            img.save(out_path)
            generated += 1
            print(f"  [{generated:02d}]    {filename}")

    print(f"\n✓ Generated : {generated} images")
    if skipped:
        print(f"  Skipped   : {skipped} (already exist; use --force to regenerate)")
    print(f"  Saved to  : {output_dir}/")
    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate Müller-Lyer stimulus grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./stimuli",
        help="Output directory for stimulus images (default: ./stimuli)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate images even if they already exist",
    )
    args = parser.parse_args()
    generate_grid(Path(args.output_dir), force=args.force)


if __name__ == "__main__":
    main()