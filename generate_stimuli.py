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

import matplotlib

matplotlib.use("Agg")
from multiprocessing import Pool
from pathlib import Path

import pyllusion

# ============================================================================
# GRID DEFINITION
# ============================================================================

ILLUSION_STRENGTHS = [
    -49.0,
    -42.0,
    -35.0,
    -28.0,
    -21.0,
    -14.0,
    -7.0,
    0.0,
    7.0,
    14.0,
    21.0,
    28.0,
    35.0,
    42.0,
    49.0,
]

# Signed physical difference (matches the stimulus metadata field "Difference"
# in RealityBending/IllusionGameValidation study2/stimuli/stimuli_part{1,2}.js)
DIFFERENCES = [
    -0.46,
    -0.3587,
    -0.27349,
    -0.20297,
    -0.14575,
    -0.10044,
    -0.06565,
    -0.04,
    0.04,
    0.06565,
    0.10044,
    0.14575,
    0.20297,
    0.27349,
    0.3587,
    0.46,
]


# ============================================================================
# HELPERS
# ============================================================================


def make_filename(strength: float, diff: float) -> str:
    """
    Encode illusion parameters into a filename for downstream parsing.

    Sign and magnitude are handled separately so zero-padding is always
    applied to the magnitude only, giving consistent 3-digit fields:
        strength -49 → '-049'   strength +7 → '+007'

    Diff is stored at 5 decimal places to preserve the full precision of
    the IllusionGameValidation parameter set.

    Examples:
        make_filename( 50,   0.2)    → 'ML_str+050_diff+0.20000.png'
        make_filename(-49,  -0.3587) → 'ML_str-049_diff-0.35870.png'
    """
    s_sign = "-" if strength < 0 else "+"
    d_sign = "-" if diff < 0 else "+"
    return f"ML_str{s_sign}{abs(int(strength)):03d}" f"_diff{d_sign}{abs(diff):.5f}.png"


def parse_filename(stem: str) -> tuple[float, float]:
    """
    Reverse of make_filename — extract (illusion_strength, true_diff) from a stem.

    Strips the known 'str' / 'diff' prefixes by position rather than using
    str.replace(), so the logic is explicit and unambiguous.

    Example:
        parse_filename('ML_str+050_diff+0.20000') → (50.0, 0.2)
        parse_filename('ML_str-049_diff-0.35870') → (-49.0, -0.3587)
    """
    _, str_part, diff_part = stem.split("_")
    strength = float(str_part[3:])  # strip leading 'str'
    diff = float(diff_part[4:])  # strip leading 'diff'
    return strength, diff


# ============================================================================
# WORKER (must be top-level for multiprocessing pickling)
# ============================================================================


def _generate_one(args: tuple) -> str:
    """
    Generate and save a single stimulus. Returns the filename (or 'skipped').

    Args:
        args: (output_dir, strength, diff, force)
    """
    output_dir, strength, diff, force = args
    filename = make_filename(strength, diff)
    out_path = Path(output_dir) / filename

    if out_path.exists() and not force:
        return f"[skip]  {filename}"

    illusion = pyllusion.MullerLyer(illusion_strength=strength, difference=diff)
    illusion.to_image().save(out_path)
    return f"[done]  {filename}"


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

    task_args = [
        (str(output_dir), strength, diff, force)
        for strength in ILLUSION_STRENGTHS
        for diff in DIFFERENCES
    ]

    with Pool() as pool:
        results = pool.map(_generate_one, task_args)

    for line in results:
        print(f"  {line}")

    generated = sum(1 for r in results if r.startswith("[done]"))
    skipped = sum(1 for r in results if r.startswith("[skip]"))

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
