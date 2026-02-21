#!/usr/bin/env python3
"""
plot_results.py - Visualise psychometric curves and PSE drift (Phase 3)

Produces two figures:

  Figure 1 — Psychometric curves (one per illusion strength)
      x-axis: true physical difference (right − left line length)
      y-axis: proportion of "Right" responses
      Each curve is a fitted cumulative Gaussian. The dashed vertical line
      at x=0 marks physical equality; the PSE is where each curve crosses 0.5.

  Figure 2 — PSE vs. illusion strength
      x-axis: illusion strength (0–100)
      y-axis: PSE (with ±SE error bars)
      A non-zero slope here is the key result: it means illusion strength
      systematically biases the model's perceptual judgments.

Usage:
    python plot_results.py [--results-dir PATH] [--output-dir PATH] [--show]

Requirements:
    pip install matplotlib numpy pandas scipy
"""

import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf

# ============================================================================
# HELPERS
# ============================================================================


def cumulative_gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def strength_colormap(strengths: list) -> dict:
    """Map each illusion strength to a distinct colour from a sequential palette."""
    cmap = cm.get_cmap("plasma", len(strengths))
    return {s: cmap(i) for i, s in enumerate(strengths)}


# ============================================================================
# FIGURE 1: PSYCHOMETRIC CURVES
# ============================================================================


def plot_psychometric_curves(
    psych_data: pd.DataFrame,
    pse_summary: pd.DataFrame,
    output_path: Path,
    show: bool = False,
) -> None:
    """Overlay fitted psychometric curves for all illusion strength levels."""

    strengths = sorted(psych_data["illusion_strength"].unique())
    colors = strength_colormap(strengths)
    x_smooth = np.linspace(
        psych_data["true_diff"].min(), psych_data["true_diff"].max(), 300
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for strength in strengths:
        color = colors[strength]
        subset = psych_data[psych_data["illusion_strength"] == strength]
        pse_row = pse_summary[pse_summary["illusion_strength"] == strength]

        # Scatter: observed proportions
        ax.scatter(
            subset["true_diff"],
            subset["prop_right"],
            color=color,
            s=40,
            zorder=3,
            alpha=0.8,
        )

        # Fitted curve (only if fit succeeded)
        if not pse_row.empty and pse_row.iloc[0]["fit_success"]:
            pse = pse_row.iloc[0]["pse"]
            sigma = pse_row.iloc[0]["sigma"]
            y_fit = cumulative_gaussian(x_smooth, pse, sigma)
            ax.plot(
                x_smooth,
                y_fit,
                color=color,
                linewidth=2.0,
                label=f"Strength {strength}  (PSE={pse:+.3f})",
            )

    # Reference lines
    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        label="Physical equality",
    )
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("True physical difference  (right − left)", fontsize=12)
    ax.set_ylabel("P(respond 'Right')", fontsize=12)
    ax.set_title("Psychometric Functions — GPT-4o on Müller-Lyer Illusion", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Figure 1 saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================================
# FIGURE 2: PSE vs. ILLUSION STRENGTH
# ============================================================================


def plot_pse_vs_strength(
    pse_summary: pd.DataFrame,
    output_path: Path,
    show: bool = False,
) -> None:
    """Plot PSE (with ±SE bars) as a function of illusion strength."""

    valid = pse_summary[pse_summary["fit_success"]].copy()
    if valid.empty:
        print("⚠️  No successful fits — skipping PSE-vs-strength plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(
        valid["illusion_strength"],
        valid["pse"],
        yerr=valid["pse_se"],
        fmt="o-",
        color="#5c4dc9",
        markersize=7,
        linewidth=2,
        capsize=4,
        label="PSE ± SE",
    )

    # Fit a linear trend to summarise direction
    if len(valid) >= 2:
        z = np.polyfit(valid["illusion_strength"], valid["pse"], 1)
        x_line = np.linspace(
            valid["illusion_strength"].min(), valid["illusion_strength"].max(), 200
        )
        ax.plot(
            x_line,
            np.polyval(z, x_line),
            "--",
            color="#a0a0a0",
            linewidth=1.2,
            label=f"Linear trend  (slope={z[0]:+.4f})",
        )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        label="No bias (PSE = 0)",
    )

    ax.set_xlabel("Illusion strength", fontsize=12)
    ax.set_ylabel("PSE  (physical difference at 50% threshold)", fontsize=12)
    ax.set_title("PSE Shift vs. Illusion Strength — GPT-4o", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Figure 2 saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Plot psychometric curves and PSE vs. illusion strength",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory containing CSVs from fit_psychometrics.py (default: ./results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save figures (default: ./results)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive figures in addition to saving",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    psych_path = results_dir / "psychometric_data.csv"
    pse_path = results_dir / "pse_summary.csv"

    for p in [psych_path, pse_path]:
        if not p.exists():
            print(f"Error: Required file not found: {p}")
            print("Run fit_psychometrics.py first.")
            raise SystemExit(1)

    psych_data = pd.read_csv(psych_path)
    pse_summary = pd.read_csv(pse_path)
    pse_summary["fit_success"] = pse_summary["fit_success"].astype(bool)

    print(
        f"Loaded {len(psych_data)} data points across "
        f"{psych_data['illusion_strength'].nunique()} strength levels.\n"
    )

    # Generate figures
    plot_psychometric_curves(
        psych_data,
        pse_summary,
        output_dir / "fig1_psychometric_curves.png",
        show=args.show,
    )
    plot_pse_vs_strength(
        pse_summary,
        output_dir / "fig2_pse_vs_strength.png",
        show=args.show,
    )

    print("\nDone. Check the results/ directory for figures.")


if __name__ == "__main__":
    main()
