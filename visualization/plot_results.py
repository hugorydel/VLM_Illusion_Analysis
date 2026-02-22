#!/usr/bin/env python3
"""
plot_results.py - Visualise psychometric curves and PSE drift

Produces three figures:

  Figure 1 — Makowski core: Error rate vs. illusion strength × difficulty
      x-axis: illusion strength (negative = congruent/facilitating)
      y-axis: probability of error
      curves: 6 task-difficulty bins (|true_diff|), hard → easy
      Directly comparable to Makowski et al. (2023) Figure 3.
      The key signature of illusion susceptibility is a monotonic interaction:
      stronger illusion → more errors, amplified on harder trials (low |Δ|).

  Figure 2 — PSE shift vs. illusion strength
      x-axis: illusion strength
      y-axis: PSE (with ±SE error bars)
      Points where the fit failed or the PSE sits at the edge of the tested
      Δ range are flagged explicitly rather than silently dropped.

  Figure 3 — Raw psychometric curves
      x-axis: true physical difference
      y-axis: P(respond 'Top') with fitted cumulative Gaussians.
      Diagnostic figure for verifying PSE fits.

Colour convention (consistent across all figures):
  Red   = hard / incongruent (high positive illusion strength)
  Green = easy / congruent   (high negative / facilitating illusion strength)
  Intermediate orange/yellow = mid difficulty / mid strength

Usage:
    python plot_results.py [--results-dir PATH] [--output-dir PATH] [--show]
                           [--participants-dir PATH] [--model-name NAME]

Requirements:
    pip install matplotlib numpy pandas scipy
"""

import argparse
import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf

# from model_parameters import OPENAI_MODEL

# ============================================================================
# CONSTANTS
# ============================================================================

N_DIFFICULTY_BINS = 6  # number of |diff| bins for Figure 1
MODEL_NAME = "gpt-5.2"

# Verbal labels for difficulty bins, ordered hard → easy (matches bin index 0 → N-1)
DIFFICULTY_LABELS = [
    "Very Hard",
    "Hard",
    "Medium-Hard",
    "Medium-Easy",
    "Easy",
    "Very Easy",
]


# ============================================================================
# HELPERS
# ============================================================================


def cumulative_gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def difficulty_colormap(n: int) -> list:
    """
    Return n colours from Red→Orange→Teal→Green (hard→easy).

    Samples only the saturated outer portions of RdYlGn (roughly 0.0–0.42
    for the red-orange side and 0.58–1.0 for the teal-green side), skipping
    the pale yellow centre that becomes invisible on a white background.

    Index 0 = red (hardest), index n-1 = green (easiest).
    Used by both Figure 1 (difficulty bins) and Figure 3 (illusion strength).
    """
    cmap = cm.get_cmap("RdYlGn")
    half = n // 2
    remainder = n - half
    positions = np.concatenate(
        [
            np.linspace(0.02, 0.42, half),  # red → orange
            np.linspace(0.58, 0.98, remainder),  # teal → green
        ]
    )
    return [cmap(p) for p in positions]


# ============================================================================
# DATA LOADING
# ============================================================================


def load_all_participants(participants_dir: Path) -> pd.DataFrame:
    """
    Load all participant_*.jsonl files into a single DataFrame.

    Trials with no ground truth (true_diff == 0) are retained — callers
    that need ground truth should filter on correct.notna() themselves.
    """
    jsonl_files = sorted(participants_dir.glob("participant_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No participant_*.jsonl files found in {participants_dir}"
        )

    records = []
    for path in jsonl_files:
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    return pd.DataFrame(records)


# ============================================================================
# FIGURE 1:  ERROR RATE × DIFFICULTY
# ============================================================================


def build_difficulty_bins(
    df: pd.DataFrame, n_bins: int = N_DIFFICULTY_BINS
) -> pd.DataFrame:
    """
    Assign each trial to one of n_bins difficulty bins based on |true_diff|.

    Bins are defined by quantiles of the unique |true_diff| values so that
    each bin spans a roughly equal range of difficulty levels.

    Returns df with added columns:
        abs_diff      — |true_diff|
        diff_bin      — verbal label for the bin ("Very hard" … "Very easy")
        diff_bin_mid  — numeric midpoint of the bin (used for sorting / colouring)
        diff_bin_idx  — integer index 0 (hardest) … n_bins-1 (easiest)
    """
    df = df.copy()
    df = df[df["correct"].notna()].copy()
    df["correct"] = df["correct"].astype(int)
    df["abs_diff"] = df["true_diff"].abs()

    unique_abs = np.sort(df["abs_diff"].unique())
    bin_edges = np.quantile(unique_abs, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] -= 1e-9  # ensure the smallest value is captured

    bin_labels, bin_mids = [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        label = DIFFICULTY_LABELS[i] if i < len(DIFFICULTY_LABELS) else f"Bin {i + 1}"
        bin_labels.append(label)
        bin_mids.append((lo + hi) / 2)

    label_map, mid_map, idx_map = {}, {}, {}
    for val in unique_abs:
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if lo < val <= hi:
                label_map[val] = bin_labels[i]
                mid_map[val] = bin_mids[i]
                idx_map[val] = i
                break

    df["diff_bin"] = df["abs_diff"].map(label_map)
    df["diff_bin_mid"] = df["abs_diff"].map(mid_map)
    df["diff_bin_idx"] = df["abs_diff"].map(idx_map)
    return df


def plot_error_by_difficulty(
    participants_dir: Path,
    output_path: Path,
    model_name: str = "Model",
    show: bool = False,
) -> pd.DataFrame:
    """
    Figure 1: error rate vs. illusion strength, one curve per difficulty bin.

    Colours: red (Very hard) → orange → yellow → green (Very easy).
    Returns the aggregated error DataFrame (reused by Figure 3).
    """
    raw_df = load_all_participants(participants_dir)
    df = build_difficulty_bins(raw_df, n_bins=N_DIFFICULTY_BINS)
    df["error"] = 1 - df["correct"]

    # Aggregate mean error per (illusion_strength, diff_bin)
    agg = (
        df.groupby(["illusion_strength", "diff_bin", "diff_bin_mid", "diff_bin_idx"])
        .agg(prop_error=("error", "mean"), n=("error", "count"))
        .reset_index()
        .sort_values(["diff_bin_mid", "illusion_strength"])
    )

    # Order bins hard → easy; assign colours from RdYlGn
    bins_ordered = (
        agg.drop_duplicates("diff_bin").sort_values("diff_bin_idx")["diff_bin"].tolist()
    )
    colors = difficulty_colormap(len(bins_ordered))
    bin_colors = {label: colors[i] for i, label in enumerate(bins_ordered)}

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for bin_label in bins_ordered:
        color = bin_colors[bin_label]
        subset = agg[agg["diff_bin"] == bin_label].sort_values("illusion_strength")
        ax.plot(
            subset["illusion_strength"],
            subset["prop_error"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            alpha=0.9,
            label=bin_label,
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.9,
        alpha=0.6,
        label="No illusion (strength = 0)",
    )
    ax.axhline(
        0.5,
        color="grey",
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
        label="Chance (50% error)",
    )

    ax.set_xlabel("Illusion strength", fontsize=12)
    ax.set_ylabel("Probability of error", fontsize=12)
    ax.set_title(
        f"Error Rate vs. Illusion Strength — {model_name}\n",
        fontsize=13,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(
        fontsize=9,
        loc="upper left",
        ncol=2,
        title="Task difficulty  (red = hard, green = easy)",
        title_fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Figure 1  saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)

    return agg  # passed to Figure 3


# ============================================================================
# FIGURE 2: PSE vs. ILLUSION STRENGTH (with range limits)
# ============================================================================


def plot_pse_vs_strength(
    pse_summary: pd.DataFrame,
    tested_diff_range: tuple[float, float],
    output_path: Path,
    model_name: str = "Model",
    show: bool = False,
) -> None:
    """
    Figure 2: PSE (with ±SE bars) vs. illusion strength.

    Points are classified into three categories:
      • Good fit  — PSE well within the tested Δ range  → solid marker
      • Edge fit  — PSE within range but near boundary (within 10%)
                    → hollow marker with warning in legend
      • Failed    — curve_fit did not converge          → dotted vertical line
    """
    df = pse_summary.copy()
    df["fit_success"] = df["fit_success"].astype(bool)

    diff_min, diff_max = tested_diff_range
    margin = 0.10 * (diff_max - diff_min)  # 10% boundary zone

    def classify(row):
        if not row["fit_success"] or np.isnan(row["pse"]):
            return "failed"
        if row["pse"] <= diff_min + margin or row["pse"] >= diff_max - margin:
            return "edge"
        return "good"

    df["fit_class"] = df.apply(classify, axis=1)

    good = df[df["fit_class"] == "good"]
    edge = df[df["fit_class"] == "edge"]
    failed = df[df["fit_class"] == "failed"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if not good.empty:
        ax.errorbar(
            good["illusion_strength"],
            good["pse"],
            yerr=good["pse_se"],
            fmt="o-",
            color="#5c4dc9",
            markersize=7,
            linewidth=2,
            capsize=4,
            label="PSE ± SE (reliable fit)",
        )

    if not edge.empty:
        ax.errorbar(
            edge["illusion_strength"],
            edge["pse"],
            yerr=edge["pse_se"],
            fmt="o",
            color="#e08c00",
            markersize=8,
            linewidth=0,
            capsize=4,
            markerfacecolor="none",
            markeredgewidth=2,
            label="PSE near range boundary (interpret with caution)",
        )

    if not failed.empty:
        for _, row in failed.iterrows():
            ax.axvline(
                row["illusion_strength"],
                color="#cc3333",
                linewidth=1.2,
                linestyle=":",
                alpha=0.6,
            )
        ax.plot(
            [],
            [],
            linestyle=":",
            color="#cc3333",
            linewidth=1.5,
            label="Fit failed — PSE outside tested Δ range",
        )

    # Linear trend through reliable fits only
    if len(good) >= 3:
        z = np.polyfit(good["illusion_strength"], good["pse"], 1)
        x_line = np.linspace(
            df["illusion_strength"].min(), df["illusion_strength"].max(), 200
        )
        ax.plot(
            x_line,
            np.polyval(z, x_line),
            "--",
            color="#a0a0a0",
            linewidth=1.2,
            label=f"Linear trend  (slope={z[0]:+.4f})",
        )

    # Shade the boundary zone and mark the tested range
    ax.axhspan(diff_min, diff_min + margin, alpha=0.07, color="orange")
    ax.axhspan(diff_max - margin, diff_max, alpha=0.07, color="orange")
    ax.axhline(
        diff_min,
        color="orange",
        linewidth=0.7,
        linestyle="--",
        alpha=0.5,
        label=f"Tested Δ boundary (±{diff_max:.2f})",
    )
    ax.axhline(
        diff_max,
        color="orange",
        linewidth=0.7,
        linestyle="--",
        alpha=0.5,
        label="_nolegend_",
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
    ax.set_title(f"PSE Shift vs. Illusion Strength — {model_name}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Figure 2  saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================================
# FIGURE 3: RAW PSYCHOMETRIC CURVES
# ============================================================================


def plot_psychometric_curves(
    psych_data: pd.DataFrame,
    pse_summary: pd.DataFrame,
    output_path: Path,
    model_name: str,
    show: bool = False,
) -> None:
    """
    Figure 3: fitted psychometric curves, one per illusion strength (diagnostic).

    Colours follow the same Red→Yellow→Green convention as Figure 1:
      most incongruent (positive) strength = red
      most congruent   (negative) strength = green
    """
    strengths = sorted(
        psych_data["illusion_strength"].unique()
    )  # ascending: negative → positive
    # Reverse so that most incongruent (last after sort = highest positive) gets red (index 0)
    colors_list = difficulty_colormap(len(strengths))
    # sort gives negative→positive; RdYlGn index 0=red, n-1=green
    # We want negative (congruent/easy) = green = high index, positive (incongruent/hard) = red = low index
    # So reverse the color list
    colors_list = list(reversed(colors_list))
    colors = {s: colors_list[i] for i, s in enumerate(strengths)}

    x_smooth = np.linspace(
        psych_data["true_diff"].min(), psych_data["true_diff"].max(), 300
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for strength in strengths:
        color = colors[strength]
        subset = psych_data[psych_data["illusion_strength"] == strength]
        pse_row = pse_summary[pse_summary["illusion_strength"] == strength]

        ax.scatter(
            subset["true_diff"],
            subset["prop_top"],
            color=color,
            s=35,
            zorder=3,
            alpha=0.75,
        )

        if not pse_row.empty and pse_row.iloc[0]["fit_success"]:
            pse = pse_row.iloc[0]["pse"]
            sigma = pse_row.iloc[0]["sigma"]
            y_fit = cumulative_gaussian(x_smooth, pse, sigma)
            ax.plot(
                x_smooth,
                y_fit,
                color=color,
                linewidth=1.8,
                label=f"{strength:+.0f}  (PSE={pse:+.3f})",
            )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        label="Physical equality",
    )
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("True physical difference  (top − bottom)", fontsize=12)
    ax.set_ylabel("P(respond 'Top')", fontsize=12)
    ax.set_title(
        f"Psychometric Functions — {model_name} on Müller-Lyer",
        fontsize=13,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(
        fontsize=8,
        loc="upper left",
        title="Illusion strength  (red = incongruent, green = congruent)",
        title_fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Figure 3 saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Plot illusion susceptibility figures (Figures 1–3)",
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
        "--participants-dir",
        type=str,
        default="./results/synthetic_participants",
        help="Directory of participant_*.jsonl files (default: ./results/synthetic_participants)",
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
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    participants_dir = Path(args.participants_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load fitted data (Figures 2 & 3) ────────────────────────────────────
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

    tested_diff_range = (psych_data["true_diff"].min(), psych_data["true_diff"].max())

    print(
        f"Loaded {len(psych_data)} aggregated data points across "
        f"{psych_data['illusion_strength'].nunique()} strength levels.\n"
    )

    # ── Figure 1: require raw participant files ───────────────────────────
    if not participants_dir.exists():
        print(
            f"⚠️  Skipping Figure 1: participants directory not found "
            f"({participants_dir}). Pass --participants-dir to enable."
        )
    else:
        plot_error_by_difficulty(
            participants_dir,
            output_dir / "fig1_error_by_difficulty.png",
            model_name=args.model_name,
            show=args.show,
        )

    # ── Figure 2 ──────────────────────────────────────────────────────────────
    plot_pse_vs_strength(
        pse_summary,
        tested_diff_range=tested_diff_range,
        output_path=output_dir / "fig2_pse_vs_strength.png",
        model_name=args.model_name,
        show=args.show,
    )

    # ── Figure 3 ─────────────────────────────────────────────────────────────
    plot_psychometric_curves(
        psych_data,
        pse_summary,
        output_dir / "fig3_psychometric_curves.png",
        model_name=args.model_name,
        show=args.show,
    )

    print("\nDone. Check the results/ directory for figures.")


if __name__ == "__main__":
    main()
