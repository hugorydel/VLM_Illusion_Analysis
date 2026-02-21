#!/usr/bin/env python3
"""
fit_psychometrics.py - Fit sigmoid psychometric functions and extract PSE (Phase 3)

Reads raw_responses.jsonl, groups responses by illusion_strength, and for each
strength level fits a cumulative Gaussian (sigmoid) to the proportion of "Top"
responses as a function of true physical difference.

The Point of Subjective Equality (PSE) is the x-value where the fitted curve
crosses 50% — i.e., the physical difference needed for the model to be indifferent.
  PSE = 0    → no bias
  PSE > 0    → model perceives top as shorter (needs extra length to seem equal)
  PSE < 0    → model perceives top as longer

Outputs:
    results/pse_summary.csv      — one row per illusion strength level
    results/psychometric_data.csv — proportion-top per (strength, diff) cell

Usage:
    python fit_psychometrics.py [--results PATH] [--output-dir PATH]

Requirements:
    pip install numpy scipy pandas
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf

# ============================================================================
# PSYCHOMETRIC FUNCTION
# ============================================================================


def cumulative_gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Cumulative Gaussian (Φ) — the standard psychometric function.

    P("Top" | x) = Φ((x − μ) / σ)

    Args:
        x:     True physical difference (top − bottom line length).
        mu:    PSE — the x-value at which P = 0.5.
        sigma: Slope parameter (JND / √2); smaller = steeper curve.

    Returns:
        Probability of responding "Top".
    """
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


# ============================================================================
# DATA LOADING
# ============================================================================


def load_responses(jsonl_path: Path) -> pd.DataFrame:
    """
    Load raw VLM responses from JSONL into a DataFrame.

    Columns: image_id, illusion_strength, true_diff, response, correct
    """
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {jsonl_path}")

    df = pd.DataFrame(records)
    df["responded_top"] = (df["response"] == "Top").astype(int)
    return df


# ============================================================================
# PSYCHOMETRIC DATA AGGREGATION
# ============================================================================


def aggregate_psychometric_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute proportion of "Top" responses per (illusion_strength, true_diff) cell.

    Returns a DataFrame with columns:
        illusion_strength, true_diff, n_trials, n_top, prop_top
    """
    grouped = (
        df.groupby(["illusion_strength", "true_diff"])
        .agg(
            n_trials=("responded_top", "count"),
            n_top=("responded_top", "sum"),
        )
        .reset_index()
    )
    grouped["prop_top"] = grouped["n_top"] / grouped["n_trials"]
    return grouped.sort_values(["illusion_strength", "true_diff"]).reset_index(
        drop=True
    )


# ============================================================================
# PSE FITTING
# ============================================================================


def fit_pse(
    diff_values: np.ndarray,
    prop_top: np.ndarray,
) -> dict:
    """
    Fit a cumulative Gaussian to the psychometric data and return PSE + slope.

    Args:
        diff_values: Array of true physical differences (x-axis).
        prop_top:  Corresponding proportions of "Top" responses (y-axis).

    Returns:
        Dict with keys: pse, sigma, fit_success, note.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                cumulative_gaussian,
                diff_values,
                prop_top,
                p0=[0.0, 0.1],  # initial guess: unbiased, moderate slope
                bounds=([-1.0, 0.001], [1.0, 2.0]),
                maxfev=10_000,
            )

        pse, sigma = popt
        perr = np.sqrt(np.diag(pcov))
        return {
            "pse": round(float(pse), 4),
            "sigma": round(float(sigma), 4),
            "pse_se": round(float(perr[0]), 4),
            "sigma_se": round(float(perr[1]), 4),
            "fit_success": True,
            "note": "",
        }

    except Exception as e:
        return {
            "pse": np.nan,
            "sigma": np.nan,
            "pse_se": np.nan,
            "sigma_se": np.nan,
            "fit_success": False,
            "note": str(e),
        }


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Fit psychometric functions and extract PSE per illusion strength",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results",
        type=str,
        default="./results/raw_responses.jsonl",
        help="Path to raw_responses.jsonl (default: ./results/raw_responses.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save output CSVs (default: ./results)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        print("Run query_vlm.py first.")
        raise SystemExit(1)

    # Load
    print(f"Loading responses from {results_path}...")
    df = load_responses(results_path)
    print(
        f"  {len(df)} total responses across {df['illusion_strength'].nunique()} strength levels\n"
    )

    # Aggregate
    psych_data = aggregate_psychometric_data(df)
    psych_path = output_dir / "psychometric_data.csv"
    psych_data.to_csv(psych_path, index=False)
    print(f"✓ Psychometric data saved to {psych_path}\n")

    # Fit PSE per strength level
    print("=" * 60)
    print("PSE FITTING RESULTS")
    print("=" * 60)
    print(f"  {'Strength':>8}  {'PSE':>8}  {'±SE':>8}  {'Sigma':>8}  {'OK':>4}")
    print("  " + "-" * 46)

    pse_rows = []
    for strength in sorted(df["illusion_strength"].unique()):
        subset = psych_data[psych_data["illusion_strength"] == strength]
        result = fit_pse(
            subset["true_diff"].values,
            subset["prop_top"].values,
        )
        pse_rows.append({"illusion_strength": strength, **result})

        ok_str = "✓" if result["fit_success"] else "✗"
        pse_str = f"{result['pse']:+.4f}" if not np.isnan(result["pse"]) else "   NaN"
        se_str = (
            f"{result['pse_se']:.4f}" if not np.isnan(result["pse_se"]) else "   NaN"
        )
        sig_str = (
            f"{result['sigma']:.4f}" if not np.isnan(result["sigma"]) else "   NaN"
        )
        note = f"  [{result['note']}]" if result["note"] else ""
        print(
            f"  {strength:>8}  {pse_str:>8}  {se_str:>8}  {sig_str:>8}  {ok_str:>4}{note}"
        )

    print("=" * 60)

    pse_df = pd.DataFrame(pse_rows)
    pse_path = output_dir / "pse_summary.csv"
    pse_df.to_csv(pse_path, index=False)
    print(f"\n✓ PSE summary saved to {pse_path}")
    print("\nInterpretation:")
    print("  PSE = 0   → no systematic bias at this illusion strength")
    print(
        "  PSE > 0   → model perceives top line as shorter (illusion biases toward bottom)"
    )
    print(
        "  PSE < 0   → model perceives top line as longer  (illusion biases toward top)"
    )
    print("\nCore finding: does PSE shift systematically with illusion strength?")


if __name__ == "__main__":
    main()
