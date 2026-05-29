#!/usr/bin/env python3
"""
Generate GNNs-FAME-style scatter figure: Accuracy (x) vs SPD (y) with 95% CI error bars,
one panel per dataset. Saves to paper/figures/results_scatter.pdf.

Run from repo root: python scripts/plot_results_scatter.py
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Accuracy vs SPD with error bars")
    parser.add_argument(
        "--results-csv",
        type=str,
        default="outputs/multi_seed/results_full_table.csv",
        help="Path to results_full_table.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="paper/figures/results_scatter.pdf",
        help="Output figure path",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / args.results_csv
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Not found: {csv_path}. Run analyze_uncertainty first.")

    df = pd.read_csv(csv_path)
    datasets = df["dataset"].unique()
    models = df["model"].unique()

    # Marker and color per model (match GNNs-FAME style: GCN=square, GAT=circle; FAME variants distinct)
    style = {
        "GAT": ("o", "C0"),
        "GAT+FAME": ("o", "C1"),
        "GCN": ("s", "C2"),
        "GCN+FAME": ("s", "C3"),
    }
    # Fallback if key missing
    def get_style(m):
        return style.get(m, ("o", "gray"))

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5, 2.8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset]
        for _, row in sub.iterrows():
            model = row["model"]
            acc_mean = row["Accuracy_mean"]
            acc_lo = row["Accuracy_ci_low"]
            acc_hi = row["Accuracy_ci_high"]
            spd_mean = row["SPD_mean"]
            spd_lo = row["SPD_ci_low"]
            spd_hi = row["SPD_ci_high"]
            xerr = np.array([[acc_mean - acc_lo], [acc_hi - acc_mean]])
            yerr = np.array([[spd_mean - spd_lo], [spd_hi - spd_mean]])
            marker, color = get_style(model)
            ax.errorbar(
                acc_mean,
                spd_mean,
                xerr=xerr,
                yerr=yerr,
                fmt=marker,
                color=color,
                label=model,
                capsize=2,
                markersize=6,
            )
        ax.set_xlabel("Accuracy (↑)")
        ax.set_ylabel("SPD (↓)" if ax == axes[0] else "")
        ax.set_title(dataset.capitalize())
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlim(0.35, 0.85)
        ax.set_ylim(bottom=-0.01)
        ax.grid(True, alpha=0.3)
        # Ideal: bottom-right
        ax.axhline(0, color="gray", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    main()
