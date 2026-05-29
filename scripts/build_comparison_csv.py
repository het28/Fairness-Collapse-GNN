#!/usr/bin/env python3
"""
Build comparison CSV: baseline (multi_seed, fixed split) vs split_seed (varying split seeds).
Compares only vanilla GCN and GAT for fair comparison. Output: outputs/comparison_baseline_vs_split_seed.csv
"""
import csv
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parent.parent
    baseline_path = repo / "outputs" / "multi_seed" / "results_full_table.csv"
    split_seed_path = repo / "outputs" / "split_seed" / "results_full_table.csv"
    out_path = repo / "outputs" / "comparison_baseline_vs_split_seed.csv"

    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")
    if not split_seed_path.is_file():
        raise FileNotFoundError(f"Split-seed not found: {split_seed_path}")

    # Key columns to compare
    key_cols = [
        "dataset", "model", "experiment", "n_runs",
        "train_split", "val_split", "test_split",
        "Accuracy_mean", "Accuracy_std", "Accuracy_ci_low", "Accuracy_ci_high",
        "SPD_mean", "SPD_std", "SPD_ci_low", "SPD_ci_high",
        "EOD_mean", "EOD_std", "EOD_ci_low", "EOD_ci_high",
    ]

    rows = []

    # Baseline: only GCN and GAT (no +FAME)
    with open(baseline_path) as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("model") not in ("GCN", "GAT"):
                continue
            rows.append({
                "dataset": row["dataset"],
                "model": row["model"],
                "experiment": "baseline",
                "n_runs": row.get("n_seeds", ""),
                "train_split": "0.80",
                "val_split": "0.10",
                "test_split": "0.10",
                "Accuracy_mean": row.get("Accuracy_mean", ""),
                "Accuracy_std": row.get("Accuracy_std", ""),
                "Accuracy_ci_low": row.get("Accuracy_ci_low", ""),
                "Accuracy_ci_high": row.get("Accuracy_ci_high", ""),
                "SPD_mean": row.get("SPD_mean", ""),
                "SPD_std": row.get("SPD_std", ""),
                "SPD_ci_low": row.get("SPD_ci_low", ""),
                "SPD_ci_high": row.get("SPD_ci_high", ""),
                "EOD_mean": row.get("EOD_mean", ""),
                "EOD_std": row.get("EOD_std", ""),
                "EOD_ci_low": row.get("EOD_ci_low", ""),
                "EOD_ci_high": row.get("EOD_ci_high", ""),
            })

    # Split-seed: GCN and GAT (same models)
    with open(split_seed_path) as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("model") not in ("GCN", "GAT"):
                continue
            rows.append({
                "dataset": row["dataset"],
                "model": row["model"],
                "experiment": "split_seed",
                "n_runs": row.get("n_seeds", ""),
                "train_split": "0.80",
                "val_split": "0.10",
                "test_split": "0.10",
                "Accuracy_mean": row.get("Accuracy_mean", ""),
                "Accuracy_std": row.get("Accuracy_std", ""),
                "Accuracy_ci_low": row.get("Accuracy_ci_low", ""),
                "Accuracy_ci_high": row.get("Accuracy_ci_high", ""),
                "SPD_mean": row.get("SPD_mean", ""),
                "SPD_std": row.get("SPD_std", ""),
                "SPD_ci_low": row.get("SPD_ci_low", ""),
                "SPD_ci_high": row.get("SPD_ci_high", ""),
                "EOD_mean": row.get("EOD_mean", ""),
                "EOD_std": row.get("EOD_std", ""),
                "EOD_ci_low": row.get("EOD_ci_low", ""),
                "EOD_ci_high": row.get("EOD_ci_high", ""),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=key_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
