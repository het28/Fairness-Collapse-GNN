#!/usr/bin/env python3
"""
Summarize runs where SPD and EOD are 0.000: many are trivial (constant) predictors.

Reads results_all_runs.csv (stratified_split or any condition). For each (dataset, model):
- Count runs with SPD=0 and EOD=0.
- Among those, count "trivial": pred_pos_S0 and pred_pos_S1 both 0 or both 1
  (model predicts same class for everyone → SPD/EOD are 0 by construction, not fairness).

Usage:
  python scripts/analyze_trivial_runs.py [outputs/stratified_split/results_all_runs.csv]
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "outputs" / "stratified_split" / "results_all_runs.csv"

# Tolerance for "zero"
TOL = 1e-6


def is_zero(x) -> bool:
    try:
        return abs(float(x)) <= TOL
    except (TypeError, ValueError):
        return False


def is_trivial(p0, p1) -> bool:
    """Trivial = constant predictor: both rates 0 or both 1 (or very close)."""
    try:
        p0, p1 = float(p0), float(p1)
        return (p0 <= TOL and p1 <= TOL) or (p0 >= 1 - TOL and p1 >= 1 - TOL)
    except (TypeError, ValueError):
        return False


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.is_file():
        print(f"File not found: {csv_path}")
        return 1

    # (dataset, model) -> { total, zero_spd_eod, trivial }
    by_dm = defaultdict(lambda: {"total": 0, "zero_spd_eod": 0, "trivial": 0})

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "Statistical Parity Difference" not in reader.fieldnames:
            print("CSV missing 'Statistical Parity Difference' column.")
            return 1
        for row in reader:
            dataset = row.get("dataset", "").strip()
            model = row.get("model", "").strip()
            if not dataset or not model:
                continue
            key = (dataset, model)
            by_dm[key]["total"] += 1
            spd = row.get("Statistical Parity Difference", "")
            eod = row.get("Equal Opportunity Difference", "")
            if is_zero(spd) and is_zero(eod):
                by_dm[key]["zero_spd_eod"] += 1
                p0 = row.get("diagnostics_pred_pos_S0", "")
                p1 = row.get("diagnostics_pred_pos_S1", "")
                if is_trivial(p0, p1):
                    by_dm[key]["trivial"] += 1

    # Print summary table
    print(f"Source: {csv_path.name}")
    print()
    print("Runs with SPD=0 and EOD=0: many are trivial (constant predictor = all 0 or all 1).")
    print("Trivial => SPD and EOD are 0 by construction, not because the model is fair.")
    print()
    header = f"{'Dataset':<10} {'Model':<14} {'Total':<8} {'SPD=EOD=0':<12} {'Trivial (0/0 or 1/1)':<22} {'% zero':<8} {'% trivial of zero':<18}"
    print(header)
    print("-" * 92)

    for (dataset, model) in sorted(by_dm.keys()):
        d = by_dm[(dataset, model)]
        total = d["total"]
        zero = d["zero_spd_eod"]
        trivial = d["trivial"]
        pct_zero = (100.0 * zero / total) if total else 0
        pct_trivial_of_zero = (100.0 * trivial / zero) if zero else 0
        print(f"{dataset:<10} {model:<14} {total:<8} {zero:<12} {trivial:<22} {pct_zero:<8.1f} {pct_trivial_of_zero:<18.1f}")

    print()
    print("Interpretation: High 'Trivial' count for a model (e.g. GIN on credit) means most")
    print("runs with 0.000 SPD/EOD are constant predictors, not meaningfully fair.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
