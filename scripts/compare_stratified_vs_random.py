#!/usr/bin/env python3
"""
Compare stratified-split vs random-split (multi_seed) results.
Run after: (1) multi_seed all_runs.json exists, (2) stratified_split all_runs.json has multiple seeds (e.g. --n_seeds 10).
Reads outputs/multi_seed/results_summary.csv (or all_runs.json) and outputs/stratified_split/all_runs.json,
outputs a comparison table: mean ± std and 95% CI for SPD, EOD, Accuracy per (dataset, model).
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MULTI_SEED_DIR = PROJECT_ROOT / "outputs" / "multi_seed"
STRATIFIED_DIR = PROJECT_ROOT / "outputs" / "stratified_split"


def load_runs(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def summarize_runs(runs: list, metric: str) -> dict:
    """Per (dataset, model_label) -> { mean, std, n, ci_low, ci_high }."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in runs:
        if "error" in r:
            continue
        d = r.get("_data_name", "")
        m = r.get("_model", "")
        f = r.get("_fame", False)
        label = m + ("+FAME" if f else "")
        v = r.get(metric)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            groups[(d, label)].append(float(v))
    out = {}
    for (d, label), vals in groups.items():
        vals = np.array(vals)
        n = len(vals)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 0 else 0
        from scipy import stats
        h = se * stats.t.ppf(0.975, n - 1) if n > 1 else 0
        out[(d, label)] = {"mean": mean, "std": std, "n": n, "ci_low": mean - h, "ci_high": mean + h}
    return out


def main():
    multi_path = MULTI_SEED_DIR / "all_runs.json"
    strat_path = STRATIFIED_DIR / "all_runs.json"
    if not multi_path.is_file():
        print(f"Not found: {multi_path}", file=sys.stderr)
        sys.exit(1)
    if not strat_path.is_file():
        print(f"Not found: {strat_path}", file=sys.stderr)
        sys.exit(1)

    multi_runs = load_runs(multi_path)
    strat_runs = load_runs(strat_path)
    # Exclude PFR for fair comparison
    multi_runs = [r for r in multi_runs if r.get("_fairness_algo") != "pfr"]
    strat_runs = [r for r in strat_runs if r.get("_fairness_algo") != "pfr"]

    for metric, name in [
        ("Statistical Parity Difference", "SPD"),
        ("Equal Opportunity Difference", "EOD"),
        ("Accuracy", "Accuracy"),
    ]:
        multi_sum = summarize_runs(multi_runs, metric)
        strat_sum = summarize_runs(strat_runs, metric)
        keys = sorted(set(multi_sum.keys()) | set(strat_sum.keys()))
        print(f"\n{'='*80}")
        print(f"  {name} (mean ± std, n)  |  Random (multi_seed)  vs  Stratified")
        print("=" * 80)
        print(f"{'dataset':<8} {'model':<12} {'random':<28} {'stratified':<28} stratified < random?")
        print("-" * 80)
        for (d, label) in keys:
            m = multi_sum.get((d, label), {})
            s = strat_sum.get((d, label), {})
            rm = m.get("mean", np.nan)
            rs = m.get("std", np.nan)
            rn = m.get("n", 0)
            sm = s.get("mean", np.nan)
            ss = s.get("std", np.nan)
            sn = s.get("n", 0)
            r_str = f"{rm:.3f} ± {rs:.3f} (n={rn})" if rn else "—"
            s_str = f"{sm:.3f} ± {ss:.3f} (n={sn})" if sn else "—"
            better = "yes" if (sn and rn and sm < rm) else "no" if (sn and rn) else "—"
            print(f"{d:<8} {label:<12} {r_str:<28} {s_str:<28} {better}")
    print()
    out_csv = STRATIFIED_DIR / "comparison_stratified_vs_random.csv"
    # Write a simple CSV: dataset, model, metric, random_mean, random_std, stratified_mean, stratified_std
    rows = []
    for metric in ["Statistical Parity Difference", "Equal Opportunity Difference", "Accuracy"]:
        multi_sum = summarize_runs(multi_runs, metric)
        strat_sum = summarize_runs(strat_runs, metric)
        for (d, label) in sorted(set(multi_sum.keys()) | set(strat_sum.keys())):
            m, s = multi_sum.get((d, label), {}), strat_sum.get((d, label), {})
            short = "SPD" if "Parity" in metric else "EOD" if "Opportunity" in metric else "Accuracy"
            rows.append({
                "dataset": d, "model": label, "metric": short,
                "random_mean": m.get("mean"), "random_std": m.get("std"), "random_n": m.get("n"),
                "stratified_mean": s.get("mean"), "stratified_std": s.get("std"), "stratified_n": s.get("n"),
            })
    if rows:
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
