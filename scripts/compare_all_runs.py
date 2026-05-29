#!/usr/bin/env python3
"""
Compare all experimental conditions run-by-run (paired by seed).

Loads all_runs.json from:
  - outputs/multi_seed (random split)
  - outputs/stratified_split (stratified split)
  - outputs/option_a_stability (Option A: stability regularization)
  - outputs/option_b_nifty (Option B: NIFTY-style)

For each (dataset, model, seed), aligns metrics across conditions so you can compare
every run—not just means/stds. Outputs:
  1. paired_by_seed.csv — one row per (dataset, model, seed), columns = SPD/EOD/Acc per condition
  2. all_runs_long.csv — every run in long format (dataset, model, seed, condition, SPD, EOD, Acc)
  3. paired_differences_summary.csv — per (dataset, model): mean and std of (condition_A - condition_B) across seeds
  4. run_by_run/ — one CSV per (dataset, model) with seeds as rows, conditions as columns for SPD/EOD/Acc
  5. correlation_across_seeds.csv — per (dataset, model): correlation of SPD (and EOD, Acc) between condition pairs
  6. spread_per_condition.csv — per (dataset, model, condition): min, max, IQR, range, std (full spread)
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
OUT_DIR = OUTPUTS / "comparison_all_runs"

CONDITIONS = [
    ("multi_seed", "random"),
    ("stratified_split", "stratified"),
    ("option_a_stability", "option_a"),
    ("option_b_nifty", "option_b"),
]
METRICS = [
    ("Statistical Parity Difference", "SPD"),
    ("Equal Opportunity Difference", "EOD"),
    ("Accuracy", "Accuracy"),
]


def model_label(r: dict) -> str:
    m = r.get("_model", "")
    return m + ("+FAME" if r.get("_fame") else "")


def load_condition(dir_name: str, condition_tag: str) -> List[dict]:
    path = OUTPUTS / dir_name / "all_runs.json"
    if not path.is_file():
        return []
    with open(path) as f:
        runs = json.load(f)
    # Exclude errors and PFR runs for fair comparison; include only baseline model configs
    out = []
    for r in runs:
        if r.get("error") or r.get("_fairness_algo") == "pfr":
            continue
        # For option_a/option_b we want runs with that mitigation; for random/stratified, no mitigation
        if condition_tag in ("option_a", "option_b"):
            if r.get("_mitigation") != condition_tag:
                continue
        else:
            if r.get("_mitigation") not in (None, ""):
                continue
        out.append(r)
    return out


def index_by_dataset_model_seed(runs: list, condition_tag: str) -> dict:
    """(dataset, model_label, seed) -> { SPD, EOD, Accuracy }."""
    idx = {}
    for r in runs:
        d = r.get("_data_name", "")
        lab = model_label(r)
        seed = r.get("_seed")
        if d == "" or lab == "" or seed is None:
            continue
        key = (d, lab, seed)
        idx[key] = {
            "SPD": r.get("Statistical Parity Difference"),
            "EOD": r.get("Equal Opportunity Difference"),
            "Accuracy": r.get("Accuracy"),
        }
    return idx


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "run_by_run").mkdir(exist_ok=True)

    # Load all conditions
    condition_runs = {}
    for dir_name, tag in CONDITIONS:
        runs = load_condition(dir_name, tag)
        condition_runs[tag] = index_by_dataset_model_seed(runs, tag)
        print(f"Loaded {tag}: {len(runs)} runs -> {len(condition_runs[tag])} indexed by (dataset, model, seed)")

    tags = [t for _, t in CONDITIONS if condition_runs[t]]
    if len(tags) < 2:
        print("Need at least two conditions with data.")
        return 1

    # All (dataset, model) and seeds that appear in any condition
    all_keys = set()
    for idx in condition_runs.values():
        for (d, lab, seed) in idx:
            all_keys.add((d, lab, seed))
    # Per (dataset, model), which seeds have at least one condition
    seeds_by_dm = defaultdict(set)
    for (d, lab, seed) in all_keys:
        seeds_by_dm[(d, lab)].add(seed)
    datasets_models = sorted(seeds_by_dm.keys())

    # ---- 1. paired_by_seed.csv: one row per (dataset, model, seed), columns per condition ----
    rows_paired = []
    for (d, lab) in datasets_models:
        for seed in sorted(seeds_by_dm[(d, lab)]):
            row = {"dataset": d, "model": lab, "seed": seed}
            for tag in tags:
                idx = condition_runs[tag]
                v = idx.get((d, lab, seed), {})
                for _, short in METRICS:
                    row[f"{short}_{tag}"] = v.get(short)
            rows_paired.append(row)

    with open(OUT_DIR / "paired_by_seed.csv", "w", newline="") as f:
        fieldnames = ["dataset", "model", "seed"] + [f"{short}_{tag}" for short in ["SPD", "EOD", "Accuracy"] for tag in tags]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_paired)
    print(f"Wrote {OUT_DIR / 'paired_by_seed.csv'} ({len(rows_paired)} rows)")

    # ---- 2. all_runs_long.csv: every run, long format ----
    rows_long = []
    for tag in tags:
        for (d, lab, seed), vals in condition_runs[tag].items():
            rows_long.append({
                "dataset": d, "model": lab, "seed": seed, "condition": tag,
                "SPD": vals.get("SPD"), "EOD": vals.get("EOD"), "Accuracy": vals.get("Accuracy"),
            })
    with open(OUT_DIR / "all_runs_long.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "seed", "condition", "SPD", "EOD", "Accuracy"])
        w.writeheader()
        w.writerows(rows_long)
    print(f"Wrote {OUT_DIR / 'all_runs_long.csv'} ({len(rows_long)} rows)")

    # ---- 3. paired_differences_summary: per (dataset, model), mean and std of (A - B) across seeds ----
    diff_rows = []
    for (d, lab) in datasets_models:
        seeds = sorted(seeds_by_dm[(d, lab)])
        for i, tag_a in enumerate(tags):
            for tag_b in tags[i + 1 :]:
                diffs = {short: [] for _, short in METRICS}
                for seed in seeds:
                    va = condition_runs[tag_a].get((d, lab, seed), {})
                    vb = condition_runs[tag_b].get((d, lab, seed), {})
                    for _, short in METRICS:
                        a, b = va.get(short), vb.get(short)
                        if a is not None and b is not None and not (isinstance(a, float) and np.isnan(a)) and not (isinstance(b, float) and np.isnan(b)):
                            diffs[short].append(float(a) - float(b))
                for _, short in METRICS:
                    arr = np.array(diffs[short])
                    n = len(arr)
                    diff_rows.append({
                        "dataset": d, "model": lab,
                        "condition_A": tag_a, "condition_B": tag_b, "metric": short,
                        "diff_mean": float(np.mean(arr)) if n else None,
                        "diff_std": float(np.std(arr, ddof=1)) if n > 1 else None,
                        "n_paired": n,
                    })
    with open(OUT_DIR / "paired_differences_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "condition_A", "condition_B", "metric", "diff_mean", "diff_std", "n_paired"])
        w.writeheader()
        w.writerows(diff_rows)
    print(f"Wrote {OUT_DIR / 'paired_differences_summary.csv'}")

    # ---- 4. run_by_run/: one CSV per (dataset, model) ----
    for (d, lab) in datasets_models:
        seeds = sorted(seeds_by_dm[(d, lab)])
        rows = []
        for seed in seeds:
            row = {"seed": seed}
            for tag in tags:
                v = condition_runs[tag].get((d, lab, seed), {})
                for _, short in METRICS:
                    row[f"{short}_{tag}"] = v.get(short)
            rows.append(row)
        safe = f"{d}_{lab}".replace("+", "_")
        with open(OUT_DIR / "run_by_run" / f"{safe}.csv", "w", newline="") as f:
            fieldnames = ["seed"] + [f"{short}_{tag}" for short in ["SPD", "EOD", "Accuracy"] for tag in tags]
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote run_by_run CSVs for {len(datasets_models)} (dataset, model) pairs -> {OUT_DIR / 'run_by_run'}")

    # ---- 5. correlation_across_seeds: per (dataset, model), correlation between condition pairs ----
    corr_rows = []
    for (d, lab) in datasets_models:
        seeds = sorted(seeds_by_dm[(d, lab)])
        for i, tag_a in enumerate(tags):
            for tag_b in tags[i + 1 :]:
                for _, short in METRICS:
                    va = [condition_runs[tag_a].get((d, lab, s), {}).get(short) for s in seeds]
                    vb = [condition_runs[tag_b].get((d, lab, s), {}).get(short) for s in seeds]
                    pairs = [(float(a), float(b)) for a, b in zip(va, vb) if a is not None and b is not None and not (isinstance(a, float) and np.isnan(a)) and not (isinstance(b, float) and np.isnan(b))]
                    if len(pairs) < 3:
                        r = None
                    else:
                        arr_a, arr_b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
                        r = float(np.corrcoef(arr_a, arr_b)[0, 1]) if np.std(arr_a) > 1e-12 and np.std(arr_b) > 1e-12 else None
                    corr_rows.append({
                        "dataset": d, "model": lab, "condition_A": tag_a, "condition_B": tag_b, "metric": short,
                        "correlation": r, "n_paired": len(pairs),
                    })
    with open(OUT_DIR / "correlation_across_seeds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "condition_A", "condition_B", "metric", "correlation", "n_paired"])
        w.writeheader()
        w.writerows(corr_rows)
    print(f"Wrote {OUT_DIR / 'correlation_across_seeds.csv'}")

    # ---- 6. spread_per_condition: min, max, IQR, range, std per (dataset, model, condition) ----
    spread_rows = []
    for (d, lab) in datasets_models:
        for tag in tags:
            vals = [condition_runs[tag].get((d, lab, s), {}) for s in sorted(seeds_by_dm[(d, lab)])]
            for _, short in METRICS:
                arr = np.array([v.get(short) for v in vals if v.get(short) is not None and not (isinstance(v.get(short), float) and np.isnan(v.get(short)))], dtype=float)
                if len(arr) == 0:
                    spread_rows.append({"dataset": d, "model": lab, "condition": tag, "metric": short, "n": 0, "min": None, "max": None, "range": None, "std": None, "q25": None, "q75": None, "iqr": None})
                    continue
                q25, q75 = np.percentile(arr, [25, 75])
                spread_rows.append({
                    "dataset": d, "model": lab, "condition": tag, "metric": short,
                    "n": len(arr), "min": float(np.min(arr)), "max": float(np.max(arr)), "range": float(np.max(arr) - np.min(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25),
                })
    with open(OUT_DIR / "spread_per_condition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "condition", "metric", "n", "min", "max", "range", "std", "q25", "q75", "iqr"])
        w.writeheader()
        w.writerows(spread_rows)
    print(f"Wrote {OUT_DIR / 'spread_per_condition.csv'}")

    # ---- Brief summary to stdout ----
    print("\n--- Paired difference (stratified - random) for SPD, where both exist ---")
    for (d, lab) in datasets_models[:6]:  # first 6
        dr = [r for r in diff_rows if r["dataset"] == d and r["model"] == lab and r["condition_A"] == "random" and r["condition_B"] == "stratified" and r["metric"] == "SPD"]
        for r in dr:
            if r["n_paired"]:
                print(f"  {d} {lab}: diff_mean={r['diff_mean']:.4f} diff_std={r['diff_std']:.4f} n={r['n_paired']}")

    print("\nDone. All outputs under", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
