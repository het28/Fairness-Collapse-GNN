#!/usr/bin/env python3
"""
Print terminal tables: Multi-seed (fixed split) vs Split-seed (varying split)
in the same style as baseline vs mitigation: one row per seed, columns for
Acc, SPD, EOD for each condition.

- multi_seed: fixed split (e.g. split_seed=42), one run per (dataset, model, seed).
- split_seed: varying split seeds; we show the mean over split_seeds for each
  (dataset, model, seed) so rows align by seed.

Usage:
  python scripts/table_multi_seed_vs_split_seed.py
  python scripts/table_multi_seed_vs_split_seed.py --dataset credit
  python scripts/table_multi_seed_vs_split_seed.py --dataset german
  python scripts/table_multi_seed_vs_split_seed.py --dataset bail
  (default dataset is credit if omitted)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
MULTI_SEED_PATH = OUTPUTS / "multi_seed" / "all_runs.json"
SPLIT_SEED_PATH = OUTPUTS / "split_seed" / "all_runs.json"

SEEDS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
MODELS_BASE = ["GCN", "GAT", "GIN"]


def model_label(r: dict) -> str:
    m = r.get("_model", "")
    return m + ("+FAME" if r.get("_fame") else "")


def norm(s: str) -> str:
    return (s or "").lower().strip()


def load_multi_seed(dataset: str) -> dict:
    """(dataset, model_label, seed) -> { SPD, EOD, Accuracy }."""
    dataset = norm(dataset)
    if not MULTI_SEED_PATH.is_file():
        return {}
    runs = json.loads(MULTI_SEED_PATH.read_text())
    idx = {}
    for r in runs:
        if r.get("error"):
            continue
        if norm(r.get("_data_name", "")) != dataset:
            continue
        if r.get("_fairness_algo") not in (None, ""):
            continue
        if r.get("_mitigation") not in (None, ""):
            continue
        lab = model_label(r)
        seed = r.get("_seed")
        if lab == "" or seed is None:
            continue
        key = (norm(r.get("_data_name", "")), lab, seed)
        idx[key] = {
            "SPD": r.get("Statistical Parity Difference"),
            "EOD": r.get("Equal Opportunity Difference"),
            "Accuracy": r.get("Accuracy"),
        }
    return idx


def load_split_seed(dataset: str) -> dict:
    """(dataset, model_label, seed) -> { SPD, EOD, Accuracy } (mean over split_seeds)."""
    dataset = norm(dataset)
    if not SPLIT_SEED_PATH.is_file():
        return {}
    runs = json.loads(SPLIT_SEED_PATH.read_text())
    groups = defaultdict(list)  # key -> list of {SPD, EOD, Accuracy}
    for r in runs:
        if r.get("error"):
            continue
        if norm(r.get("_data_name", "")) != dataset:
            continue
        if r.get("_fairness_algo") not in (None, ""):
            continue
        if r.get("_mitigation") not in (None, ""):
            continue
        lab = model_label(r)
        seed = r.get("_seed")
        if lab == "" or seed is None:
            continue
        key = (norm(r.get("_data_name", "")), lab, seed)
        spd = r.get("Statistical Parity Difference")
        eod = r.get("Equal Opportunity Difference")
        acc = r.get("Accuracy")
        groups[key].append({"SPD": spd, "EOD": eod, "Accuracy": acc})
    idx = {}
    for key, vals in groups.items():
        accs = [float(v["Accuracy"]) for v in vals if v.get("Accuracy") is not None and not (isinstance(v["Accuracy"], float) and np.isnan(v["Accuracy"]))]
        spds = [float(v["SPD"]) for v in vals if v.get("SPD") is not None and not (isinstance(v.get("SPD"), float) and np.isnan(v["SPD"]))]
        eods = [float(v["EOD"]) for v in vals if v.get("EOD") is not None and not (isinstance(v.get("EOD"), float) and np.isnan(v["EOD"]))]
        idx[key] = {
            "Accuracy": float(np.mean(accs)) if accs else None,
            "SPD": float(np.mean(spds)) if spds else None,
            "EOD": float(np.mean(eods)) if eods else None,
        }
    return idx


def fmt(x) -> str:
    if x is None:
        return " — "
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    ap = argparse.ArgumentParser(description="Multi-seed vs Split-seed tables (same style as baseline vs mitigation)")
    ap.add_argument("--dataset", default="credit", choices=["credit", "german", "bail", "pokec-z", "pokec-n"], help="Dataset (default: credit)")
    args = ap.parse_args()
    dataset = norm(args.dataset)

    multi = load_multi_seed(dataset)
    split = load_split_seed(dataset)

    if not multi:
        print(f"No multi_seed runs for {dataset}. Ensure outputs/multi_seed/all_runs.json exists with {dataset} runs.")
        return 1
    if not split:
        print(f"No split_seed runs for {dataset}. Ensure outputs/split_seed/all_runs.json exists with {dataset} runs.")
        return 1

    models = [m for m in MODELS_BASE if any((dataset, m, s) in multi for s in SEEDS)]
    if not models:
        print(f"No models with seeds in SEEDS for {dataset}. Check multi_seed all_runs.json.")
        return 1

    for model in models:
        print()
        print("=" * 100)
        print(f"  Dataset: {dataset}  |  Model: {model}  |  Multi-seed (fixed split) vs Split-seed (mean over split seeds)")
        print("=" * 100)
        header = (
            f"{'Seed':<6} | "
            f"{'Multi-seed Acc':<16} {'Multi-seed SPD':<16} {'Multi-seed EOD':<16} | "
            f"{'Split-seed Acc':<16} {'Split-seed SPD':<16} {'Split-seed EOD':<16}"
        )
        print(header)
        print("-" * 100)
        for seed in SEEDS:
            km = (dataset, model, seed)
            m_row = multi.get(km, {})
            s_row = split.get(km, {})
            row = (
                f"{seed:<6} | "
                f"{fmt(m_row.get('Accuracy')):<16} {fmt(m_row.get('SPD')):<16} {fmt(m_row.get('EOD')):<16} | "
                f"{fmt(s_row.get('Accuracy')):<16} {fmt(s_row.get('SPD')):<16} {fmt(s_row.get('EOD')):<16}"
            )
            print(row)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
