#!/usr/bin/env python3
"""
Print terminal tables: Baseline (stratified) vs Mitigation (PFR, Option A, Option B)
for seeds 5, 10, 15, ..., 50 and models GCN, GAT, GIN.

Usage:
  python scripts/table_baseline_vs_mitigation.py [--dataset credit] [--mitigation pfr]
  python scripts/table_baseline_vs_mitigation.py --dataset german --mitigation pfr
  python scripts/table_baseline_vs_mitigation.py --dataset german --mitigation option_a
  python scripts/table_baseline_vs_mitigation.py --dataset german --mitigation option_b
  python scripts/table_baseline_vs_mitigation.py --dataset bail --mitigation pfr
  # Residual GNN runs (baseline=residual_gnn, mitigation=residual_gnn_pfr/option_a/option_b), seeds 1,5,...,50:
  python scripts/table_baseline_vs_mitigation.py --residual-runs --dataset credit --mitigation pfr
  # Baseline = stratified_split or residual_gnn. Mitigation: pfr, option_a, or option_b.
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Run outputs (stratified, residual, mitigations) live under output_base, default "outputs"
OUTPUTS = PROJECT_ROOT / "outputs"
STRATIFIED_PATH = OUTPUTS / "stratified_split" / "all_runs.json"
MULTI_SEED_PATH = OUTPUTS / "multi_seed" / "all_runs.json"
OPTION_A_PATH = OUTPUTS / "option_a_stability" / "all_runs.json"
OPTION_B_PATH = OUTPUTS / "option_b_nifty" / "all_runs.json"
RESIDUAL_GNN_PATH = OUTPUTS / "residual_gnn" / "all_runs.json"
RESIDUAL_PFR_PATH = OUTPUTS / "residual_gnn_pfr" / "all_runs.json"
RESIDUAL_OPTION_A_PATH = OUTPUTS / "residual_gnn_option_a" / "all_runs.json"
RESIDUAL_OPTION_B_PATH = OUTPUTS / "residual_gnn_option_b" / "all_runs.json"

SEEDS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
SEEDS_RESIDUAL = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
MODELS_BASE = ["GCN", "GAT", "GIN"]  # without +FAME for clarity


def model_label(r: dict) -> str:
    m = r.get("_model", "")
    return m + ("+FAME" if r.get("_fame") else "")


def norm(s: str) -> str:
    return (s or "").lower().strip()


def load_baseline(dataset: str, residual_runs: bool = False) -> dict:
    """(dataset, model_label, seed) -> { SPD, EOD, Accuracy }."""
    dataset = norm(dataset)
    path = RESIDUAL_GNN_PATH if residual_runs else STRATIFIED_PATH
    if not path.is_file():
        return {}
    runs = json.loads(path.read_text())
    idx = {}
    for r in runs:
        if r.get("error"):
            continue
        if norm(r.get("_data_name", "")) != dataset:
            continue
        if residual_runs and not r.get("_residual"):
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


def load_mitigation(mitigation: str, dataset: str, residual_runs: bool = False) -> dict:
    """(dataset, model_label, seed) -> { SPD, EOD, Accuracy }."""
    dataset = norm(dataset)
    path_map = {
        "pfr": RESIDUAL_PFR_PATH if residual_runs else MULTI_SEED_PATH,
        "option_a": RESIDUAL_OPTION_A_PATH if residual_runs else OPTION_A_PATH,
        "option_b": RESIDUAL_OPTION_B_PATH if residual_runs else OPTION_B_PATH,
    }
    path = path_map.get(mitigation)
    if not path or not path.is_file():
        return {}
    runs = json.loads(path.read_text())
    idx = {}
    for r in runs:
        if r.get("error"):
            continue
        if norm(r.get("_data_name", "")) != dataset:
            continue
        if residual_runs and not r.get("_residual"):
            continue
        if mitigation == "pfr" and r.get("_fairness_algo") != "pfr":
            continue
        if mitigation in ("option_a", "option_b") and r.get("_mitigation") != mitigation:
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


def fmt(x) -> str:
    if x is None:
        return " — "
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    ap = argparse.ArgumentParser(description="Baseline vs Mitigation tables by dataset")
    ap.add_argument("--dataset", default="credit", choices=["credit", "german", "bail", "pokec-z", "pokec-n"], help="Dataset: credit, german, bail, pokec-z, pokec-n (default: credit)")
    ap.add_argument("--mitigation", default="pfr", choices=["pfr", "option_a", "option_b"], help="Mitigation to compare: pfr, option_a, or option_b (default: pfr)")
    ap.add_argument("--residual-runs", action="store_true", help="Use residual GNN runs: baseline=residual_gnn, mitigation=residual_gnn_pfr/option_a/option_b. Seeds 1,5,...,50.")
    args = ap.parse_args()
    dataset = norm(args.dataset)
    residual_runs = getattr(args, "residual_runs", False)
    seeds = SEEDS_RESIDUAL if residual_runs else SEEDS

    baseline = load_baseline(args.dataset, residual_runs=residual_runs)
    mitigation = load_mitigation(args.mitigation, args.dataset, residual_runs=residual_runs)

    if not baseline:
        src = "residual_gnn" if residual_runs else "stratified_split"
        print(f"No baseline ({src} {dataset}) runs found. Ensure outputs/{src}/all_runs.json exists with {dataset} runs.")
        return 1
    if not mitigation:
        src = f"residual_gnn_{args.mitigation}" if residual_runs else ("option_a_stability" if args.mitigation == "option_a" else "option_b_nifty" if args.mitigation == "option_b" else "multi_seed")
        print(f"No mitigation ({args.mitigation}) runs found for {dataset}. Check outputs/{src}.")
        return 1

    # Use models that appear in baseline without +FAME for a clean comparison
    models = [m for m in MODELS_BASE if any((dataset, m, s) in baseline for s in seeds)]

    base_label = "Residual GNN (no mitigation)" if residual_runs else "Stratified (no mitigation)"
    for model in models:
        print()
        print("=" * 100)
        mit_label = {"option_a": "Option A", "option_b": "Option B"}.get(args.mitigation, args.mitigation.upper())
        if residual_runs:
            mit_label = f"Residual+{mit_label}"
        print(f"  Dataset: {dataset}  |  Model: {model}  |  Baseline = {base_label}  |  Mitigation = {mit_label}")
        print("=" * 100)
        header = (
            f"{'Seed':<6} | "
            f"{'Baseline Acc':<14} {'Baseline SPD':<14} {'Baseline EOD':<14} | "
            f"{mit_label + ' Acc':<14} {mit_label + ' SPD':<14} {mit_label + ' EOD':<14}"
        )
        print(header)
        print("-" * 100)
        for seed in seeds:
            kb = (dataset, model, seed)
            km = (dataset, model, seed)
            b = baseline.get(kb, {})
            m = mitigation.get(km, {})
            row = (
                f"{seed:<6} | "
                f"{fmt(b.get('Accuracy')):<14} {fmt(b.get('SPD')):<14} {fmt(b.get('EOD')):<14} | "
                f"{fmt(m.get('Accuracy')):<14} {fmt(m.get('SPD')):<14} {fmt(m.get('EOD')):<14}"
            )
            print(row)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
