#!/usr/bin/env python3
"""
Analyze inherent bias in the graph datasets (credit, german, bail).
Computes data-level disparity: base rates by sensitive group, SPD in the data, correlation S–Y.
Uses the same data loaders as the experiments (GNNs-FAME preprocess_data).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# Resolve project root and GNNs-FAME root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GNN_FAME_ROOT = PROJECT_ROOT / "GNNs-FAME-main"
if not GNN_FAME_ROOT.is_dir():
    raise FileNotFoundError(f"GNNs-FAME not found at {GNN_FAME_ROOT}")

os.chdir(GNN_FAME_ROOT)
if str(GNN_FAME_ROOT) not in sys.path:
    sys.path.insert(0, str(GNN_FAME_ROOT))

from preprocess_data import preprocess_data


def binarize_sens_if_needed(sens: np.ndarray) -> np.ndarray:
    """If sensitive attribute has more than 2 values, binarize at median (S=1 if >= median)."""
    uniq = np.unique(sens)
    if len(uniq) <= 2:
        return sens
    return (sens >= np.median(sens)).astype(np.int64)


def compute_bias_metrics(y: np.ndarray, sens: np.ndarray, subset_name: str = "full") -> dict:
    """
    Compute data-level bias metrics for binary Y and binary S.
    - Base rate (positive rate) per group: P(Y=1|S=0), P(Y=1|S=1)
    - Base rate difference (inherent SPD): P(Y=1|S=1) - P(Y=1|S=0)
    - Group sizes and proportions
    - Correlation between S and Y (point-biserial)
    """
    sens = binarize_sens_if_needed(sens)
    y = np.asarray(y).ravel()
    sens = np.asarray(sens).ravel()
    n = len(y)
    mask0 = sens == 0
    mask1 = sens == 1
    n0 = mask0.sum()
    n1 = mask1.sum()
    if n0 == 0 or n1 == 0:
        return {
            "subset": subset_name,
            "n": n,
            "n_S0": int(n0),
            "n_S1": int(n1),
            "frac_S1": float(n1 / n) if n else 0,
            "base_rate_S0": None,
            "base_rate_S1": None,
            "base_rate_diff_inherent_SPD": None,
            "corr_S_Y": None,
        }
    y0 = y[mask0]
    y1 = y[mask1]
    br0 = float(np.mean(y0))
    br1 = float(np.mean(y1))
    base_rate_diff = br1 - br0  # inherent SPD in the data
    corr = float(np.corrcoef(sens.astype(float), y.astype(float))[0, 1]) if n > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0
    # 4-cell (S,Y): (S=0,Y=0), (S=0,Y=1), (S=1,Y=0), (S=1,Y=1)
    n_00 = int((mask0 & (y == 0)).sum())
    n_01 = int((mask0 & (y == 1)).sum())
    n_10 = int((mask1 & (y == 0)).sum())
    n_11 = int((mask1 & (y == 1)).sum())
    pct_biased = round(100 * n_10 / n, 2)  # disadvantaged (S=1) with negative outcome (Y=0)
    return {
        "subset": subset_name,
        "n": n,
        "n_S0": int(n0),
        "n_S1": int(n1),
        "frac_S1": float(n1 / n),
        "base_rate_S0": round(br0, 4),
        "base_rate_S1": round(br1, 4),
        "base_rate_diff_inherent_SPD": round(base_rate_diff, 4),
        "corr_S_Y": round(corr, 4),
        "cell_S0_Y0": n_00,
        "cell_S0_Y1": n_01,
        "cell_S1_Y0": n_10,
        "cell_S1_Y1": n_11,
        "pct_S0_Y0": round(100 * n_00 / n, 2),
        "pct_S0_Y1": round(100 * n_01 / n, 2),
        "pct_S1_Y0": round(100 * n_10 / n, 2),
        "pct_S1_Y1": round(100 * n_11 / n, 2),
        "pct_biased": pct_biased,
        "pct_unbiased": round(100 - pct_biased, 2),
    }


def main():
    data_path = "dataset"
    train_split = 0.8
    test_split = 0.1
    split_seed = 42
    datasets = ["credit", "german", "bail", "pokec-z", "pokec-n"]

    # Sensitive attribute and label meaning for reporting
    meta = {
        "credit": {"sens": "Age", "sens_note": "Binary or binarized at median if multi-valued", "label": "NoDefaultNextMonth (1=no default)"},
        "german": {"sens": "Gender", "sens_note": "Female=1, Male=0", "label": "GoodCustomer (1=good)"},
        "bail": {"sens": "Race (WHITE=1, non-WHITE=0)", "sens_note": "", "label": "RECID (1=recidivism)"},
        "pokec-z": {"sens": "Region (Zilina)", "sens_note": "Pokec region_job; binary or binarized", "label": "Job/outcome (dataset-specific)"},
        "pokec-n": {"sens": "Region (Nitra)", "sens_note": "Pokec region_job_2; binary or binarized", "label": "Job/outcome (dataset-specific)"},
    }

    all_results = {}
    for data_name in datasets:
        try:
            data, sens_attr = preprocess_data(
                data_path, data_name,
                train_split=train_split, test_split=test_split, split_seed=split_seed,
            )
        except Exception as e:
            all_results[data_name] = {"error": str(e)}
            continue
        y = data.y.numpy()
        sens = sens_attr.numpy()
        train_mask = data.train_mask.numpy()
        test_mask = data.test_mask.numpy()

        full = compute_bias_metrics(y, sens, "full")
        train_only = compute_bias_metrics(y[train_mask], sens[train_mask], "train")
        test_only = compute_bias_metrics(y[test_mask], sens[test_mask], "test")

        all_results[data_name] = {
            "meta": meta.get(data_name, {}),
            "full": full,
            "train": train_only,
            "test": test_only,
        }

    # Print human-readable summary
    print("=" * 70)
    print("Dataset bias analysis (inherent disparity in labels by sensitive group)")
    print("=" * 70)
    for data_name, rec in all_results.items():
        if "error" in rec:
            print(f"\n{data_name}: ERROR — {rec['error']}")
            continue
        m = rec.get("meta", {})
        print(f"\n--- {data_name.upper()} ---")
        print(f"  Sensitive: {m.get('sens', '?')}  |  Label: {m.get('label', '?')}")
        for subset in ["full", "train", "test"]:
            s = rec[subset]
            if s.get("base_rate_S0") is None:
                print(f"  [{subset}] n={s['n']}  n_S0={s['n_S0']}  n_S1={s['n_S1']}  (one group empty)")
                continue
            print(f"  [{subset}] n={s['n']}  n_S0={s['n_S0']}  n_S1={s['n_S1']}  frac_S1={s['frac_S1']:.3f}")
            print(f"       Base rate S=0: {s['base_rate_S0']:.4f}   S=1: {s['base_rate_S1']:.4f}")
            print(f"       Base rate diff (inherent SPD): {s['base_rate_diff_inherent_SPD']:+.4f}   corr(S,Y)= {s['corr_S_Y']:+.4f}")
            if "pct_S0_Y0" in s:
                print(f"       4-cell %: (S=0,Y=0)={s['pct_S0_Y0']}%  (S=0,Y=1)={s['pct_S0_Y1']}%  (S=1,Y=0)={s['pct_S1_Y0']}%  (S=1,Y=1)={s['pct_S1_Y1']}%")
                print(f"       Bias: {s['pct_biased']}% biased (S=1,Y=0), {s['pct_unbiased']}% unbiased")

    # Save JSON
    out_path = PROJECT_ROOT / "outputs" / "dataset_bias_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return all_results


if __name__ == "__main__":
    main()
