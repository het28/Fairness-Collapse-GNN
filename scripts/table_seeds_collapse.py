#!/usr/bin/env python3
"""
Print per-seed tables from results_all_runs.csv with dataset and model fixed.
Marks which seeds collapsed (trivial predictor: all 0 or all 1) vs OK.
Includes only vanilla models: GCN, GAT, GIN (no FAME/PFR variants).

Reads: results/multi_seed/results_all_runs.csv (or --csv path).
Output: one table per (dataset, model); rows = seeds; columns = Acc, SPD, EOD, OAED, TED, Status.
"""

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "results" / "multi_seed" / "results_all_runs.csv"
MODELS_ONLY = ("GCN", "GAT", "GIN")  # vanilla models only, no FAME/PFR variants

# Collapse = model predicts only one class (pred_pos_overall ~0 or ~1)
COLLAPSE_THRESH = 1e-6


def is_collapsed(row: dict) -> bool:
    try:
        p = row.get("diagnostics_pred_pos_overall", "")
        if p == "" or p is None:
            return False
        p = float(p)
        return p <= COLLAPSE_THRESH or p >= (1.0 - COLLAPSE_THRESH)
    except (TypeError, ValueError):
        return False


def safe_float(x, default=None):
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser(description="Per-seed tables with collapse status")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help=f"Path to results_all_runs.csv (default: {DEFAULT_CSV})")
    ap.add_argument("--dataset", default=None, help="Filter to one dataset (default: all)")
    ap.add_argument("--model", default=None, help="Filter to one model (default: all)")
    ap.add_argument("--out", default=None, help="Write text tables to this file (default: print only)")
    ap.add_argument("--out-csv", default=None, help="Write all runs with Status (OK/COLLAPSED) to this CSV file")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.is_file():
        print(f"File not found: {path}")
        return 1

    rows_by_key = {}  # (dataset, model) -> list of row dicts
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            model = row.get("model", "").strip()
            if model not in MODELS_ONLY:
                continue
            d = (row.get("dataset", "").strip(), model)
            if args.dataset and row.get("dataset", "").strip().lower() != args.dataset.lower():
                continue
            if args.model and model != args.model:
                continue
            rows_by_key.setdefault(d, []).append(row)

    if not rows_by_key:
        print("No rows found.")
        return 1

    lines = []
    for (dataset, model), rows in sorted(rows_by_key.items()):
        # Sort by seed
        rows = sorted(rows, key=lambda r: (safe_float(r.get("seed"), -1),))

        lines.append("")
        lines.append("=" * 120)
        lines.append(f"  Dataset: {dataset}   |   Model: {model}")
        lines.append("=" * 120)
        header = (
            f"{'Seed':<6} | {'Acc':<8} {'SPD':<8} {'EOD':<8} {'OAED':<8} {'TED':<8} | {'Status':<10}"
        )
        lines.append(header)
        lines.append("-" * 120)

        for r in rows:
            seed = r.get("seed", "")
            acc = safe_float(r.get("Accuracy"))
            spd = safe_float(r.get("Statistical Parity Difference"))
            eod = safe_float(r.get("Equal Opportunity Difference"))
            oaed = safe_float(r.get("Overall Accuracy Equality Difference"))
            ted = safe_float(r.get("Treatment Equality Difference"))
            status = "COLLAPSED" if is_collapsed(r) else "OK"
            acc_s = f"{acc:.4f}" if acc is not None else " — "
            spd_s = f"{spd:.4f}" if spd is not None else " — "
            eod_s = f"{eod:.4f}" if eod is not None else " — "
            oaed_s = f"{oaed:.4f}" if oaed is not None else " — "
            ted_s = f"{ted:.4f}" if ted is not None else " — "
            lines.append(
                f"{str(seed):<6} | {acc_s:<8} {spd_s:<8} {eod_s:<8} {oaed_s:<8} {ted_s:<8} | {status:<10}"
            )
        lines.append("")

    text = "\n".join(lines)
    print(text)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")

    # Optional: write one row per run with Status column to CSV
    if args.out_csv:
        all_rows = []
        for (dataset, model), rows in sorted(rows_by_key.items()):
            rows_sorted = sorted(rows, key=lambda r: (safe_float(r.get("seed"), -1),))
            for r in rows_sorted:
                status = "COLLAPSED" if is_collapsed(r) else "OK"
                all_rows.append({
                    "dataset": r.get("dataset", ""),
                    "model": r.get("model", ""),
                    "seed": r.get("seed", ""),
                    "Accuracy": r.get("Accuracy", ""),
                    "Statistical Parity Difference": r.get("Statistical Parity Difference", ""),
                    "Equal Opportunity Difference": r.get("Equal Opportunity Difference", ""),
                    "Overall Accuracy Equality Difference": r.get("Overall Accuracy Equality Difference", ""),
                    "Treatment Equality Difference": r.get("Treatment Equality Difference", ""),
                    "diagnostics_pred_pos_overall": r.get("diagnostics_pred_pos_overall", ""),
                    "status": status,
                })
        fieldnames = [
            "dataset", "model", "seed", "Accuracy", "Statistical Parity Difference",
            "Equal Opportunity Difference", "Overall Accuracy Equality Difference",
            "Treatment Equality Difference", "diagnostics_pred_pos_overall", "status",
        ]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"Wrote collapse table CSV: {args.out_csv} ({len(all_rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
