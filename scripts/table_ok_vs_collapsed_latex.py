#!/usr/bin/env python3
"""
Build the run-type decomposition table (OK vs collapsed) from collapse_table.csv
and output LaTeX for the paper.

Reads: results/multi_seed/collapse_table.csv (or --csv path).
Output: LaTeX table with (1) collapse rate and dominant mode (all-1 vs all-0),
(2) near-collapse rate, (3) mean Acc/Disp/Ineq on OK runs, (4) mean Acc/Disp/Ineq on collapsed runs.
Disp = SPD (%), Ineq = EOD (%).
"""

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "results" / "multi_seed" / "collapse_table.csv"
DATASETS_ORDER = ("credit", "german", "bail")
MODELS_ORDER = ("GAT", "GCN", "GIN")

# Collapse: pred_pos <= LO or >= HI
COLLAPSE_LO = 1e-6
COLLAPSE_HI = 1.0 - 1e-6
# Near-collapse: not collapsed but pred_pos in (0.05, 0.1] or [0.9, 0.95)
NEAR_LO_LOW, NEAR_LO_HI = 0.05, 0.1
NEAR_HI_LOW, NEAR_HI_HI = 0.9, 0.95


def safe_float(x, default=None):
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def is_collapsed(pred_pos: float) -> bool:
    if pred_pos is None:
        return False
    return pred_pos <= COLLAPSE_LO or pred_pos >= COLLAPSE_HI


def is_near_collapse(pred_pos: float, status: str) -> bool:
    """Not collapsed but pred_pos close to 0 or 1."""
    if pred_pos is None or status == "COLLAPSED":
        return False
    return (NEAR_LO_LOW < pred_pos <= NEAR_LO_HI) or (NEAR_HI_LOW <= pred_pos < NEAR_HI_HI)


def main():
    ap = argparse.ArgumentParser(description="OK vs collapsed profile table → LaTeX")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to collapse_table.csv")
    ap.add_argument("--out", default=None, help="Write LaTeX to this file (default: print)")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.is_file():
        print(f"File not found: {path}")
        return 1

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("model", "").strip() not in MODELS_ORDER:
                continue
            rows.append(row)

    # Group by (dataset, model)
    by_key = {}
    for r in rows:
        d = r.get("dataset", "").strip().lower()
        m = r.get("model", "").strip()
        key = (d, m)
        by_key.setdefault(key, []).append(r)

    def pct(n, total):
        return round(100 * n / total, 0) if total else 0

    def mean_acc_disp_ineq(run_rows):
        if not run_rows:
            return None, None, None
        accs = [safe_float(r.get("Accuracy")) for r in run_rows]
        spds = [safe_float(r.get("Statistical Parity Difference")) for r in run_rows]
        eods = [safe_float(r.get("Equal Opportunity Difference")) for r in run_rows]
        accs = [a for a in accs if a is not None]
        spds = [s for s in spds if s is not None]
        eods = [e for e in eods if e is not None]
        n = len(run_rows)
        mean_acc = sum(accs) / n if accs else None
        mean_spd_pct = 100 * sum(spds) / n if spds else None
        mean_eod_pct = 100 * sum(eods) / n if eods else None
        return mean_acc, mean_spd_pct, mean_eod_pct

    # Build profile per (dataset, model)
    data = {}
    for key in by_key:
        run_list = by_key[key]
        total = len(run_list)
        collapsed = [r for r in run_list if r.get("status") == "COLLAPSED"]
        ok_runs = [r for r in run_list if r.get("status") == "OK"]
        pred_pos_col = "diagnostics_pred_pos_overall"

        n_coll = len(collapsed)
        pct_coll = pct(n_coll, total)
        all1 = sum(1 for r in collapsed if safe_float(r.get(pred_pos_col), -1) >= COLLAPSE_HI)
        all0 = sum(1 for r in collapsed if safe_float(r.get(pred_pos_col), -1) <= COLLAPSE_LO)
        if n_coll == 0:
            mode_str = "--"
        else:
            p1 = pct(all1, n_coll)
            p0 = pct(all0, n_coll)
            if all1 == n_coll:
                mode_str = f"all-1 100%"
            elif all0 == n_coll:
                mode_str = f"all-0 100%"
            else:
                mode_str = f"all-1 {int(p1)}% / all-0 {int(p0)}%"

        near = sum(1 for r in run_list if is_near_collapse(safe_float(r.get(pred_pos_col)), r.get("status", "")))
        pct_near = pct(near, total)

        acc_ok, disp_ok, ineq_ok = mean_acc_disp_ineq(ok_runs)
        acc_col, disp_col, ineq_col = mean_acc_disp_ineq(collapsed)

        def fmt3(acc, disp, ineq):
            if acc is None:
                return "--"
            d = f"{disp:.1f}" if disp is not None else "?"
            i = f"{ineq:.1f}" if ineq is not None else "?"
            return f"{acc:.2f} / {d} / {i}"

        data[key] = {
            "collapse_rate": int(pct_coll),
            "mode": mode_str,
            "near_rate": int(pct_near),
            "ok_str": fmt3(acc_ok, disp_ok, ineq_ok) if ok_runs else "--",
            "coll_str": fmt3(acc_col, disp_col, ineq_col) if collapsed else "--",
        }

    # LaTeX
    lines = [
        r"\begin{table*}[t]",
        r"  \caption{Run-type decomposition per (Dataset, Model). Each dataset block reports:",
        r"  (1) collapse rate and dominant collapse mode (all-1 vs all-0),",
        r"  (2) near-collapse rate,",
        r"  (3) mean metrics on OK runs (Acc / SPD\% / EOD\%),",
        r"  (4) mean metrics on collapsed runs (Acc / SPD\% / EOD\%).}",
        r"  \label{tab:ok_vs_collapsed_profile}",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{@{}llccc@{}}",
        r"    \toprule",
        r"    Dataset & Factor & GAT & GCN & GIN \\",
        r"    \midrule",
    ]

    for dataset in DATASETS_ORDER:
        d = data.get((dataset, "GAT"), {}), data.get((dataset, "GCN"), {}), data.get((dataset, "GIN"), {})
        for i, factor in enumerate([
            "Collapse rate (mode)",
            "Near-collapse rate",
            "OK runs: Acc / Disp / Ineq",
            "Collapsed: Acc / Disp / Ineq",
        ]):
            if i == 0:
                c1 = f"{d[0].get('collapse_rate', 0)}\\% ({d[0].get('mode', '--')})"
                c2 = f"{d[1].get('collapse_rate', 0)}\\% ({d[1].get('mode', '--')})"
                c3 = f"{d[2].get('collapse_rate', 0)}\\% ({d[2].get('mode', '--')})"
            elif i == 1:
                c1 = f"{d[0].get('near_rate', 0)}\\%"
                c2 = f"{d[1].get('near_rate', 0)}\\%"
                c3 = f"{d[2].get('near_rate', 0)}\\%"
            elif i == 2:
                c1 = d[0].get("ok_str", "--")
                c2 = d[1].get("ok_str", "--")
                c3 = d[2].get("ok_str", "--")
            else:
                c1 = d[0].get("coll_str", "--")
                c2 = d[1].get("coll_str", "--")
                c3 = d[2].get("coll_str", "--")

            first_col = f"    \\multirow{{4}}{{*}}{{{dataset.capitalize()}}}" if i == 0 else "    "
            lines.append(f"{first_col} & {factor} & {c1} & {c2} & {c3} \\\\")
        lines.append(r"    \midrule")

    lines = lines[:-1]  # drop last midrule
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)
    print(latex)
    if args.out:
        Path(args.out).write_text(latex, encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
