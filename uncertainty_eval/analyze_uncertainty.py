"""
Load multi-seed results, compute CIs and overlap analysis, write summary and comparable tables (CSV + LaTeX).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .confidence_intervals import (
    FAIRNESS_METRIC_KEYS,
    bootstrap_ci,
    summarize_runs,
    overlap_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze uncertainty from multi-seed runs")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="outputs/multi_seed",
        help="Directory containing all_runs.json",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="CI level (e.g. 0.95)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Write summary/overlap here (default: same as results_dir)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        # Assume relative to package parent
        pkg_root = Path(__file__).resolve().parent.parent
        results_dir = pkg_root / results_dir
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_file = results_dir / "all_runs.json"
    if not runs_file.is_file():
        raise FileNotFoundError(f"Not found: {runs_file}. Run run_multi_seed first.")

    with open(runs_file) as f:
        runs = json.load(f)

    summary = summarize_runs(runs, metric_keys=FAIRNESS_METRIC_KEYS, confidence=args.confidence)

    # Serialize summary (convert tuple keys to strings for JSON)
    summary_serializable = {}
    for (data_name, model_label), metrics in summary.items():
        key = f"{data_name}|{model_label}"
        summary_serializable[key] = metrics
    with open(out_dir / "summary_ci.json", "w") as f:
        json.dump(summary_serializable, f, indent=2)

    # Overlap analysis per dataset and metric
    overlap_report = {}
    for (data_name, model_label) in summary:
        if data_name not in overlap_report:
            overlap_report[data_name] = {}
        for metric in FAIRNESS_METRIC_KEYS:
            pairs = overlap_pairs(summary, data_name, metric)
            if pairs:
                overlap_report[data_name][metric] = [f"{a} vs {b}" for a, b in pairs]

    with open(out_dir / "overlap_analysis.json", "w") as f:
        json.dump(overlap_report, f, indent=2)

    # Plain-text summary table (one metric: Statistical Parity Difference)
    metric_short = "Statistical Parity Difference"
    lines = [
        "# Uncertainty summary (mean ± std, 95% CI)",
        f"# Metric: {metric_short}",
        "",
    ]
    for (data_name, model_label), metrics in sorted(summary.items()):
        m = metrics.get(metric_short, {})
        mean = m.get("mean", float("nan"))
        std = m.get("std", float("nan"))
        lo, hi = m.get("ci_low", float("nan")), m.get("ci_high", float("nan"))
        n = m.get("n", 0)
        lines.append(f"{data_name}\t{model_label}\t{mean:.4f}\t{std:.4f}\t[{lo:.4f}, {hi:.4f}]\tn={n}")
    with open(out_dir / "summary_spd.txt", "w") as f:
        f.write("\n".join(lines))

    # --- Comparable table format: long CSV (dataset, model, metric, mean, std, ci_low, ci_high, n) ---
    long_rows = []
    for (data_name, model_label), metrics in sorted(summary.items()):
        for mk in FAIRNESS_METRIC_KEYS:
            m = metrics.get(mk, {})
            long_rows.append({
                "dataset": data_name,
                "model": model_label,
                "metric": mk,
                "mean": f"{m.get('mean', float('nan')):.4f}",
                "std": f"{m.get('std', float('nan')):.4f}",
                "ci_low": f"{m.get('ci_low', float('nan')):.4f}",
                "ci_high": f"{m.get('ci_high', float('nan')):.4f}",
                "n": m.get("n", 0),
            })
    with open(out_dir / "results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "metric", "mean", "std", "ci_low", "ci_high", "n"])
        w.writeheader()
        w.writerows(long_rows)

    # --- Wide summary CSV: one row per (dataset, model), columns = metric_mean, metric_std, metric_mean_pm_std ---
    short_metrics = ["Accuracy", "Statistical Parity Difference", "Equal Opportunity Difference", "Overall Accuracy Equality Difference", "Treatment Equality Difference"]
    wide_headers = ["dataset", "model", "n_seeds"]
    for mk in short_metrics:
        wide_headers.append(f"{mk}_mean")
        wide_headers.append(f"{mk}_std")
        wide_headers.append(f"{mk}")  # "mean ± std" string
    wide_rows = []
    for (data_name, model_label), metrics in sorted(summary.items()):
        row = {"dataset": data_name, "model": model_label, "n_seeds": next(iter(metrics.values()), {}).get("n", 0)}
        for mk in short_metrics:
            m = metrics.get(mk, {})
            mean, std = m.get("mean", float("nan")), m.get("std", float("nan"))
            row[f"{mk}_mean"] = f"{mean:.4f}"
            row[f"{mk}_std"] = f"{std:.4f}"
            row[mk] = f"{mean:.4f} ± {std:.4f}"
        wide_rows.append(row)
    with open(out_dir / "results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=wide_headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(wide_rows)

    # --- LaTeX table snippet (paper-ready) ---
    latex_lines = [
        "% Main results: mean ± std over seeds. Paste into paper.",
        "\\begin{tabular}{llcccc}",
        "\\hline",
        "Dataset & Model & Acc & SPD & EOD & OAED \\\\",
        "\\hline",
    ]
    for (data_name, model_label), metrics in sorted(summary.items()):
        acc = metrics.get("Accuracy", {})
        spd = metrics.get("Statistical Parity Difference", {})
        eod = metrics.get("Equal Opportunity Difference", {})
        oaed = metrics.get("Overall Accuracy Equality Difference", {})
        acc_s = f"{acc.get('mean', 0):.2f} $\\pm$ {acc.get('std', 0):.2f}"
        spd_s = f"{spd.get('mean', 0):.2f} $\\pm$ {spd.get('std', 0):.2f}"
        eod_s = f"{eod.get('mean', 0):.2f} $\\pm$ {eod.get('std', 0):.2f}"
        oaed_s = f"{oaed.get('mean', 0):.2f} $\\pm$ {oaed.get('std', 0):.2f}"
        latex_lines.append(f"{data_name} & {model_label} & {acc_s} & {spd_s} & {eod_s} & {oaed_s} \\\\")
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    with open(out_dir / "results_table_latex.txt", "w") as f:
        f.write("\n".join(latex_lines))

    # --- Diagnostics table (pred rate, TPR/FPR by group, test-split counts) for reviewers ---
    diag_rate_keys = [
        "diagnostics_pred_pos_overall", "diagnostics_pred_pos_S0", "diagnostics_pred_pos_S1",
        "diagnostics_TPR_S0", "diagnostics_TPR_S1", "diagnostics_FPR_S0", "diagnostics_FPR_S1",
    ]
    diag_count_keys = [
        "diagnostics_n_test", "diagnostics_n_S0", "diagnostics_n_S1",
        "diagnostics_n_pos_S0", "diagnostics_n_pos_S1",
        "diagnostics_TPR_denom_S0", "diagnostics_TPR_denom_S1",
        "diagnostics_FPR_denom_S0", "diagnostics_FPR_denom_S1",
    ]
    groups = {}
    for r in runs:
        if "error" in r:
            continue
        key = (r.get("_data_name"), r.get("_model"), r.get("_fame"), r.get("_fairness_algo"))
        model_label = key[1] + ("+FAME" if key[2] else "") + ("+PFR" if key[3] == "pfr" else "")
        row_key = (key[0], model_label)
        if row_key not in groups:
            groups[row_key] = []
        groups[row_key].append(r)
    diag_rows = []
    for (data_name, model_label) in sorted(groups.keys()):
        group = groups[(data_name, model_label)]
        row = {"dataset": data_name, "model": model_label, "n_seeds": len(group)}
        for k in diag_count_keys:
            vals = [r[k] for r in group if k in r and r[k] is not None]
            row[k] = int(round(sum(vals) / len(vals))) if vals else ""
        for k in diag_rate_keys:
            vals = [float(r[k]) for r in group if k in r and r[k] is not None and (isinstance(r[k], (int, float)) and (r[k] == r[k]))]
            if vals:
                row[f"{k}_mean"] = f"{np.mean(vals):.4f}"
                row[f"{k}_std"] = f"{np.std(vals, ddof=1):.4f}" if len(vals) > 1 else "0.0000"
            else:
                row[f"{k}_mean"] = row[f"{k}_std"] = ""
        diag_rows.append(row)
    diag_headers = ["dataset", "model", "n_seeds"] + diag_count_keys
    for k in diag_rate_keys:
        diag_headers.append(f"{k}_mean")
        diag_headers.append(f"{k}_std")
    with open(out_dir / "results_diagnostics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=diag_headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(diag_rows)

    # --- Full paper-style table: one row per (dataset, model), all metrics + CIs + diagnostics (Excel-ready) ---
    full_headers = [
        "dataset", "model", "n_seeds",
        "Accuracy_mean", "Accuracy_std", "Accuracy", "Accuracy_ci_low", "Accuracy_ci_high",
        "SPD_mean", "SPD_std", "SPD", "SPD_ci_low", "SPD_ci_high",
        "EOD_mean", "EOD_std", "EOD", "EOD_ci_low", "EOD_ci_high",
        "OAED_mean", "OAED_std", "OAED", "OAED_ci_low", "OAED_ci_high",
        "TED_mean", "TED_std", "TED", "TED_ci_low", "TED_ci_high",
    ]
    metric_to_short = {
        "Accuracy": "Accuracy",
        "Statistical Parity Difference": "SPD",
        "Equal Opportunity Difference": "EOD",
        "Overall Accuracy Equality Difference": "OAED",
        "Treatment Equality Difference": "TED",
    }
    full_headers += diag_count_keys
    for k in diag_rate_keys:
        full_headers.append(f"{k}_mean")
        full_headers.append(f"{k}_std")
    full_rows = []
    for (data_name, model_label) in sorted(summary.keys()):
        metrics = summary[(data_name, model_label)]
        diag_row = next((r for r in diag_rows if r["dataset"] == data_name and r["model"] == model_label), {})
        row = {"dataset": data_name, "model": model_label, "n_seeds": next(iter(metrics.values()), {}).get("n", 0)}
        for mk in short_metrics:
            m = metrics.get(mk, {})
            mean, std = m.get("mean", float("nan")), m.get("std", float("nan"))
            ci_lo, ci_hi = m.get("ci_low", float("nan")), m.get("ci_high", float("nan"))
            short = metric_to_short.get(mk, mk)
            row[f"{short}_mean"] = f"{mean:.4f}"
            row[f"{short}_std"] = f"{std:.4f}"
            row[short] = f"{mean:.4f} ± {std:.4f}"
            row[f"{short}_ci_low"] = f"{ci_lo:.4f}"
            row[f"{short}_ci_high"] = f"{ci_hi:.4f}"
        for k in diag_count_keys:
            row[k] = diag_row.get(k, "")
        for k in diag_rate_keys:
            row[f"{k}_mean"] = diag_row.get(f"{k}_mean", "")
            row[f"{k}_std"] = diag_row.get(f"{k}_std", "")
        full_rows.append(row)
    with open(out_dir / "results_full_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=full_headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(full_rows)

    # --- All runs (one row per experiment = per seed): so you see every seed, not just the summary ---
    metric_cols = ["Accuracy", "Statistical Parity Difference", "Equal Opportunity Difference", "Overall Accuracy Equality Difference", "Treatment Equality Difference"]
    diag_cols = [
        "diagnostics_n_test", "diagnostics_n_S0", "diagnostics_n_S1",
        "diagnostics_n_pos_S0", "diagnostics_n_pos_S1",
        "diagnostics_TPR_denom_S0", "diagnostics_TPR_denom_S1",
        "diagnostics_FPR_denom_S0", "diagnostics_FPR_denom_S1",
        "diagnostics_pred_pos_overall", "diagnostics_pred_pos_S0", "diagnostics_pred_pos_S1",
        "diagnostics_TPR_S0", "diagnostics_TPR_S1", "diagnostics_FPR_S0", "diagnostics_FPR_S1",
    ]
    # Include split info when present (split_seed experiment)
    has_split_info = any(r.get("_split_seed") is not None for r in runs if r.get("error") is None)
    all_runs_headers = ["dataset", "model", "seed"] + (["split_seed", "train_split", "val_split", "test_split"] if has_split_info else []) + metric_cols + diag_cols
    all_runs_rows = []
    for r in runs:
        if r.get("error") is not None:
            continue
        data_name = r.get("_data_name", "")
        model_base = r.get("_model", "")
        fame = r.get("_fame", False)
        fa = r.get("_fairness_algo")
        model_label = model_base + ("+FAME" if fame else "") + ("+PFR" if fa == "pfr" else "")
        seed = r.get("_seed", "")
        row = {"dataset": data_name, "model": model_label, "seed": seed}
        if has_split_info:
            train_s = r.get("_train_split")
            test_s = r.get("_test_split")
            val_s = (1.0 - train_s - test_s) if train_s is not None and test_s is not None else None
            row["split_seed"] = r.get("_split_seed", "")
            row["train_split"] = f"{train_s:.2f}" if isinstance(train_s, (int, float)) else train_s
            row["val_split"] = f"{val_s:.2f}" if isinstance(val_s, (int, float)) else val_s
            row["test_split"] = f"{test_s:.2f}" if isinstance(test_s, (int, float)) else test_s
        for mk in metric_cols:
            v = r.get(mk)
            if v is None:
                row[mk] = ""
            elif isinstance(v, float) and (v != v or abs(v) == float("inf")):
                row[mk] = "" if v != v else ("Inf" if v > 0 else "-Inf")
            else:
                row[mk] = f"{v:.4f}" if isinstance(v, (int, float)) else v
        for k in diag_cols:
            v = r.get(k)
            if v is None:
                row[k] = ""
            elif isinstance(v, float) and (v != v or abs(v) == float("inf")):
                row[k] = "" if v != v else ("Inf" if v > 0 else "-Inf")
            elif isinstance(v, int):
                row[k] = v
            elif isinstance(v, float):
                row[k] = f"{v:.4f}"
            else:
                row[k] = str(v) if v is not None else ""
        all_runs_rows.append(row)
    with open(out_dir / "results_all_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_runs_headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_runs_rows)

    # --- Which metric is noisiest? (CI width and coefficient of variation per dataset, model, metric) ---
    short_metrics = ["Accuracy", "Statistical Parity Difference", "Equal Opportunity Difference", "Overall Accuracy Equality Difference", "Treatment Equality Difference"]
    noise_rows = []
    for (data_name, model_label) in sorted(summary.keys()):
        metrics = summary[(data_name, model_label)]
        for mk in short_metrics:
            m = metrics.get(mk, {})
            mean = m.get("mean", float("nan"))
            std = m.get("std", float("nan"))
            ci_lo, ci_hi = m.get("ci_low", float("nan")), m.get("ci_high", float("nan"))
            ci_width = (ci_hi - ci_lo) if not (np.isnan(ci_lo) or np.isnan(ci_hi)) else float("nan")
            cv = (std / mean) if mean and not np.isnan(mean) and abs(mean) > 1e-9 else float("nan")
            noise_rows.append({
                "dataset": data_name,
                "model": model_label,
                "metric": mk,
                "mean": f"{mean:.4f}",
                "std": f"{std:.4f}",
                "ci_low": f"{ci_lo:.4f}",
                "ci_high": f"{ci_hi:.4f}",
                "ci_width": f"{ci_width:.4f}" if not np.isnan(ci_width) else "",
                "cv": f"{cv:.4f}" if not np.isnan(cv) else "",
            })
    with open(out_dir / "results_metric_noise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "metric", "mean", "std", "ci_low", "ci_high", "ci_width", "cv"])
        w.writeheader()
        w.writerows(noise_rows)

    # --- Paired-seed analysis: same seed for two models -> difference in metric; mean and 95% CI of difference ---
    # Group runs by (dataset, seed) -> { model_label -> run }
    by_dataset_seed = {}
    for r in runs:
        if r.get("error") is not None:
            continue
        data_name = r.get("_data_name", "")
        model_base = r.get("_model", "")
        fame = r.get("_fame", False)
        fa = r.get("_fairness_algo")
        model_label = model_base + ("+FAME" if fame else "") + ("+PFR" if fa == "pfr" else "")
        seed = r.get("_seed", None)
        if seed is None:
            continue
        key = (data_name, seed)
        by_dataset_seed.setdefault(key, {})[model_label] = r
    # All (dataset, model_a, model_b) pairs (model_a < model_b for consistency)
    model_labels = sorted({m for _, d in by_dataset_seed.items() for m in d.keys()})
    paired_rows = []
    for (data_name, model_a, model_b) in [(d, a, b) for d in set(k[0] for k in by_dataset_seed) for a in model_labels for b in model_labels if a < b]:
        diffs = {mk: [] for mk in short_metrics}
        for (ds, seed), models in by_dataset_seed.items():
            if ds != data_name or model_a not in models or model_b not in models:
                continue
            ra, rb = models[model_a], models[model_b]
            for mk in short_metrics:
                va, vb = ra.get(mk), rb.get(mk)
                if va is None or vb is None or (isinstance(va, float) and (np.isnan(va) or np.isinf(abs(va)))) or (isinstance(vb, float) and (np.isnan(vb) or np.isinf(abs(vb)))):
                    continue
                diffs[mk].append(float(vb) - float(va))
        for mk in short_metrics:
            vals = np.array(diffs[mk])
            if len(vals) < 2:
                continue
            diff_mean = float(np.mean(vals))
            diff_std = float(np.std(vals, ddof=1))
            diff_ci_lo, diff_ci_hi = bootstrap_ci(vals, confidence=args.confidence)
            paired_rows.append({
                "dataset": data_name,
                "model_a": model_a,
                "model_b": model_b,
                "metric": mk,
                "diff_mean": f"{diff_mean:.4f}",
                "diff_std": f"{diff_std:.4f}",
                "diff_ci_low": f"{diff_ci_lo:.4f}",
                "diff_ci_high": f"{diff_ci_hi:.4f}",
                "n_seeds": len(vals),
            })
    with open(out_dir / "results_paired_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model_a", "model_b", "metric", "diff_mean", "diff_std", "diff_ci_low", "diff_ci_high", "n_seeds"])
        w.writeheader()
        w.writerows(paired_rows)

    print(f"Summary and CIs -> {out_dir / 'summary_ci.json'}")
    print(f"Overlap analysis -> {out_dir / 'overlap_analysis.json'}")
    print(f"SPD table -> {out_dir / 'summary_spd.txt'}")
    print(f"Comparable tables -> {out_dir / 'results_table.csv'}, {out_dir / 'results_summary.csv'}, {out_dir / 'results_table_latex.txt'}")
    print(f"Diagnostics (pred rate, TPR/FPR, counts) -> {out_dir / 'results_diagnostics.csv'}")
    print(f"Full paper table (all metrics + CIs + diagnostics, Excel-ready) -> {out_dir / 'results_full_table.csv'}")
    print(f"All runs (one row per seed, every experiment) -> {out_dir / 'results_all_runs.csv'}")
    print(f"Which metric is noisiest (CI width, CV) -> {out_dir / 'results_metric_noise.csv'}")
    print(f"Paired-seed model comparison (diff mean and 95% CI) -> {out_dir / 'results_paired_seed.csv'}")


if __name__ == "__main__":
    main()
