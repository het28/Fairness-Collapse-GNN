#!/usr/bin/env python3
"""
Compare two runs side-by-side (one collapsed, one OK) with everything constant except seed.
Print each and every thing so you can compare both runs; use Cursor's debugger from this app.

Usage:
  python scripts/compare_two_seeds_debug.py --dataset bail --model GAT --seed-collapsed 0 --seed-ok 2
  python scripts/compare_two_seeds_debug.py --dataset bail --model GAT --print-all   # print every epoch
  python scripts/compare_two_seeds_debug.py --dataset bail --model GAT --save-logs results/debug_logs  # two files to compare side-by-side in Cursor
  python scripts/compare_two_seeds_debug.py --no-run   # only print how to use Cursor's debugger

Use Cursor's debugger: Run and Debug (sidebar) → set breakpoints in run_one.py → launch
"Run one seed (collapsed)" or "Run one seed (OK)" to step through and print each and every thing.
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_config():
    import yaml
    cfg_path = PROJECT_ROOT / "config" / "experiments.yaml"
    if not cfg_path.is_file():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser(
        description="Compare two seeds side-by-side (collapsed vs OK) with per-epoch metrics"
    )
    ap.add_argument("--dataset", default="bail", help="Dataset (default: bail)")
    ap.add_argument("--model", default="GAT", choices=["GCN", "GAT", "GIN"], help="Model (default: GAT)")
    ap.add_argument(
        "--seed-collapsed",
        type=int,
        default=0,
        help="Seed that collapses (trivial predictor) (default: 0)",
    )
    ap.add_argument(
        "--seed-ok",
        type=int,
        default=2,
        help="Seed that does not collapse (default: 2)",
    )
    ap.add_argument("--epochs", type=int, default=None, help="Epochs (default: from config, else 50)")
    ap.add_argument("--out", default=None, help="Write side-by-side CSV here")
    ap.add_argument("--print-all", action="store_true", help="Print every epoch (and pass debug_print to run_one so each run prints every epoch)")
    ap.add_argument("--save-logs", metavar="DIR", default=None, help="Write two log files (run_collapsed.txt, run_ok.txt) to DIR for side-by-side comparison in Cursor")
    ap.add_argument("--no-run", action="store_true", help="Only print debugger instructions")
    args = ap.parse_args()

    if args.no_run:
        print(_debugger_instructions(args.dataset, args.model, args.seed_collapsed, args.seed_ok))
        return 0

    sys.path.insert(0, str(PROJECT_ROOT))
    from uncertainty_eval.run_one import run_one

    config = _load_config()
    epochs = args.epochs or config.get("epochs", 50)
    data_path = config.get("data_path", "dataset")
    layers = config.get("layers", 4)
    hidden = config.get("hidden", 32)
    dropout = config.get("dropout", 0.5)
    lr = config.get("lr", 0.01)

    debug_print = getattr(args, "print_all", False)
    print(f"Running seed {args.seed_collapsed} (collapsed) ...")
    result_collapsed = run_one(
        data_path=data_path,
        data_name=args.dataset,
        model=args.model,
        fame=False,
        seed=args.seed_collapsed,
        split_seed=42,
        train_split=0.8,
        test_split=0.1,
        epochs=epochs,
        layers=layers,
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        record_per_epoch=True,
        debug_print_every_epoch=debug_print,
    )
    pe_c = result_collapsed.get("_per_epoch") or []
    print(f"  Final: Acc={result_collapsed.get('Accuracy')} pred_pos={pe_c[-1].get('pred_pos_overall') if pe_c else '?'}")

    print(f"Running seed {args.seed_ok} (OK) ...")
    result_ok = run_one(
        data_path=data_path,
        data_name=args.dataset,
        model=args.model,
        fame=False,
        seed=args.seed_ok,
        split_seed=42,
        train_split=0.8,
        test_split=0.1,
        epochs=epochs,
        layers=layers,
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        record_per_epoch=True,
        debug_print_every_epoch=debug_print,
    )
    pe_ok = result_ok.get("_per_epoch") or []
    print(f"  Final: Acc={result_ok.get('Accuracy')} pred_pos={pe_ok[-1].get('pred_pos_overall') if pe_ok else '?'}")

    # Side-by-side table
    n = max(len(pe_c), len(pe_ok))
    rows = []
    for i in range(n):
        c = pe_c[i] if i < len(pe_c) else {}
        o = pe_ok[i] if i < len(pe_ok) else {}
        rows.append({
            "epoch": i,
            "loss_collapsed": c.get("loss"),
            "loss_ok": o.get("loss"),
            "val_loss_collapsed": c.get("val_loss"),
            "val_loss_ok": o.get("val_loss"),
            "pred_pos_collapsed": c.get("pred_pos_overall"),
            "pred_pos_ok": o.get("pred_pos_overall"),
            "acc_collapsed": c.get("Accuracy"),
            "acc_ok": o.get("Accuracy"),
            "SPD_collapsed": c.get("SPD"),
            "SPD_ok": o.get("SPD"),
            "EOD_collapsed": c.get("EOD"),
            "EOD_ok": o.get("EOD"),
        })

    # Print table
    print()
    print("=" * 120)
    print(f"  Side-by-side: {args.dataset} {args.model}  |  seed_collapsed={args.seed_collapsed}  vs  seed_ok={args.seed_ok}")
    print("=" * 120)
    header = (
        f"{'Epoch':<6} | "
        f"{'loss_c':<10} {'loss_ok':<10} | "
        f"{'val_c':<10} {'val_ok':<10} | "
        f"{'pred_pos_c':<10} {'pred_pos_ok':<10} | "
        f"{'acc_c':<8} {'acc_ok':<8} | "
        f"{'SPD_c':<8} {'SPD_ok':<8} | "
        f"{'EOD_c':<8} {'EOD_ok':<8}"
    )
    def _f(x):
        if x is None:
            return " — "
        try:
            return f"{float(x):.4f}"
        except (TypeError, ValueError):
            return str(x)

    print(header)
    print("-" * 120)
    print_all_epochs = getattr(args, "print_all", False)
    for r in rows:
        if print_all_epochs or r["epoch"] % 5 == 0 or r["epoch"] < 3 or r["epoch"] >= n - 2:
            print(
                f"{r['epoch']:<6} | "
                f"{_f(r['loss_collapsed']):<10} {_f(r['loss_ok']):<10} | "
                f"{_f(r['val_loss_collapsed']):<10} {_f(r['val_loss_ok']):<10} | "
                f"{_f(r['pred_pos_collapsed']):<10} {_f(r['pred_pos_ok']):<10} | "
                f"{_f(r['acc_collapsed']):<8} {_f(r['acc_ok']):<8} | "
                f"{_f(r['SPD_collapsed']):<8} {_f(r['SPD_ok']):<8} | "
                f"{_f(r['EOD_collapsed']):<8} {_f(r['EOD_ok']):<8}"
            )
    if not print_all_epochs and n > 12:
        print("  ...")
    print()
    print(_debugger_instructions(args.dataset, args.model, args.seed_collapsed, args.seed_ok))

    if args.out:
        out_path = Path(args.out)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote full per-epoch comparison to {out_path}")

    if getattr(args, "save_logs", None):
        log_dir = Path(args.save_logs)
        log_dir.mkdir(parents=True, exist_ok=True)
        for label, pe_list, seed in [
            ("collapsed", pe_c, args.seed_collapsed),
            ("ok", pe_ok, args.seed_ok),
        ]:
            path = log_dir / f"run_seed_{seed}_{label}.txt"
            with open(path, "w") as f:
                f.write(f"dataset={args.dataset} model={args.model} seed={seed} ({label})\n")
                f.write("epoch\tloss\tval_loss\tpred_pos\tacc\tSPD\tEOD\n")
                for e in pe_list:
                    f.write(
                        f"{e.get('epoch')}\t{e.get('loss')}\t{e.get('val_loss')}\t"
                        f"{e.get('pred_pos_overall')}\t{e.get('Accuracy')}\t{e.get('SPD')}\t{e.get('EOD')}\n"
                    )
            print(f"Wrote {path} — open both log files in Cursor side-by-side to compare.")
    return 0


def _debugger_instructions(dataset: str, model: str, seed_collapsed: int, seed_ok: int) -> str:
    return f"""
To debug with a proper debugger (Cursor / VS Code):
  1. Open uncertainty_eval/run_one.py and set breakpoints, e.g.:
     - After set_seed(seed) (line ~98)
     - Inside the record_per_epoch loop: after loss.backward() or optimizer.step()
     - In fairness_test_only.test_set_diagnostics (to inspect pred_pos when it goes to 0 or 1)
  2. Use Run and Debug (Ctrl+Shift+D / Cmd+Shift+D) and choose:
     "Run one seed (collapsed)" — runs with --seed {seed_collapsed}
     "Run one seed (OK)"        — runs with --seed {seed_ok}
  3. Compare: step through both runs and watch loss, gradients, and pred_pos_overall each epoch.
     Collapse usually shows as pred_pos_overall → 0 or 1 early; the OK run keeps pred_pos in (0,1).
"""
