"""
Run multi-seed experiments: for each (dataset, model), run N seeds and save metrics.
Reads config from config/experiments.yaml or --config.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import yaml

from .run_one import run_one, _gnns_fame_root


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_gnns_fame_root(config: dict, project_root: Path) -> Optional[str]:
    root = config.get("gnns_fame_root")
    if root:
        return os.path.abspath(root)
    # Auto-detect next to project root
    for folder in ("GNNs-FAME-main", "GNNs-FAME"):
        candidate = project_root / folder
        if candidate.is_dir():
            return str(candidate)
    try:
        return _gnns_fame_root()
    except FileNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed fairness experiments")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiments.yaml (default: config/experiments.yaml next to package)",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None,
        help="Override n_seeds from config (e.g. 5 for a quick test)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print experiment grid without running",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Override config datasets (e.g. --datasets credit german)",
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=None,
        help="Start of seed range (use with --seed_end). E.g. 31 for seeds 31..50.",
    )
    parser.add_argument(
        "--seed_end",
        type=int,
        default=None,
        help="End of seed range (inclusive). E.g. 50 for seeds 31..50.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Explicit seed list (e.g. 0 5 10 15 20 25 30 35 40 45 50). Overrides n_seeds and seed_start/seed_end.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing all_runs.json (load, run only requested seeds, merge, save). Use with --seed_start/--seed_end.",
    )
    parser.add_argument(
        "--split_seeds",
        type=int,
        nargs="*",
        default=None,
        help="List of split seeds (e.g. 0 1 2 ... 9). When set, each run uses a different train/val/test split; combines with train seeds to study split-induced variance.",
    )
    parser.add_argument(
        "--split_configs",
        type=str,
        nargs="*",
        default=None,
        help="List of 'train_split,test_split' (e.g. '0.8,0.1' '0.7,0.15' '0.9,0.05'). When set, each run uses a different split size; combines with train seeds.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Override config models: only run these model names (e.g. GIN). Use with --append to add new model runs.",
    )
    parser.add_argument(
        "--vanilla_only",
        action="store_true",
        help="Use only vanilla GCN and GAT (no FAME) for 'break every metric' experiments.",
    )
    parser.add_argument(
        "--fairness_algo",
        type=str,
        default=None,
        help="In-processing fairness: e.g. 'pfr'. When set, all runs use this fairness_algo (use with --models GCN GAT GIN for PFR runs).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Override results directory. If not set, split-seed runs go to results/split_seed, split-size to results/split_size, baseline to config results_dir.",
    )
    parser.add_argument(
        "--stratify_split",
        action="store_true",
        help="Stratify train/val/test by (sensitive, label) so each split has same S,Y composition. Results go to results/stratified_split/ (separate from multi_seed/split_seed).",
    )
    parser.add_argument(
        "--option_a",
        action="store_true",
        help="Option A: stability regularization (Huang–Vishnoi style). Results go to results/option_a_stability/.",
    )
    parser.add_argument(
        "--option_b",
        action="store_true",
        help="Option B: NIFTY-style augmentation + similarity. Results go to results/option_b_nifty/.",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Use GNN with residual connections (anti-oversmoothing). Results go to results/residual_gnn/. Seeds default to 1,5,10,...,50.",
    )
    parser.add_argument(
        "--stability_lambda",
        type=float,
        default=0.5,
        help="Weight for stability loss when --option_a (default 0.5).",
    )
    parser.add_argument(
        "--nifty_sim_coeff",
        type=float,
        default=0.5,
        help="Weight for NIFTY similarity loss when --option_b (default 0.5).",
    )
    parser.add_argument(
        "--drop_edge_rate",
        type=float,
        default=0.1,
        help="Edge drop rate for NIFTY-style augmentation when --option_b (default 0.1).",
    )
    parser.add_argument(
        "--drop_feature_rate",
        type=float,
        default=0.1,
        help="Feature drop rate for NIFTY-style augmentation when --option_b (default 0.1).",
    )
    parser.add_argument(
        "--pair_norm",
        action="store_true",
        help="Use PairNorm after each conv (GCN/GAT/GIN). Reduces oversmoothing.",
    )
    parser.add_argument(
        "--gat_balanced_init",
        action="store_true",
        help="Use balanced initialization for GAT only (Mustafa et al., NeurIPS 2023).",
    )
    parser.add_argument(
        "--anti_oversmooth_drop_edge",
        type=float,
        default=0.0,
        help="Drop this fraction of edges each epoch during training (DropEdge). Use for GAT/GIN anti-collapse (default 0).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="Override GNN depth from config. Use 2 for anti-collapse when GAT/GIN still collapse with 4 layers (default: from config).",
    )
    args = parser.parse_args()

    # Resolve config path
    if args.config and os.path.isabs(args.config):
        config_path = Path(args.config)
    else:
        pkg_root = Path(__file__).resolve().parent.parent
        config_path = pkg_root / "config" / "experiments.yaml"
        if args.config:
            config_path = pkg_root / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(str(config_path))
    project_root = config_path.parent.parent
    gnns_fame_root = resolve_gnns_fame_root(config, project_root)
    if not gnns_fame_root:
        raise FileNotFoundError(
            "GNNs-FAME repo not found. Put GNNs-FAME-main/ in project root or set gnns_fame_root in config."
        )

    n_seeds = args.n_seeds if args.n_seeds is not None else config.get("n_seeds", 30)
    datasets = args.datasets if args.datasets is not None else config.get("datasets", ["credit", "german"])
    models = config.get("models", [{"name": "GCN", "fame": False}])
    if args.models is not None:
        model_names = set(args.models)
        models = [m for m in models if m.get("name") in model_names]
        if not models:
            raise ValueError(f"--models {args.models} matched no config models. Config has: {[m.get('name') for m in config.get('models', [])]}")
    if args.vanilla_only:
        models = [m for m in models if not m.get("fame", False)]
    split_seeds = args.split_seeds if args.split_seeds is not None else config.get("split_seeds")
    split_configs_raw = args.split_configs if args.split_configs is not None else config.get("split_configs")
    # Parse split_configs: list of [train_split, test_split] e.g. [[0.8, 0.1], [0.7, 0.15]]
    split_configs = None
    if split_configs_raw:
        split_configs = []
        for s in split_configs_raw:
            if isinstance(s, (list, tuple)) and len(s) >= 2:
                split_configs.append((float(s[0]), float(s[1])))
            elif isinstance(s, str):
                a, b = s.strip().split(",")
                split_configs.append((float(a), float(b)))
    data_path = config.get("data_path", "dataset")
    layers = args.layers if getattr(args, "layers", None) is not None else config.get("layers", 4)
    hidden = config.get("hidden", 32)
    dropout = config.get("dropout", 0.5)
    epochs = config.get("epochs", 50)
    lr = config.get("lr", 0.01)
    pfr_lambda = config.get("pfr_lambda", 0.1)
    output_base = config.get("output_base", "outputs")
    # Default results_dir: all run outputs (stratified, residual, option_a/b, etc.) go under output_base (e.g. outputs/)
    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
    elif getattr(args, "residual", False) and getattr(args, "pair_norm", False):
        results_dir = project_root / output_base / "residual_pair_norm"
    elif getattr(args, "residual", False):
        results_dir = project_root / output_base / "residual_gnn"
    elif getattr(args, "option_a", False):
        results_dir = project_root / output_base / "option_a_stability"
    elif getattr(args, "option_b", False):
        results_dir = project_root / output_base / "option_b_nifty"
    elif getattr(args, "stratify_split", False):
        results_dir = project_root / output_base / "stratified_split"
    elif split_seeds and split_configs:
        results_dir = project_root / output_base / "split_seed_and_size"
    elif split_seeds:
        results_dir = project_root / output_base / "split_seed"
    elif split_configs:
        results_dir = project_root / output_base / "split_size"
    else:
        results_dir = Path(config.get("results_dir", "outputs/multi_seed"))
    results_dir = results_dir if results_dir.is_absolute() else project_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    if split_seeds or split_configs:
        print(f"Results will be saved to {results_dir} (separate from baseline {output_base}/multi_seed)")
    if getattr(args, "stratify_split", False):
        print(f"Stratified split enabled. Results will be saved to {results_dir} (separate from multi_seed/split_seed)")
    if getattr(args, "option_a", False):
        print(f"Option A (stability regularization) enabled. Results will be saved to {results_dir}")
    if getattr(args, "option_b", False):
        print(f"Option B (NIFTY-style) enabled. Results will be saved to {results_dir}")
    if getattr(args, "residual", False):
        if getattr(args, "pair_norm", False):
            print(f"Residual GNN + PairNorm enabled. Results will be saved to {results_dir}")
        else:
            print(f"Residual GNN enabled. Results will be saved to {results_dir}")

    if args.seeds is not None:
        seeds = list(args.seeds)
        if not seeds:
            raise ValueError("--seeds must list at least one seed (e.g. 0 5 10 15 20 25 30 35 40 45 50)")
    elif args.seed_start is not None and args.seed_end is not None:
        seeds = list(range(args.seed_start, args.seed_end + 1))
    else:
        # Residual GNN: default seeds 1, 5, 10, ..., 50
        if getattr(args, "residual", False) and args.n_seeds is None and args.seed_start is None and args.seed_end is None:
            seeds = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            print(f"Residual GNN: using seeds {seeds}. Override with --seeds or --n_seeds.")
        # Stratified split: one experiment per (dataset, model) by default — single seed unless user set --n_seeds
        elif getattr(args, "stratify_split", False) and args.n_seeds is None:
            seeds = [0]
            print("Stratified split: using 1 seed (one run per dataset × model). Override with --n_seeds or --seeds if needed.")
        else:
            seeds = list(range(n_seeds))
    runs = []
    fairness_algo_override = getattr(args, "fairness_algo", None)
    mitigation = None
    use_residual = getattr(args, "residual", False)
    use_pair_norm = getattr(args, "pair_norm", False)
    use_gat_balanced_init = getattr(args, "gat_balanced_init", False)
    anti_oversmooth_drop_edge = getattr(args, "anti_oversmooth_drop_edge", 0.0)
    if getattr(args, "option_a", False):
        mitigation = "option_a"
    elif getattr(args, "option_b", False):
        mitigation = "option_b"
    for data_name in datasets:
        for m in models:
            model_name = m.get("name", "GCN")
            fame = m.get("fame", False)
            fairness_algo = fairness_algo_override if fairness_algo_override is not None else m.get("fairness_algo")
            for seed in seeds:
                base_run = {
                    "data_name": data_name,
                    "model": model_name,
                    "fame": fame,
                    "fairness_algo": fairness_algo,
                    "seed": seed,
                    "stratify_split": getattr(args, "stratify_split", False),
                    "mitigation": mitigation,
                    "residual": use_residual,
                    "pair_norm": use_pair_norm,
                    "gat_balanced_init": use_gat_balanced_init,
                    "anti_oversmooth_drop_edge": anti_oversmooth_drop_edge,
                }
                if split_seeds and split_configs:
                    for split_seed in split_seeds:
                        for (train_split, test_split) in split_configs:
                            runs.append({**base_run, "split_seed": split_seed, "train_split": train_split, "test_split": test_split})
                elif split_seeds:
                    for split_seed in split_seeds:
                        runs.append({**base_run, "split_seed": split_seed, "train_split": 0.8, "test_split": 0.1})
                elif split_configs:
                    for (train_split, test_split) in split_configs:
                        runs.append({**base_run, "split_seed": 42, "train_split": train_split, "test_split": test_split})
                else:
                    runs.append({**base_run, "split_seed": 42, "train_split": 0.8, "test_split": 0.1})

    if args.dry_run:
        print(f"Would run {len(runs)} experiments: {len(datasets)} datasets x {len(models)} models x {len(seeds)} seeds" + (f" x {len(split_seeds)} split_seeds" if split_seeds else "") + (f" x {len(split_configs)} split_configs" if split_configs else ""))
        for r in runs[:6]:
            print(" ", r)
        if len(runs) > 6:
            print(" ...")
        return

    existing = []
    if args.append:
        out_file = results_dir / "all_runs.json"
        if out_file.is_file():
            with open(out_file) as f:
                existing = json.load(f)
            print(f"Loaded {len(existing)} existing runs from {out_file}")
        else:
            print("Append requested but no existing all_runs.json; will create new file.")

    all_results = []
    sidecar_path = results_dir / "latest_run_metrics.jsonl"
    print(f"Logging each run's full metrics to {sidecar_path}")
    with open(sidecar_path, "w") as sidecar:
        for i, r in enumerate(runs):
            extra = ""
            if r.get("split_seed", 42) != 42 or r.get("train_split") != 0.8:
                extra = f" split_seed={r.get('split_seed')} split={r.get('train_split')}/{r.get('test_split')}"
            fa = r.get("fairness_algo")
            mit = r.get("mitigation")
            fa_str = f" {fa}" if fa else ""
            mit_str = f" mitigation={mit}" if mit else ""
            print(f"[{i+1}/{len(runs)}] {r['data_name']} | {r['model']} fame={r['fame']}{fa_str}{mit_str} seed={r['seed']}{extra}")
            try:
                metrics = run_one(
                    data_path=data_path,
                    data_name=r["data_name"],
                    model=r["model"],
                    fame=r["fame"],
                    enhanced=False,
                    fairness_algo=r.get("fairness_algo"),
                    pfr_lambda=pfr_lambda,
                    layers=layers,
                    hidden=hidden,
                    dropout=dropout,
                    epochs=epochs,
                    lr=lr,
                    seed=r["seed"],
                    split_seed=r.get("split_seed", 42),
                    train_split=r.get("train_split", 0.8),
                    test_split=r.get("test_split", 0.1),
                    stratify_split=r.get("stratify_split", False),
                    mitigation=r.get("mitigation"),
                    stability_lambda=getattr(args, "stability_lambda", 0.5),
                    nifty_sim_coeff=getattr(args, "nifty_sim_coeff", 0.5),
                    drop_edge_rate=getattr(args, "drop_edge_rate", 0.1),
                    drop_feature_rate=getattr(args, "drop_feature_rate", 0.1),
                    residual=r.get("residual", False),
                    pair_norm=r.get("pair_norm", False),
                    gat_balanced_init=r.get("gat_balanced_init", False),
                    anti_oversmooth_drop_edge=r.get("anti_oversmooth_drop_edge", 0.0),
                    verbose=False,
                    gnns_fame_root=gnns_fame_root,
                )
                all_results.append(metrics)
                # Log metrics to stdout and sidecar (recoverable if save step fails)
                acc = metrics.get("Accuracy")
                spd = metrics.get("Statistical Parity Difference")
                eod = metrics.get("Equal Opportunity Difference")
                oaed = metrics.get("Overall Accuracy Equality Difference")
                acc_s = f"{acc:.4f}" if acc is not None else "N/A"
                spd_s = f"{spd:.4f}" if spd is not None else "N/A"
                eod_s = f"{eod:.4f}" if eod is not None else "N/A"
                oaed_s = f"{oaed:.4f}" if oaed is not None else "N/A"
                print(f"  -> Acc={acc_s} SPD={spd_s} EOD={eod_s} OAED={oaed_s}")
                sidecar.write(json.dumps(metrics, default=str) + "\n")
                sidecar.flush()
            except Exception as e:
                print(f"  FAILED: {e}")
                err_entry = {"error": str(e), **r}
                all_results.append(err_entry)
                sidecar.write(json.dumps(err_entry, default=str) + "\n")
                sidecar.flush()

    out_file = results_dir / "all_runs.json"
    try:
        if args.append and existing:
            def run_key(run):
                return (run.get("_data_name"), run.get("_model"), run.get("_fame"), run.get("_fairness_algo"), run.get("_mitigation"), run.get("_seed"), run.get("_split_seed", 42), run.get("_train_split", 0.8), run.get("_test_split", 0.1))
            new_keys = {run_key(r) for r in all_results}
            merged = [r for r in existing if run_key(r) not in new_keys] + all_results
            merged.sort(key=lambda r: (r.get("_data_name", ""), r.get("_model", ""), r.get("_fame", False), r.get("_fairness_algo") or "", r.get("_mitigation") or "", r.get("_seed", -1), r.get("_split_seed", 42), r.get("_train_split", 0.8), r.get("_test_split", 0.1)))
            with open(out_file, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"Appended {len(all_results)} runs (seeds {seeds[0]}-{seeds[-1]}). Total {len(merged)} runs saved to {out_file}")
        else:
            with open(out_file, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"Saved {len(all_results)} runs to {out_file}")
        print(f"Metrics sidecar: {sidecar_path} ({len(all_results)} lines)")
    except Exception as e:
        print(f"ERROR saving {out_file}: {e}")
        print(f"Recover {len(all_results)} runs from sidecar: {sidecar_path}")
        raise


if __name__ == "__main__":
    main()
