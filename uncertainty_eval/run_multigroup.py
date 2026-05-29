"""
Run multigroup experiments: Setting A (binary Y + multigroup S) or Setting B (multiclass Y + multigroup S).
Saves all results and logs under results/multigroup_setting_a/ or results/multigroup_setting_b/.
"""

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch


def _gnns_fame_root() -> str:
    root = os.environ.get("GNNs_FAME_ROOT")
    if root and os.path.isdir(root):
        return os.path.abspath(root)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)
    for folder in ("GNNs-FAME-main", "GNNs-FAME"):
        candidate = os.path.join(project_root, folder)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "GNNs-FAME repo not found. Put it in GNNs-FAME-main/ or set GNNs_FAME_ROOT."
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_multigroup(
    setting: str,
    data_path: str,
    data_name: str,
    model: str,
    fame: bool = False,
    enhanced: bool = False,
    layers: int = 4,
    hidden: int = 32,
    dropout: float = 0.5,
    epochs: int = 50,
    lr: float = 0.01,
    seed: int = 0,
    split_seed: int = 42,
    train_split: float = 0.8,
    test_split: float = 0.1,
    stratify_split: bool = False,
    gnns_fame_root: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """Run one multigroup experiment; return metrics (Accuracy, SPD, EOD, OAED, TED, etc.)."""
    root = gnns_fame_root or _gnns_fame_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    cwd_orig = os.getcwd()
    try:
        os.chdir(root)
        set_seed(seed)

        from preprocess_multigroup import preprocess_multigroup
        from model import GNN, train
        from utils import set_device

        data, sens_attr = preprocess_multigroup(
            data_path,
            data_name,
            setting=setting,
            train_split=train_split,
            test_split=test_split,
            split_seed=split_seed,
            stratify_by="sens_label" if stratify_split else None,
        )
        num_classes = getattr(data, "num_classes", 2)
        gnn = GNN(
            data,
            model,
            fame=fame,
            enhanced=enhanced,
            sens_attribute=sens_attr,
            layers=layers,
            hidden=hidden,
            dropout=dropout,
            num_classes=num_classes,
        )
        optimizer = torch.optim.AdamW(gnn.parameters(), lr=lr)
        device = set_device()
        gnn.to(device)
        data.to(device)
        sens_attr.to(device)

        gnn.train()
        loss_fn = torch.nn.NLLLoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            out = gnn(data.x, data.edge_index)
            loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        gnn.eval()
        with torch.no_grad():
            out = gnn(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        mask_cpu = data.test_mask.cpu()
        y_test = data.y.cpu()[mask_cpu]
        pred_test = pred.cpu()[mask_cpu]
        sens_test = sens_attr.cpu()[mask_cpu]

        accuracy = (pred_test == y_test).float().mean().item()
        if num_classes > 2:
            from sklearn.metrics import f1_score
            f1_macro = float(f1_score(y_test.numpy(), pred_test.numpy(), average="macro", zero_division=0))
        else:
            f1_macro = None

        from .multigroup_fairness import multigroup_fairness_setting_a, multigroup_fairness_setting_b

        if setting == "a":
            metrics = multigroup_fairness_setting_a(
                y_test.float(), pred_test.float(), sens_test
            )
        else:
            metrics = multigroup_fairness_setting_b(
                y_test, pred_test, sens_test, num_classes=num_classes
            )

        metrics["Accuracy"] = accuracy
        if f1_macro is not None:
            metrics["F1_macro"] = f1_macro
        metrics["_seed"] = seed
        metrics["_data_name"] = data_name
        metrics["_model"] = model
        metrics["_setting"] = setting
        metrics["_split_seed"] = split_seed
        metrics["_num_classes"] = num_classes

        def _sanitize(x):
            if hasattr(x, "item"):
                return float(x.item())
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            if isinstance(x, (int, float)):
                return x
            if isinstance(x, dict):
                return {str(k): _sanitize(v) for k, v in x.items()}
            return x

        out_metrics = {}
        for k, v in metrics.items():
            if k.startswith("multigroup_"):
                out_metrics[k] = _sanitize(v)
            elif hasattr(v, "item"):
                out_metrics[k] = float(v.item())
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                out_metrics[k] = None
            elif isinstance(v, (int, float)):
                out_metrics[k] = v
            else:
                out_metrics[k] = _sanitize(v)
        return out_metrics
    finally:
        os.chdir(cwd_orig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multigroup experiments (Setting A or B). Results and logs go to results/multigroup_setting_a/ or results/multigroup_setting_b/."
    )
    parser.add_argument(
        "--setting",
        choices=["a", "b"],
        required=True,
        help="Setting A (binary Y + multigroup S) or B (multiclass Y + multigroup S)",
    )
    parser.add_argument(
        "--dataset",
        default="credit",
        choices=["credit", "german", "bail"],
        help="Dataset (default: credit)",
    )
    parser.add_argument(
        "--model",
        default="GCN",
        choices=["GCN", "GAT", "GIN"],
        help="Model (default: GCN)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None,
        help="If set, run seeds 0..n_seeds-1 and save to all_runs.json",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Data split seed (default: 42)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs (default: 50)",
    )
    parser.add_argument(
        "--data_path",
        default="dataset",
        help="Path to dataset folder (default: dataset)",
    )
    parser.add_argument(
        "--stratify_split",
        action="store_true",
        help="Stratify train/val/test by (S,Y)",
    )
    parser.add_argument(
        "--fame",
        action="store_true",
        help="Use FAME variant of the model",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="GNN depth (default: 4)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=32,
        help="Hidden dimension (default: 32)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout (default: 0.5)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--gnns_fame_root",
        type=str,
        default=None,
        help="Path to GNNs-FAME repo (default: auto-detect)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Run multiple datasets (e.g. credit german bail). Default: single --dataset.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Run multiple models (e.g. GCN GAT GIN). Default: single --model.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results" / f"multigroup_setting_{args.setting}"
    results_dir.mkdir(parents=True, exist_ok=True)
    # Timestamped log file so multiple runs don't overwrite each other
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"multigroup_{args.setting}_{ts}.log"

    gnns_fame_root = args.gnns_fame_root or _gnns_fame_root()
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(gnns_fame_root, data_path)

    datasets = args.datasets if args.datasets else [args.dataset]
    models = args.models if args.models else [args.model]

    if args.n_seeds is not None:
        seeds = list(range(args.n_seeds))
        total = len(datasets) * len(models) * len(seeds)
        print(f"Running {total} runs: {len(datasets)} datasets x {len(models)} models x {len(seeds)} seeds, setting {args.setting}")
        all_results = []
        run_idx = 0
        with open(log_path, "w") as log_file:
            for data_name in datasets:
                for model_name in models:
                    for seed in seeds:
                        run_idx += 1
                        print(f"[{run_idx}/{total}] {data_name} | {model_name} | seed={seed}")
                        log_file.write(f"[{run_idx}/{total}] {data_name} {model_name} setting={args.setting} seed={seed}\n")
                        log_file.flush()
                        try:
                            metrics = run_one_multigroup(
                                setting=args.setting,
                                data_path=data_path,
                                data_name=data_name,
                                model=model_name,
                                fame=args.fame,
                                layers=args.layers,
                                hidden=args.hidden,
                                dropout=args.dropout,
                                epochs=args.epochs,
                                lr=args.lr,
                                seed=seed,
                                split_seed=args.split_seed,
                                stratify_split=args.stratify_split,
                                gnns_fame_root=gnns_fame_root,
                            )
                            all_results.append(metrics)
                            # Print all metrics (including multigroup details) for full transparency
                            line = json.dumps(metrics, indent=2)
                            print(line)
                            log_file.write(line + "\n")
                            log_file.flush()
                        except Exception as e:
                            print(f"  FAILED: {e}")
                            log_file.write(f"  FAILED: {e}\n")
                            log_file.flush()
                            all_results.append({"error": str(e), "_seed": seed, "_data_name": data_name, "_model": model_name, "_setting": args.setting})

        out_file = results_dir / "all_runs.json"
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved {len(all_results)} runs to {out_file}")
        print(f"Log: {log_path}")
    else:
        metrics = run_one_multigroup(
            setting=args.setting,
            data_path=data_path,
            data_name=args.dataset,
            model=args.model,
            fame=args.fame,
            layers=args.layers,
            hidden=args.hidden,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            split_seed=args.split_seed,
            stratify_split=args.stratify_split,
            gnns_fame_root=gnns_fame_root,
        )
        # Print full metrics for the single run
        print("Metrics:")
        print(json.dumps(metrics, indent=2))
        single_path = results_dir / f"run_{args.dataset}_{args.model}_seed{args.seed}.json"
        with open(single_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved to {single_path}")


if __name__ == "__main__":
    main()
