"""
Run a single GNNs-FAME experiment with a fixed seed and return metrics.
Uses the GNNs-FAME codebase without modifying it (imports from their repo).
"""

import math
import os
import random
import sys
from typing import Any, Optional

# GAT uses scatter_reduce which is not implemented on MPS; fallback to CPU for that op.
if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch


def _gnns_fame_root() -> str:
    root = os.environ.get("GNNs_FAME_ROOT")
    if root and os.path.isdir(root):
        return os.path.abspath(root)
    # Default: GNNs-FAME or GNNs-FAME-main next to this package (e.g. gnn-fairness-uncertainty/GNNs-FAME-main)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)
    for folder in ("GNNs-FAME-main", "GNNs-FAME"):
        candidate = os.path.join(project_root, folder)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "GNNs-FAME repo not found. Put it in GNNs-FAME-main/ or GNNs-FAME/ "
        "or set GNNs_FAME_ROOT=/path/to/GNNs-FAME"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(
    data_path: str = "dataset",
    data_name: str = "credit",
    model: str = "GCN",
    fame: bool = False,
    enhanced: bool = False,
    fairness_algo: Optional[str] = None,
    pfr_lambda: float = 0.1,
    layers: int = 4,
    hidden: int = 32,
    dropout: float = 0.5,
    epochs: int = 20,
    lr: float = 0.01,
    seed: int = 0,
    split_seed: Optional[int] = None,
    train_split: float = 0.8,
    test_split: float = 0.1,
    stratify_split: bool = False,
    mitigation: Optional[str] = None,
    stability_lambda: float = 0.5,
    nifty_sim_coeff: float = 0.5,
    drop_edge_rate: float = 0.1,
    drop_feature_rate: float = 0.1,
    residual: bool = False,
    pair_norm: bool = False,
    gat_balanced_init: bool = False,
    anti_oversmooth_drop_edge: float = 0.0,
    verbose: bool = False,
    record_per_epoch: bool = False,
    debug_print_every_epoch: bool = False,
    gnns_fame_root: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run one training/eval run with fixed seed; return metrics dict.
    seed: controls training (init, shuffle). split_seed: controls train/val/test split (defaults to 42 if None).
    train_split, test_split: fractions (val = 1 - train_split - test_split). Vary to study split-size effects.
    fairness_algo: in-processing fairness; "pfr" = Prejudice Remover (loss regularization). None = standard training.
    pfr_lambda: weight for PFR term when fairness_algo=="pfr".
    mitigation: "option_a" = stability regularization (Huang–Vishnoi style); "option_b" = NIFTY-style augmentation+similarity. None = standard.
    stability_lambda: weight for stability loss when mitigation=="option_a".
    nifty_sim_coeff, drop_edge_rate, drop_feature_rate: used when mitigation=="option_b".
    pair_norm: use PairNorm after each conv (GCN/GAT/GIN). Reduces oversmoothing.
    gat_balanced_init: use balanced init for GAT only (Mustafa et al., NeurIPS 2023).
    anti_oversmooth_drop_edge: drop this fraction of edges each epoch during training (Rong et al., DropEdge). Use for GAT/GIN anti-collapse. Val/test use full graph.
    record_per_epoch: if True (vanilla training only), record loss/val_loss/acc/SPD/EOD/pred_pos each epoch for debug comparison.
    debug_print_every_epoch: if True (and record_per_epoch), print every epoch's loss/val_loss/pred_pos/acc/SPD/EOD so you can compare both runs in the debugger.
    Metrics include Accuracy and fairness metrics (SPD, EOD, OAED, TED, etc.).
    """
    root = gnns_fame_root or _gnns_fame_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    _split_seed = 42 if split_seed is None else split_seed
    cwd_orig = os.getcwd()
    try:
        os.chdir(root)
        set_seed(seed)

        from preprocess_data import preprocess_data
        from model import GNN, train, test
        from utils import set_device

        device = set_device()
        data, sens_attributes = preprocess_data(
            data_path, data_name, train_split=train_split, test_split=test_split, split_seed=_split_seed,
            stratify_by="sens_label" if stratify_split else None,
        )
        gnn = GNN(
            data, model, fame, enhanced, sens_attributes,
            layers=layers, hidden=hidden, dropout=dropout,
            residual=residual,
            pair_norm=pair_norm,
            gat_balanced_init=gat_balanced_init,
        )
        optimizer = torch.optim.AdamW(gnn.parameters(), lr=lr)
        gnn.to(device)
        data.to(device)
        sens_attributes.to(device)

        gnn.train()
        per_epoch_recorded = None
        if mitigation == "option_a":
            from .stability_training import train_with_stability
            train_with_stability(
                gnn, data, optimizer, epochs,
                stability_lambda=stability_lambda,
                drop_edge_rate=anti_oversmooth_drop_edge,
            )
        elif mitigation == "option_b":
            from .nifty_style_training import train_with_nifty
            train_with_nifty(
                gnn, data, optimizer, epochs,
                sim_coeff=nifty_sim_coeff,
                drop_edge_rate=drop_edge_rate,
                drop_feature_rate=drop_feature_rate,
                anti_oversmooth_drop_edge=anti_oversmooth_drop_edge,
            )
        elif fairness_algo == "pfr":
            from .pfr_training import train_with_pfr
            train_with_pfr(
                gnn, data, sens_attributes, optimizer, epochs,
                pfr_lambda=pfr_lambda,
                drop_edge_rate=anti_oversmooth_drop_edge,
            )
        else:
            if anti_oversmooth_drop_edge > 0:
                from .train_with_dropedge import train_standard_with_dropedge
                train_standard_with_dropedge(
                    gnn, data, optimizer, epochs,
                    drop_edge_rate=anti_oversmooth_drop_edge,
                )
            elif record_per_epoch:
                from torch.nn import NLLLoss
                from .fairness_test_only import fairness_metrics_on_subset, test_set_diagnostics
                loss_fn = NLLLoss()
                # Once per run: test-set base rate and size (for collapse check)
                _mask_cpu = data.test_mask.cpu()
                _y_test = data.y.cpu()[_mask_cpu].float()
                base_rate_test = (_y_test == 1).float().mean().item()
                n_test = int(_y_test.numel())
                # D) Class weighting: train-set imbalance (plain NLL = trivial basin attractive)
                _train_y = data.y[data.train_mask].cpu()
                n_pos_train = int((_train_y == 1).sum().item())
                n_neg_train = int((_train_y == 0).sum().item())
                n_train = int(_train_y.numel())
                if debug_print_every_epoch:
                    print(
                        f"  Test set: n_test={n_test} base_rate={base_rate_test:.6f} "
                        f"(collapse check: acc=base_rate if pred_pos=1, acc=1-base_rate if pred_pos=0)"
                    )
                    print(
                        f"  Class weighting: none (plain NLL). Train n_pos={n_pos_train} n_neg={n_neg_train} "
                        f"→ consider weight=[1, {n_neg_train / max(1, n_pos_train):.2f}] for CrossEntropyLoss"
                    )
                    # C) Sanity: val_loss on val_mask, metrics on test_mask, labels Long for NLL
                    print(
                        "  Sanity: val_loss on val_mask, confusion/SPD/EOD on test_mask; labels Long for NLL; logits shape (N, num_classes)"
                    )
                per_epoch = []
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    out = gnn(data.x, data.edge_index)
                    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
                    loss.backward(retain_graph=True)
                    # B) Gradient and parameter norms (blow-up → reduce LR/clip; ~0 early → saturating)
                    total_grad_norm = 0.0
                    for p in gnn.parameters():
                        if p.grad is not None:
                            total_grad_norm += p.grad.data.norm(2).item() ** 2
                    total_grad_norm = math.sqrt(total_grad_norm)
                    params_list = list(gnn.parameters())
                    last_W = next((p for p in reversed(params_list) if p.dim() >= 2), None)
                    last_W_norm = float(last_W.norm().item()) if last_W is not None else float("nan")
                    optimizer.step()
                    gnn.eval()
                    with torch.no_grad():
                        val_out = gnn(data.x, data.edge_index)
                        val_loss = loss_fn(val_out[data.val_mask], data.y[data.val_mask]).item()
                        out_t = gnn(data.x, data.edge_index)
                        pred = out_t.argmax(dim=1)
                        # A) Train-set collapse (if train collapses → optimization/imbalance; if only val/test → generalization/mask bug)
                        mask_train_cpu = data.train_mask.cpu()
                        pred_train = pred.cpu()[mask_train_cpu].float()
                        y_train = data.y.cpu()[mask_train_cpu].float()
                        pred_pos_train = pred_train.mean().item()
                        tp_train = ((pred_train == 1) & (y_train == 1)).sum().item()
                        fp_train = ((pred_train == 1) & (y_train == 0)).sum().item()
                        fn_train = ((pred_train == 0) & (y_train == 1)).sum().item()
                        tn_train = ((pred_train == 0) & (y_train == 0)).sum().item()
                        # Index on CPU to avoid device mismatch (e.g. sens_attributes on CPU, data.test_mask on MPS)
                        mask_cpu = data.test_mask.cpu()
                        y_t = data.y.cpu()[mask_cpu].float()
                        pred_t = pred.cpu()[mask_cpu].float()
                        sens_t = sens_attributes.cpu()[mask_cpu].float()
                        acc = (pred_t == y_t).float().mean().item()
                        # Logits (log-probs) on test set: min/mean/max per class to diagnose collapse
                        out_test = out_t.cpu()[mask_cpu]
                        if out_test.dim() >= 2 and out_test.size(-1) >= 2:
                            lp0 = out_test[:, 0]
                            lp1 = out_test[:, 1]
                            logprob0_min, logprob0_mean, logprob0_max = lp0.min().item(), lp0.mean().item(), lp0.max().item()
                            logprob1_min, logprob1_mean, logprob1_max = lp1.min().item(), lp1.mean().item(), lp1.max().item()
                            logprob0_std = lp0.std().item()
                            logprob1_std = lp1.std().item()
                            # Margin diagnostic: margin = logprob1 - logprob0 (same sign as logit margin; %margin>0 should match pred_pos)
                            margin_test = lp1 - lp0
                            margin_mean = margin_test.mean().item()
                            margin_std = margin_test.std().item()
                            pct_margin_positive = (margin_test > 0).float().mean().item()
                            out_train = out_t.cpu()[mask_train_cpu]
                            if out_train.dim() >= 2 and out_train.size(-1) >= 2:
                                margin_train = out_train[:, 1] - out_train[:, 0]
                                margin_train_mean = margin_train.mean().item()
                                margin_train_std = margin_train.std().item()
                                pct_margin_positive_train = (margin_train > 0).float().mean().item()
                            else:
                                margin_train_mean = margin_train_std = pct_margin_positive_train = None
                        else:
                            logprob0_min = logprob0_mean = logprob0_max = logprob1_min = logprob1_mean = logprob1_max = None
                            logprob0_std = logprob1_std = None
                            margin_mean = margin_std = pct_margin_positive = None
                            margin_train_mean = margin_train_std = pct_margin_positive_train = None
                        # Confusion matrix counts (collapse diagnosis)
                        tp = ((pred_t == 1) & (y_t == 1)).sum().item()
                        fp = ((pred_t == 1) & (y_t == 0)).sum().item()
                        fn_ = ((pred_t == 0) & (y_t == 1)).sum().item()
                        tn = ((pred_t == 0) & (y_t == 0)).sum().item()
                        m_test = fairness_metrics_on_subset(y_t, pred_t, sens_t)
                        diag = test_set_diagnostics(y_t, pred_t, sens_t)
                    gnn.train()
                    spd = m_test.get("Statistical Parity Difference")
                    eod = m_test.get("Equal Opportunity Difference")
                    pred_pos = diag.get("diagnostics_pred_pos_overall")
                    per_epoch.append({
                        "epoch": epoch,
                        "loss": float(loss.item()),
                        "val_loss": val_loss,
                        "Accuracy": acc,
                        "SPD": spd,
                        "EOD": eod,
                        "pred_pos_overall": pred_pos,
                        "TP": tp, "FP": fp, "TN": tn, "FN": fn_,
                        "pred_pos_train": pred_pos_train,
                        "TP_train": tp_train, "FP_train": fp_train, "TN_train": tn_train, "FN_train": fn_train,
                        "total_grad_norm": total_grad_norm,
                        "last_W_norm": last_W_norm,
                        "margin_mean": margin_mean, "margin_std": margin_std, "pct_margin_positive": pct_margin_positive,
                        "margin_train_mean": margin_train_mean, "margin_train_std": margin_train_std, "pct_margin_positive_train": pct_margin_positive_train,
                        "logprob0_min": logprob0_min, "logprob0_mean": logprob0_mean, "logprob0_max": logprob0_max, "logprob0_std": logprob0_std,
                        "logprob1_min": logprob1_min, "logprob1_mean": logprob1_mean, "logprob1_max": logprob1_max, "logprob1_std": logprob1_std,
                    })
                    if debug_print_every_epoch:
                        print(
                            f"  [seed={seed} epoch={epoch}] loss={loss.item():.6f} val_loss={val_loss:.6f} "
                            f"pred_pos={pred_pos} acc={acc} SPD={spd} EOD={eod} TP={tp} FP={fp} TN={tn} FN={fn_}"
                        )
                        print(
                            f"       train: pred_pos_train={pred_pos_train:.4f} (TP,FP,TN,FN)_train=({tp_train},{fp_train},{tn_train},{fn_train}) "
                            f"||grad||={total_grad_norm:.4f} ||W_last||={last_W_norm:.4f}"
                        )
                        if margin_mean is not None:
                            train_part = f" | train mean={margin_train_mean:.4f} std={margin_train_std:.4f} %margin>0={pct_margin_positive_train:.4f}" if margin_train_mean is not None else ""
                            print(
                                f"       margin (logprob1-logprob0): test mean={margin_mean:.4f} std={margin_std:.4f} %margin>0={pct_margin_positive:.4f} "
                                f"(=pred_pos){train_part}"
                            )
                        if logprob0_mean is not None:
                            print(
                                f"       logprob class0: min={logprob0_min:.4f} mean={logprob0_mean:.4f} std={logprob0_std:.4f} max={logprob0_max:.4f} | "
                                f"class1: min={logprob1_min:.4f} mean={logprob1_mean:.4f} std={logprob1_std:.4f} max={logprob1_max:.4f}"
                            )
                per_epoch_recorded = per_epoch
            else:
                train(gnn, data, optimizer, epochs)
        gnn.eval()
        metrics = test(gnn, data, sens_attributes, verbose=verbose)
        if per_epoch_recorded is not None:
            metrics["_per_epoch"] = per_epoch_recorded
        # Recompute fairness on TEST SET ONLY. GNNs-FAME test() uses their calculate_fairness(data, predictions, sens_attributes)
        # with full-graph predictions/labels (all nodes), so SPD/EOD are over train+val+test. We need held-out evaluation:
        # same metric definitions (SPD, EOD, OAED, TED) but on test set only. fairness_test_only does that.
        # Ensure data/model on same device (test() may have moved data to CPU in GNNs-FAME code)
        device = next(gnn.parameters()).device
        data = data.to(device)
        sens_attributes = sens_attributes.to(device)
        from .fairness_test_only import fairness_metrics_on_subset, test_set_diagnostics
        with torch.no_grad():
            out = gnn(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        mask_cpu = data.test_mask.cpu()
        y_t = data.y.cpu()[mask_cpu].float()
        pred_t = pred.cpu()[mask_cpu].float()
        sens_t = sens_attributes.cpu()[mask_cpu].float()
        metrics_test = fairness_metrics_on_subset(y_t, pred_t, sens_t)
        diagnostics = test_set_diagnostics(y_t, pred_t, sens_t)
        # Once per run: base_rate and confusion matrix (collapse check)
        base_rate_test = (y_t == 1).float().mean().item()
        tp_final = ((pred_t == 1) & (y_t == 1)).sum().item()
        fp_final = ((pred_t == 1) & (y_t == 0)).sum().item()
        fn_final = ((pred_t == 0) & (y_t == 1)).sum().item()
        tn_final = ((pred_t == 0) & (y_t == 0)).sum().item()
        metrics = {
            **metrics_test, "Accuracy": metrics["Accuracy"], **diagnostics,
            "base_rate_test": base_rate_test,
            "TP": tp_final, "FP": fp_final, "TN": tn_final, "FN": fn_final,
        }
        # Full-graph fairness (same metrics, all nodes) for comparison with test-set-only
        y_full = data.y.cpu().float()
        pred_full = pred.cpu().float()
        sens_full = sens_attributes.cpu().float()
        metrics_full = fairness_metrics_on_subset(y_full, pred_full, sens_full)
        acc_full = (pred_full == y_full).float().mean().item()
        for k, v in metrics_full.items():
            metrics[f"{k}_full_graph"] = v
        metrics["Accuracy_full_graph"] = acc_full
        # Record stratified-split composition so we can verify train/val/test have same (S,Y) %
        if stratify_split:
            _y = data.y.cpu().numpy().ravel()
            _s = sens_attributes.cpu().numpy().ravel()
            for name, mask in [("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)]:
                m = mask.cpu().numpy()
                if m.sum() == 0:
                    continue
                y_m, s_m = _y[m], _s[m]
                n = len(y_m)
                n00 = ((s_m == 0) & (y_m == 0)).sum()
                n01 = ((s_m == 0) & (y_m == 1)).sum()
                n10 = ((s_m == 1) & (y_m == 0)).sum()
                n11 = ((s_m == 1) & (y_m == 1)).sum()
                metrics[f"_stratify_{name}_pct_S0_Y0"] = round(100 * n00 / n, 2)
                metrics[f"_stratify_{name}_pct_S0_Y1"] = round(100 * n01 / n, 2)
                metrics[f"_stratify_{name}_pct_S1_Y0"] = round(100 * n10 / n, 2)
                metrics[f"_stratify_{name}_pct_S1_Y1"] = round(100 * n11 / n, 2)
    finally:
        os.chdir(cwd_orig)

    # Ensure values are JSON-serializable (e.g. torch tensors -> float; nan -> None)
    def _sanitize(x):
        if hasattr(x, "item"):
            return float(x.item())
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        if isinstance(x, (int, float)):
            return x
        if isinstance(x, list):
            return [_sanitize_item(i) for i in x]
        return float(x) if x is not None else None

    def _sanitize_item(d):
        if not isinstance(d, dict):
            return _sanitize(d)
        return {k: _sanitize(v) for k, v in d.items()}

    out = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            out[k] = _sanitize(v)
        elif hasattr(v, "item"):
            out[k] = float(v.item())
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        elif isinstance(v, (int, float)):
            out[k] = v
        else:
            out[k] = float(v) if v is not None else None
    out["_seed"] = seed
    out["_data_name"] = data_name
    out["_model"] = model
    out["_fame"] = fame
    out["_fairness_algo"] = fairness_algo
    out["_split_seed"] = _split_seed
    out["_train_split"] = train_split
    out["_test_split"] = test_split
    out["_stratify_split"] = stratify_split
    out["_mitigation"] = mitigation
    out["_residual"] = residual
    out["_pair_norm"] = pair_norm
    out["_gat_balanced_init"] = gat_balanced_init
    out["_anti_oversmooth_drop_edge"] = anti_oversmooth_drop_edge
    return out


def _cli():
    """CLI for debugging: run one seed with optional per-epoch recording."""
    import argparse
    ap = argparse.ArgumentParser(description="Run one experiment (for debugger: set breakpoints in run_one.py)")
    ap.add_argument("--dataset", default="bail", help="Dataset (default: bail)")
    ap.add_argument("--model", default="GAT", choices=["GCN", "GAT", "GIN"], help="Model (default: GAT)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    ap.add_argument("--epochs", type=int, default=50, help="Epochs (default: 50)")
    ap.add_argument("--record-per-epoch", action="store_true", help="Record per-epoch metrics for comparison")
    ap.add_argument("--debug-print", action="store_true", help="Print every epoch (loss, pred_pos, acc, SPD, EOD) for debugger comparison")
    ap.add_argument("--split-seed", type=int, default=42, help="Data split seed (default: 42)")
    args = ap.parse_args()
    out = run_one(
        data_name=args.dataset,
        model=args.model,
        fame=False,
        seed=args.seed,
        split_seed=args.split_seed,
        train_split=0.8,
        test_split=0.1,
        epochs=args.epochs,
        record_per_epoch=args.record_per_epoch or args.debug_print,
        debug_print_every_epoch=args.debug_print,
    )
    # Collapse check: base_rate ≈ acc with pred_pos=1, or 1-base_rate ≈ acc with pred_pos=0 => degenerate
    br = out.get("base_rate_test")
    acc = out.get("Accuracy")
    pred_pos = out.get("diagnostics_pred_pos_overall")
    if br is not None and acc is not None:
        print(f"Collapse check: base_rate_test={br:.6f} Accuracy={acc:.6f} pred_pos_overall={pred_pos} (TP,FP,TN,FN)=({out.get('TP')},{out.get('FP')},{out.get('TN')},{out.get('FN')})")
    print("Final:", {k: v for k, v in out.items() if not k.startswith("_") or k == "_per_epoch"})
    if out.get("_per_epoch"):
        print("Per-epoch (first/last 2):", out["_per_epoch"][:2], "...", out["_per_epoch"][-2:])
    return out


if __name__ == "__main__":
    _cli()
