"""
Fairness metrics computed on a subset of nodes (e.g. test set only).
Same definitions as GNNs-FAME calculate_fairness, but on (labels, predictions, sens) slices.
Use this so evaluation is on held-out test nodes, not train+val+test.

Also provides test-set diagnostics for reviewers: pred positive rate, TPR/FPR by group, counts.
"""

from typing import Any, Dict

import torch


def _binarize_sens_if_needed(sens: torch.Tensor) -> torch.Tensor:
    """If sensitive attribute has more than 2 values, binarize at median (S=1 if >= median)."""
    sens = sens.float()
    uniq = torch.unique(sens[~torch.isnan(sens)])
    if uniq.numel() <= 2:
        return sens
    median = torch.median(sens)
    return (sens >= median).float()


def test_set_diagnostics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    sens_attr: torch.Tensor,
) -> Dict[str, Any]:
    """
    Compute diagnostics from test-set (labels, predictions, sens_attr).
    - Predicted positive rate overall and by group (exposes constant-classifier collapse).
    - TPR/FPR by group (shows confusion-matrix rates; TE explodes when denominators are small).
    - Test-split counts and class imbalance (explains tiny denominators).
    Sensitive attribute is binarized at median if it has more than 2 values.
    """
    labels = labels.float()
    predictions = predictions.float()
    sens_attr = _binarize_sens_if_needed(sens_attr.float())
    n = len(labels)
    mask0 = sens_attr == 0
    mask1 = sens_attr == 1
    n_S0 = int(mask0.sum().item())
    n_S1 = int(mask1.sum().item())
    n_pos_S0 = int((labels[mask0] == 1).sum().item())
    n_pos_S1 = int((labels[mask1] == 1).sum().item())
    n_neg_S0 = n_S0 - n_pos_S0
    n_neg_S1 = n_S1 - n_pos_S1

    pred_pos_overall = predictions.mean().item()
    pred_pos_S0 = predictions[mask0].mean().item() if n_S0 else float("nan")
    pred_pos_S1 = predictions[mask1].mean().item() if n_S1 else float("nan")

    def tp_fp_fn_tn(mask):
        y_m = labels[mask]
        p_m = predictions[mask]
        tp = ((p_m == 1) & (y_m == 1)).sum().item()
        fp = ((p_m == 1) & (y_m == 0)).sum().item()
        fn_ = ((p_m == 0) & (y_m == 1)).sum().item()
        tn = ((p_m == 0) & (y_m == 0)).sum().item()
        return tp, fp, fn_, tn

    tp0, fp0, fn0, tn0 = tp_fp_fn_tn(mask0)
    tp1, fp1, fn1, tn1 = tp_fp_fn_tn(mask1)
    den_tpr0 = tp0 + fn0
    den_tpr1 = tp1 + fn1
    den_fpr0 = fp0 + tn0
    den_fpr1 = fp1 + tn1
    TPR_S0 = (tp0 / den_tpr0) if den_tpr0 else float("nan")
    TPR_S1 = (tp1 / den_tpr1) if den_tpr1 else float("nan")
    FPR_S0 = (fp0 / den_fpr0) if den_fpr0 else float("nan")
    FPR_S1 = (fp1 / den_fpr1) if den_fpr1 else float("nan")

    return {
        "diagnostics_n_test": n,
        "diagnostics_n_S0": n_S0,
        "diagnostics_n_S1": n_S1,
        "diagnostics_n_pos_S0": n_pos_S0,
        "diagnostics_n_pos_S1": n_pos_S1,
        "diagnostics_pred_pos_overall": pred_pos_overall,
        "diagnostics_pred_pos_S0": pred_pos_S0,
        "diagnostics_pred_pos_S1": pred_pos_S1,
        "diagnostics_TPR_S0": TPR_S0,
        "diagnostics_TPR_S1": TPR_S1,
        "diagnostics_FPR_S0": FPR_S0,
        "diagnostics_FPR_S1": FPR_S1,
        "diagnostics_TPR_denom_S0": den_tpr0,
        "diagnostics_TPR_denom_S1": den_tpr1,
        "diagnostics_FPR_denom_S0": den_fpr0,
        "diagnostics_FPR_denom_S1": den_fpr1,
    }


def fairness_metrics_on_subset(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    sens_attr: torch.Tensor,
) -> Dict[str, Any]:
    """
    Compute SPD, EOD, OAED, TED on the given (labels, predictions, sens_attr).
    All tensors must be 1D, same length (e.g. test set only).
    Sensitive attribute is binarized at median if it has more than 2 values.
    Returns nan for a metric if a group is empty (e.g. no S=0 or no S=1 in test).
    """
    labels = labels.float()
    predictions = predictions.float()
    sens_attr = _binarize_sens_if_needed(sens_attr.float())

    def _safe_mean(t: torch.Tensor):
        if t.numel() == 0:
            return float("nan")
        return t.mean().item()

    def spd():
        m0 = sens_attr == 0
        m1 = sens_attr == 1
        p0 = _safe_mean(predictions[m0])
        p1 = _safe_mean(predictions[m1])
        diff = abs(p1 - p0) if (p0 == p0 and p1 == p1) else float("nan")
        return diff, p0, p1

    def eod():
        pos0 = (labels == 1) & (sens_attr == 0)
        pos1 = (labels == 1) & (sens_attr == 1)
        tpr0 = _safe_mean(predictions[pos0])
        tpr1 = _safe_mean(predictions[pos1])
        diff = abs(tpr1 - tpr0) if (tpr0 == tpr0 and tpr1 == tpr1) else float("nan")
        return diff, tpr0, tpr1

    def oaed():
        m0 = sens_attr == 0
        m1 = sens_attr == 1
        correct0 = (predictions[m0] == labels[m0]).float()
        correct1 = (predictions[m1] == labels[m1]).float()
        acc0 = _safe_mean(correct0)
        acc1 = _safe_mean(correct1)
        diff = abs(acc1 - acc0) if (acc0 == acc0 and acc1 == acc1) else float("nan")
        return diff, acc0, acc1

    def ted():
        fn1 = ((predictions == 0) & (labels == 1) & (sens_attr == 1)).sum().item()
        fp1 = ((predictions == 1) & (labels == 0) & (sens_attr == 1)).sum().item()
        fn0 = ((predictions == 0) & (labels == 1) & (sens_attr == 0)).sum().item()
        fp0 = ((predictions == 1) & (labels == 0) & (sens_attr == 0)).sum().item()
        r1 = fn1 / fp1 if fp1 != 0 else float("inf")
        r0 = fn0 / fp0 if fp0 != 0 else float("inf")
        diff = abs(r1 - r0) if (r1 != float("inf") and r0 != float("inf")) else 0.0
        return diff, r0, r1

    spd_val, sp_g0, sp_g1 = spd()
    eod_val, eod_g0, eod_g1 = eod()
    oaed_val, oaed_g0, oaed_g1 = oaed()
    ted_val, ted_g0, ted_g1 = ted()

    return {
        "Statistical Parity Difference": spd_val,
        "Statistical Parity Group with S=0": sp_g0,
        "Statistical Parity Group S=1": sp_g1,
        "Equal Opportunity Difference": eod_val,
        "Equal Opportunity Group with S=0": eod_g0,
        "Equal Opportunity Group S=1": eod_g1,
        "Overall Accuracy Equality Difference": oaed_val,
        "Overall Accuracy Group with S=0": oaed_g0,
        "Overall Accuracy Group S=1": oaed_g1,
        "Treatment Equality Difference": ted_val,
        "Treatment Equality Group with S=0": ted_g0,
        "Treatment Equality Group S=1": ted_g1,
    }
