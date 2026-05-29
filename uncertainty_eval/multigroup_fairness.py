"""
Multigroup fairness metrics (Purificato et al.): Setting A = binary Y + multigroup S (Eqs 9-12),
Setting B = multiclass Y + multigroup S (Eqs 13-16). Computed on a subset (e.g. test set).
"""

from typing import Any, Dict

import torch


def _safe_mean(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return float("nan")
    return t.mean().item()


# ---------- Setting A: binary Y, multigroup S (Eqs 9-12) ----------


def multigroup_fairness_setting_a(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    sens_attr: torch.Tensor,
) -> Dict[str, Any]:
    """
    Binary Y, multigroup S. labels and predictions in {0, 1}; sens_attr in {0, 1, ..., N-1}.
    Returns: per-group rates + max gap (range) across groups as the fairness "difference" metric.
    """
    labels = labels.float()
    predictions = predictions.float()
    sens_attr = sens_attr.long().ravel()
    uniq = torch.unique(sens_attr)
    uniq = uniq[~torch.isnan(uniq.float())]

    sp_per_group = {}
    eo_per_group = {}
    oae_per_group = {}
    te_per_group = {}

    for n in uniq.tolist():
        mask = sens_attr == n
        if mask.sum() == 0:
            continue
        # SP: P(ŷ=1 | s=n)
        sp_per_group[n] = _safe_mean(predictions[mask])
        # EO: P(ŷ=1 | y=1, s=n)
        pos_mask = mask & (labels == 1)
        eo_per_group[n] = _safe_mean(predictions[pos_mask]) if pos_mask.sum() > 0 else float("nan")
        # OAE: P(ŷ=0|y=0,s=n) + P(ŷ=1|y=1,s=n) = accuracy within group
        correct = (predictions[mask] == labels[mask]).float()
        oae_per_group[n] = _safe_mean(correct)
        # TE (Eq. 12): P(ŷ=1 | y=0,s=n) / P(ŷ=0 | y=1,s=n)
        # Numerator: false positives over all negatives in group n
        # Denominator: false negatives over all positives in group n
        neg_mask = mask & (labels == 0)
        pos_mask_full = mask & (labels == 1)
        fp_n = ((predictions == 1) & neg_mask).sum().item()
        fn_n = ((predictions == 0) & pos_mask_full).sum().item()
        denom_fp = neg_mask.sum().item()
        denom_fn = pos_mask_full.sum().item()
        p_fp = fp_n / denom_fp if denom_fp > 0 else float("nan")
        p_fn = fn_n / denom_fn if denom_fn > 0 else float("nan")
        if p_fp != p_fp or p_fn != p_fn or p_fn == 0:
            te_per_group[n] = float("inf")
        else:
            te_per_group[n] = p_fp / p_fn

    valid_sp = [v for v in sp_per_group.values() if v == v]
    valid_eo = [v for v in eo_per_group.values() if v == v]
    valid_oae = [v for v in oae_per_group.values() if v == v]
    valid_te = [v for v in te_per_group.values() if v != float("inf") and v == v]

    spd_max_gap = (max(valid_sp) - min(valid_sp)) if len(valid_sp) >= 2 else 0.0
    eod_max_gap = (max(valid_eo) - min(valid_eo)) if len(valid_eo) >= 2 else 0.0
    oaed_max_gap = (max(valid_oae) - min(valid_oae)) if len(valid_oae) >= 2 else 0.0
    ted_max_gap = (max(valid_te) - min(valid_te)) if len(valid_te) >= 2 else 0.0

    return {
        "Statistical Parity Difference": spd_max_gap,
        "Equal Opportunity Difference": eod_max_gap,
        "Overall Accuracy Equality Difference": oaed_max_gap,
        "Treatment Equality Difference": ted_max_gap,
        "multigroup_SP_per_group": sp_per_group,
        "multigroup_EO_per_group": eo_per_group,
        "multigroup_OAE_per_group": oae_per_group,
        "multigroup_TE_per_group": te_per_group,
    }


# ---------- Setting B: multiclass Y, multigroup S (Eqs 13-16) ----------


def multigroup_fairness_setting_b(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    sens_attr: torch.Tensor,
    num_classes: int,
) -> Dict[str, Any]:
    """
    Multiclass Y, multigroup S. labels and predictions in {0, ..., M-1}; sens_attr in {0, ..., N-1}.
    Returns: aggregated max gaps for SP (per-class), EO (per-class), OAE (overall per group), TE (per-class).
    """
    labels = labels.long().ravel()
    predictions = predictions.long().ravel()
    sens_attr = sens_attr.long().ravel()
    uniq_s = torch.unique(sens_attr)
    uniq_s = uniq_s[~torch.isnan(uniq_s.float())]
    M = num_classes

    # SP multiclass: P(ŷ=m | s=n) for each m, n -> max gap over n per m
    sp_per_m = {}
    for m in range(M):
        rates = []
        for n in uniq_s.tolist():
            mask = sens_attr == n
            if mask.sum() == 0:
                continue
            rates.append((predictions[mask] == m).float().mean().item())
        sp_per_m[m] = (max(rates) - min(rates)) if len(rates) >= 2 else 0.0
    spd_multiclass = sum(sp_per_m.values()) / M if M else 0.0

    # EO multiclass: P(ŷ=m | y=m, s=n) for each m, n -> max gap over n per m
    eo_per_m = {}
    for m in range(M):
        rates = []
        for n in uniq_s.tolist():
            mask = (sens_attr == n) & (labels == m)
            if mask.sum() == 0:
                continue
            rates.append((predictions[mask] == m).float().mean().item())
        eo_per_m[m] = (max(rates) - min(rates)) if len(rates) >= 2 else 0.0
    eod_multiclass = sum(eo_per_m.values()) / M if M else 0.0

    # OAE: sum_m P(ŷ=m|y=m,s=n) for each n -> max gap over n
    oae_per_n = {}
    for n in uniq_s.tolist():
        mask = sens_attr == n
        if mask.sum() == 0:
            continue
        acc = (predictions[mask] == labels[mask]).float().mean().item()
        oae_per_n[n] = acc
    valid_oae = [v for v in oae_per_n.values() if v == v]
    oaed_multiclass = (max(valid_oae) - min(valid_oae)) if len(valid_oae) >= 2 else 0.0

    # TE multiclass (Eq. 16): for each m,n
    #   ratio = P(ŷ=m | y≠m, s=n) / P(ŷ≠m | y=m, s=n)
    # We collect all finite ratios and report the max gap.
    te_ratios = []
    for m in range(M):
        for n in uniq_s.tolist():
            mask = sens_attr == n
            if mask.sum() == 0:
                continue
            # y != m slice
            y_ne_m = mask & (labels != m)
            # y == m slice
            y_eq_m = mask & (labels == m)
            num = ((predictions == m) & y_ne_m).sum().item()
            den = ((predictions != m) & y_eq_m).sum().item()
            denom_num = y_ne_m.sum().item()
            denom_den = y_eq_m.sum().item()
            p_num = num / denom_num if denom_num > 0 else float("nan")
            p_den = den / denom_den if denom_den > 0 else float("nan")
            if p_num != p_num or p_den != p_den or p_den == 0:
                continue
            te_ratios.append(p_num / p_den)
    ted_multiclass = (max(te_ratios) - min(te_ratios)) if len(te_ratios) >= 2 else 0.0

    return {
        "Statistical Parity Difference": spd_multiclass,
        "Equal Opportunity Difference": eod_multiclass,
        "Overall Accuracy Equality Difference": oaed_multiclass,
        "Treatment Equality Difference": ted_multiclass,
        "multigroup_SP_per_class": sp_per_m,
        "multigroup_EO_per_class": eo_per_m,
        "multigroup_OAE_per_group": oae_per_n,
    }
