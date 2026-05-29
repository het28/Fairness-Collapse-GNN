"""
Confidence intervals for fairness metrics: bootstrap and (where applicable) analytical.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Fairness metric keys we care about (point estimates; exclude group breakdowns for CI)
FAIRNESS_METRIC_KEYS = [
    "Statistical Parity Difference",
    "Equal Opportunity Difference",
    "Overall Accuracy Equality Difference",
    "Treatment Equality Difference",
    "Accuracy",
]


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Bootstrap percentile CI for a scalar metric.
    Returns (lower, upper).
    """
    rng = np.random.default_rng(random_state)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    boot_means = rng.choice(values, size=(n_bootstrap, n), replace=True).mean(axis=1)
    alpha = 1 - confidence
    low = np.percentile(boot_means, 100 * alpha / 2)
    high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(low), float(high)


def analytical_ci_mean(
    values: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Normal-approximation CI for the mean (for reporting mean ± CI).
    """
    n = len(values)
    if n < 2:
        return float(np.nanmean(values)), float(np.nanmean(values))
    mean = np.nanmean(values)
    se = np.nanstd(values, ddof=1) / np.sqrt(n)
    from scipy import stats
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(mean - h), float(mean + h)


def summarize_runs(
    runs: List[Dict[str, Any]],
    metric_keys: Optional[List[str]] = None,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
) -> Dict[str, Dict[str, Any]]:
    """
    For each (data_name, model, fame, fairness_algo), aggregate runs and compute mean, std, bootstrap CI.
    runs: list of dicts from run_one (each has _data_name, _model, _fame, _fairness_algo, and metric keys).
    Returns nested dict: key = (data_name, model_label) -> { metric -> { mean, std, ci_low, ci_high } }.
    """
    metric_keys = metric_keys or FAIRNESS_METRIC_KEYS

    # Group runs by (data_name, model, fame, fairness_algo, mitigation)
    groups: Dict[tuple, List[Dict]] = {}
    for r in runs:
        if "error" in r:
            continue
        key = (r.get("_data_name"), r.get("_model"), r.get("_fame"), r.get("_fairness_algo"), r.get("_mitigation"))
        groups.setdefault(key, []).append(r)

    def _model_label(model: str, fame: bool, fairness_algo, mitigation) -> str:
        s = f"{model}" + ("+FAME" if fame else "")
        if fairness_algo == "pfr":
            s += "+PFR"
        if mitigation == "option_a":
            s += "+option_a"
        elif mitigation == "option_b":
            s += "+option_b"
        return s

    out = {}
    for (data_name, model, fame, fairness_algo, mitigation), group in groups.items():
        model_label = _model_label(model, fame, fairness_algo, mitigation)
        row_key = (data_name, model_label)
        out[row_key] = {}
        for mk in metric_keys:
            vals = []
            for r in group:
                v = r.get(mk)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
            vals = np.array(vals)
            if len(vals) == 0:
                out[row_key][mk] = {"mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci_low, ci_high = bootstrap_ci(vals, confidence=confidence, n_bootstrap=n_bootstrap)
            out[row_key][mk] = {"mean": mean, "std": std, "ci_low": ci_low, "ci_high": ci_high, "n": len(vals)}
    return out


def ci_overlap(
    low1: float, high1: float,
    low2: float, high2: float,
) -> bool:
    """True if the two intervals overlap."""
    return not (high1 < low2 or high2 < low1)


def overlap_pairs(
    summary: Dict[tuple, Dict[str, Dict[str, Any]]],
    data_name: str,
    metric: str,
) -> List[Tuple[str, str]]:
    """
    For a given dataset and metric, return list of (model_a, model_b) pairs whose CIs overlap.
    """
    keys = [k for k in summary if k[0] == data_name]
    overlapping = []
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            m1, m2 = k1[1], k2[1]
            s1 = summary[k1].get(metric, {})
            s2 = summary[k2].get(metric, {})
            if not s1 or not s2:
                continue
            l1, h1 = s1.get("ci_low"), s1.get("ci_high")
            l2, h2 = s2.get("ci_low"), s2.get("ci_high")
            if np.isnan([l1, h1, l2, h2]).any():
                continue
            if ci_overlap(l1, h1, l2, h2):
                overlapping.append((m1, m2))
    return overlapping
