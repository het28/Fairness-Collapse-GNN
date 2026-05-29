# Fairness-Collapse-GNN

Multi-seed evaluation of fairness metrics for graph neural networks on node classification. Built on [GNNs-FAME](https://github.com/HannanJaved/GNNs-FAME).

**Focus:** Fairness scores (SPD, EOD, OAED, TED) vary widely across random seeds. Many near-zero gaps come from **prediction collapse** (near-constant outputs), not equitable behavior. This repo provides the code and summarized results for binary and multigroup/multiclass settings.

## Datasets

| Dataset | Binary target | Binary sensitive | Multigroup / multiclass (Setting B) |
|---------|---------------|------------------|-------------------------------------|
| **Bail** | `RECID` | `WHITE` (0=non-White, 1=White) | Y: `PRIORS` bins; S: `AGE` quartiles |
| **Credit** | `NoDefaultNextMonth` | `Age` (0/1) | Y: `TotalMonthsOverdue` bins; S: `EducationLevel` (4 groups) |
| **German** | `GoodCustomer` | `Gender` (0=Male, 1=Female) | Y: `LoanAmount` tertiles; S: `Age` quartiles |

**Models:** GCN, GAT, GIN (30 seeds per configuration).

Datasets are **not** included. Prepare them with:

```bash
pip install -r requirements.txt
python3 scripts/prepare_gnns_fame_data.py
```

Place data under `GNNs-FAME-main/dataset/` (`bail/`, `credit/`, `german/`).

## Quick start

```bash
# Binary fairness, multi-seed (writes to results/multi_seed by default)
python3 -m uncertainty_eval.run_multi_seed --n_seeds 30 --datasets bail credit german

python3 -m uncertainty_eval.analyze_uncertainty --results_dir results/multi_seed

# Multigroup Setting A (binary Y + multigroup S)
python3 -m uncertainty_eval.run_multigroup --setting a --n_seeds 30

# Multigroup Setting B (multiclass Y + multigroup S)
python3 -m uncertainty_eval.run_multigroup --setting b --n_seeds 30

python3 -m uncertainty_eval.analyze_uncertainty --results_dir results/multigroup_setting_a
python3 -m uncertainty_eval.analyze_uncertainty --results_dir results/multigroup_setting_b
```

On Apple Silicon, PyTorch may use MPS; GAT can fall back to CPU for some ops (`PYTORCH_ENABLE_MPS_FALLBACK=1`).

## Results in this repo

Precomputed summaries (CSV/JSON):

| Folder | Description |
|--------|-------------|
| `results/multi_seed/` | Binary Y + binary S |
| `results/multigroup_setting_a/` | Binary Y + multigroup S |
| `results/multigroup_setting_b/` | Multiclass Y + multigroup S |

Key files: `results_summary.csv`, `results_all_runs.csv`, `results_full_table.csv`, `summary_ci.json`, `overlap_analysis.json`.

Fairness is computed on the **test set only** (`uncertainty_eval/fairness_test_only.py`).

## Layout

```
├── config/experiments.yaml
├── uncertainty_eval/          # run + analyze + multigroup metrics
├── scripts/                   # data prep and analysis helpers
├── GNNs-FAME-main/            # baseline GNN code (no dataset/ in git)
└── results/                   # summarized experiment outputs
```

## Reference

Purificato et al., GNNs-FAME — [github.com/HannanJaved/GNNs-FAME](https://github.com/HannanJaved/GNNs-FAME)
