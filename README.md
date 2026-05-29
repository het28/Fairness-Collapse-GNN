# Uncertainty in Fairness Metrics for GNN Evaluation

**Positioning:** This is a **survey + reliability audit**, not a new algorithm. We take **existing** GNN fairness work (e.g. [GNNs-FAME](https://github.com/HannanJaved/GNNs-FAME)) and **test** it to show that reported findings can be **unreliable** for specific reasons. The contribution is evidence and guidelines, not a new model.

**Goal:** Systematically quantify uncertainty in fairness metrics for GNN node classification, demonstrate *why* comparisons in the literature are often not statistically well-founded, and provide practical evaluation guidelines.

## Four objectives (what we need to do)

| # | Objective | What we do |
|---|-----------|------------|
| **1** | **Demonstrate metric variance** | Repeated runs → large CIs for DP gap, EO gap, etc. Show that CIs **overlap across models** → claims like “Model A is fairer than B” are often unfounded. |
| **2** | **Quantify drivers of uncertainty** | Vary and measure: **group size imbalance**, **graph homophily**, **evaluation set size**, **seed/sampling**. Report how each affects CI width. |
| **3** | **Benchmark metrics and protocols** | Compare fairness metrics (e.g. DP vs EO) by **uncertainty profile** — which have systematically larger CIs under the same settings? |
| **4** | **Provide practical guidelines** | Recommend: **minimum sample sizes** for reliable measurement, **when to use bootstrapping vs analytical CIs**, **how to report uncertainty** in fairness evaluations. |

## Scope (Option A — Node classification focus)

- **Baseline:** GNNs-FAME (GCN, GAT, FAME, A-FAME) on node classification (German, Credit, Bail). We also added **vanilla GIN** (no FAME) in `GNNs-FAME-main/model.py`.
- **Contribution:** Multi-seed runs → distributions of fairness metrics → confidence intervals → overlap analysis and practical guidelines.
- **Pokec:** Supported as in FAME (pokec-z, pokec-n). Data lives in `GNNs-FAME-main/dataset/pokec/`; if `pokec.zip` is Git LFS, run `git lfs pull` then `python3 scripts/prepare_pokec_data.py`. Uncomment `pokec-z` / `pokec-n` in `config/experiments.yaml` to include them.

## What this repo adds

| Component | Description |
|-----------|-------------|
| **Multi-seed loop** | Run N ≥ 30 seeds per (dataset, model); collect accuracy + fairness metrics. |
| **Confidence intervals** | Bootstrapped and (where applicable) analytical CIs per metric. |
| **Uncertainty profiling** | Mean, std, CI width, overlap between models. |
| **Analysis drivers** | (Future) Vary group imbalance, eval set size, homophily; report CI sensitivity. |
| **Guidelines** | Recommendations: min samples, when to bootstrap, how to report. |

## Quick start (after setup)

```bash
# 1. Prepare datasets (creates german_edges.csv, extracts credit.zip)
python3 scripts/prepare_gnns_fame_data.py

# 2. Quick test: 2 seeds, german only (8 runs)
python3 -m uncertainty_eval.run_multi_seed --n_seeds 2 --datasets german

# 3. Compute CIs and overlap analysis
python3 -m uncertainty_eval.analyze_uncertainty --results_dir outputs/multi_seed

# 4. Full run (30 seeds × 2 datasets × 5 models = 300 runs; models include GIN)
python3 -m uncertainty_eval.run_multi_seed
python3 -m uncertainty_eval.analyze_uncertainty --results_dir outputs/multi_seed

# Append only new model or new dataset (keeps existing runs, adds new keys)
python3 -m uncertainty_eval.run_multi_seed --append --models GIN --datasets credit german --n_seeds 5
python3 -m uncertainty_eval.run_multi_seed --append --models GIN --datasets credit german --split_seeds 0 1 2 3 4 5 6 7 8 9 --n_seeds 5   # append to outputs/split_seed
python3 -m uncertainty_eval.analyze_uncertainty --results_dir outputs/multi_seed   # or outputs/split_seed

# In-processing fairness (PFR = Prejudice Remover): uncomment PFR models in config/experiments.yaml, then run as above.
# Runs will be labeled GCN+PFR, GAT+PFR, GIN+PFR. See docs/FAIRNESS_ALGORITHMS_RESEARCH.md.
```

Results: `summary_ci.json`, `overlap_analysis.json`, `summary_spd.txt`, plus **comparable tables** for papers:
- **results_table.csv** — long format (dataset, model, metric, mean, std, ci_low, ci_high, n)
- **results_summary.csv** — wide format, one row per (dataset, model), columns `mean ± std` per metric
- **results_table_latex.txt** — LaTeX snippet (Acc, SPD, EOD, OAED) for pasting into paper  
- **results_diagnostics.csv** — Test-set diagnostics for reviewers: predicted positive rate (overall and by group), TPR/FPR by group, test-split counts and denominators (exposes constant-classifier collapse and small-denominator TE).
- **results_full_table.csv** — One row per (dataset, model): all metrics + 95% CIs + diagnostic summaries (Excel-ready for paper tables).
- **results_all_runs.csv** — One row per (dataset, model, **seed**): every run so you can compare seeds in Excel. Fairness metrics (Acc, SPD, EOD, OAED, TED) are present for all runs; diagnostic columns are filled only for runs that were executed *after* diagnostics were added to the code (see below).

Experiment parameters and table layout: see `docs/EXPERIMENT_PARAMETERS.md`. Literature: `docs/LITERATURE_REVIEW.md`. **What to look for and what to compare:** `docs/WHAT_TO_LOOK_FOR.md`.

**Device:** The code uses **MPS** (Apple Silicon GPU) when available, else CUDA, else CPU. GNNs-FAME’s `utils.set_device()` was updated to prefer MPS on Mac. **GAT** uses an op (`scatter_reduce`) not yet implemented on MPS; we set `PYTORCH_ENABLE_MPS_FALLBACK=1` so it falls back to CPU for that op (GCN still uses MPS).

### Why were many metrics 0.0?

Two causes (both addressed):

1. **Full-graph fairness** — The GNNs-FAME baseline computes fairness on **all nodes** (train+val+test). We now **recompute fairness on the test set only** in `run_one` (see `fairness_test_only.py`), so metrics reflect held-out evaluation.
2. **Model collapse** — With small test sets and high variance, the model often **predicts a constant class** on the test set (all 0 or all 1). Then P(ŷ=1) is 0 or 1 for both sensitive groups → SPD = 0, EOD = 0. We set default `epochs: 50`; with many seeds you’ll get both collapsed runs (0) and mixed runs (non-zero SPD). That variance is exactly what the uncertainty study measures.

## Setup

1. **Clone GNNs-FAME** (required for running experiments):

   ```bash
   ./scripts/clone_gnns_fame.sh
   ```

   Or manually:

   ```bash
   git clone https://github.com/HannanJaved/GNNs-FAME.git GNNs-FAME
   ```

2. **Install dependencies** (GNNs-FAME + this extension):

   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have PyTorch, PyTorch Geometric, and GNNs-FAME dependencies (see GNNs-FAME README: preprocess data, set_uid, etc.).

3. **GNNs-FAME location:** The code looks for the repo in `GNNs-FAME-main/` or `GNNs-FAME/` inside this project (e.g. after downloading from GitHub you may have `GNNs-FAME-main`). To use a different path:

   ```bash
   export GNNs_FAME_ROOT=/path/to/GNNs-FAME-main
   ```

## Usage

- **Run multi-seed experiments** (writes per-seed metrics and summary):

  ```bash
  python -m uncertainty_eval.run_multi_seed --config config/experiments.yaml
  ```

- **Compute CIs and overlap analysis** from collected runs:

  ```bash
  python -m uncertainty_eval.analyze_uncertainty --results_dir outputs/multi_seed
  ```

- **Plot stability and overlap** (optional):

  ```bash
  python -m uncertainty_eval.plot_stability --results_dir outputs/multi_seed --out_dir figures
  ```

## Outputs (for the paper)

- **Stability plots:** Fairness score distributions across seeds; CI bars per model.
- **Overlap analysis:** Pairs (Model A, Model B) with overlapping CIs → “no reliable difference”.
- **Drivers heatmaps:** (Planned) CI width vs group imbalance / eval size.
- **Guidelines table:** Recommended min samples, bootstrapping, reporting.

## Where run outputs go

All run outputs (stratified split, residual GNN, PFR, Option A, Option B, multi_seed, split_seed, etc.) are written under **`outputs/`**, not `results/`. This keeps experiment outputs in one place; you can use `results/` for something else or leave it for legacy. Set `output_base` in `config/experiments.yaml` to change the folder (default: `outputs`).

## Repository layout

```
gnn-fairness-uncertainty/
├── README.md
├── requirements.txt
├── config/
│   └── experiments.yaml      # datasets, models, n_seeds, output_base (default: outputs)
├── outputs/                  # run outputs: stratified_split, residual_gnn, option_a_stability, etc.
├── scripts/
│   └── clone_gnns_fame.sh
└── uncertainty_eval/
    ├── __init__.py
    ├── run_one.py            # single run with seed (imports GNNs-FAME)
    ├── run_multi_seed.py     # loop over seeds, collect metrics
    ├── confidence_intervals.py
    ├── analyze_uncertainty.py
    └── plot_stability.py
```

## References

- GNNs-FAME: [HannanJaved/GNNs-FAME](https://github.com/HannanJaved/GNNs-FAME) — Purificato et al., UMAP '25.
