#!/usr/bin/env bash
# Run the same pipeline as Credit/German/Bail for Pokec: stratified, residual, residual+PFR, residual+Option A, residual+Option B.
# Requires Pokec data: run "git lfs pull" and "python3 scripts/prepare_pokec_data.py" first.
# Uses --datasets pokec-z pokec-n and --append to add Pokec runs to existing all_runs.json in each results dir.

set -e
cd "$(dirname "$0")/.."

SEEDS="1 5 10 15 20 25 30 35 40 45 50"
DATASETS="pokec-z pokec-n"

echo "=== 1. Stratified split baseline (pokec-z, pokec-n) ==="
python3 -m uncertainty_eval.run_multi_seed \
  --stratify_split \
  --datasets $DATASETS \
  --seeds $SEEDS \
  --append

echo "=== 2. Residual GNN baseline (pokec-z, pokec-n) ==="
python3 -m uncertainty_eval.run_multi_seed \
  --residual \
  --datasets $DATASETS \
  --seeds $SEEDS \
  --append

echo "=== 3. Residual + PFR (pokec-z, pokec-n) ==="
python3 -m uncertainty_eval.run_multi_seed \
  --residual \
  --fairness_algo pfr \
  --datasets $DATASETS \
  --results_dir outputs/residual_gnn_pfr \
  --seeds $SEEDS \
  --append

echo "=== 4. Residual + Option A (pokec-z, pokec-n) ==="
python3 -m uncertainty_eval.run_multi_seed \
  --residual \
  --option_a \
  --datasets $DATASETS \
  --results_dir outputs/residual_gnn_option_a \
  --seeds $SEEDS \
  --append

echo "=== 5. Residual + Option B (pokec-z, pokec-n) ==="
python3 -m uncertainty_eval.run_multi_seed \
  --residual \
  --option_b \
  --datasets $DATASETS \
  --results_dir outputs/residual_gnn_option_b \
  --seeds $SEEDS \
  --append

echo "=== 6. Regenerate CSVs (results_*.csv, summary_ci.json, etc.) for each results dir ==="
for dir in outputs/stratified_split outputs/residual_gnn outputs/residual_gnn_pfr outputs/residual_gnn_option_a outputs/residual_gnn_option_b; do
  echo "  Analyzing $dir ..."
  python3 -m uncertainty_eval.analyze_uncertainty --results_dir "$dir" || true
done

echo "Done. CSVs saved in each outputs/*/ directory (results_all_runs.csv, results_table.csv, results_summary.csv, etc.)."
echo "Print comparison tables with:"
echo "  python3 scripts/table_baseline_vs_mitigation.py --residual-runs --dataset pokec-z --mitigation pfr"
echo "  python3 scripts/table_baseline_vs_mitigation.py --residual-runs --dataset pokec-z --mitigation option_a"
echo "  python3 scripts/table_baseline_vs_mitigation.py --residual-runs --dataset pokec-z --mitigation option_b"
echo "  (same for pokec-n)"
