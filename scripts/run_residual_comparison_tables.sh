#!/usr/bin/env bash
# Run comparison tables: Residual GNN baseline vs PFR, Option A, Option B.
# Use after residual_gnn, residual_gnn_pfr, residual_gnn_option_a, residual_gnn_option_b runs are done.
# Requires: outputs/residual_gnn/all_runs.json, outputs/residual_gnn_pfr/all_runs.json,
#           outputs/residual_gnn_option_a/all_runs.json, outputs/residual_gnn_option_b/all_runs.json

cd "$(dirname "$0")/.."
OUT="${1:-outputs/residual_comparison_tables.txt}"

echo "Residual GNN comparison tables" > "$OUT"
echo "==============================" >> "$OUT"
echo "Baseline = outputs/residual_gnn, Mitigation = residual_gnn_pfr | option_a | option_b" >> "$OUT"
echo "Seeds: 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50" >> "$OUT"
echo "" >> "$OUT"

for dataset in credit german bail pokec-z pokec-n; do
  for mitigation in pfr option_a option_b; do
    echo "--- Dataset: $dataset | Mitigation: $mitigation ---" >> "$OUT"
    if python3 scripts/table_baseline_vs_mitigation.py --residual-runs --dataset "$dataset" --mitigation "$mitigation" >> "$OUT" 2>&1; then
      echo "(ok)" >> "$OUT"
    else
      echo "(skip or no data - e.g. option_b not finished yet)" >> "$OUT"
    fi
    echo "" >> "$OUT"
  done
done

echo "Tables written to $OUT"
