#!/usr/bin/env bash
# Option B: NIFTY-style (augmentation + similarity).
# Results go to outputs/option_b_nifty/.
# Use same seeds as stratified (5,10,...,50) for comparable results.

set -e
cd "$(dirname "$0")/.."

python3 -m uncertainty_eval.run_multi_seed \
  --option_b \
  --stratify_split \
  --seeds 5 10 15 20 25 30 35 40 45 50 \
  --nifty_sim_coeff 0.5 \
  --drop_edge_rate 0.1 \
  --drop_feature_rate 0.1

echo "Done. Analyze with: python3 -m uncertainty_eval.analyze_uncertainty --results_dir outputs/option_b_nifty"
