#!/usr/bin/env bash
# Option A: stability regularization (Huang–Vishnoi style).
# Results go to outputs/option_a_stability/.
# Use same seeds as stratified (5,10,...,50) for comparable results.

set -e
cd "$(dirname "$0")/.."

python3 -m uncertainty_eval.run_multi_seed \
  --option_a \
  --stratify_split \
  --seeds 5 10 15 20 25 30 35 40 45 50 \
  --stability_lambda 0.5

echo "Done. Analyze with: python3 -m uncertainty_eval.analyze_uncertainty --results_dir outputs/option_a_stability"
