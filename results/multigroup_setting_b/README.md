# Multigroup Setting B: Multiclass Y + Multigroup S

- **Credit**: Y = risk tiers from `TotalMonthsOverdue` (4 classes); S = `EducationLevel` (0–3).
- **German**: Y = risk tiers from `LoanAmount` (3 classes); S = `AgeGroup` quartiles.
- **Bail**: Y = risk tiers from `PRIORS` (4 classes); S = `AgeGroup` quartiles on `AGE`.

Fairness: multiclass+multigroup SP/EO/OAE/TE (Eqs. 13–16). Logs and `all_runs.json` from multigroup multi-seed runs go here.

## How to run

From project root (`gnn-fairness-uncertainty/`):

```bash
# Single run
python -m uncertainty_eval.run_multigroup --setting b --dataset credit --model GCN --seed 0

# Multi-seed for one dataset and one model
python -m uncertainty_eval.run_multigroup --setting b --dataset bail --model GAT --n_seeds 30

# Full grid: all 3 datasets x 3 models x 30 seeds
python -m uncertainty_eval.run_multigroup --setting b --datasets credit german bail --models GCN GAT GIN --n_seeds 30
```

Outputs: `all_runs.json`, `multigroup_b.log`.
