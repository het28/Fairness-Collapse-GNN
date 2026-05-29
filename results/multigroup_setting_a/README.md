# Multigroup Setting A: Binary Y + Multigroup S

- **Credit**: Y = `NoDefaultNextMonth` (binary); S = `EducationLevel` (0–3).
- **German**: Y = `GoodCustomer_bin` (binary); S = `AgeGroup` quartiles [19–27], (27–33], (33–42], (42–75].
- **Bail**: Y = `RECID` (binary); S = `AgeGroup` quartiles on `AGE` (0–3).

Fairness: multigroup SP/EO/OAE/TE (Eqs. 9–12). Logs and `all_runs.json` from multigroup multi-seed runs go here.

## How to run

From project root (`gnn-fairness-uncertainty/`):

```bash
# Single run
python -m uncertainty_eval.run_multigroup --setting a --dataset credit --model GCN --seed 0

# Multi-seed (e.g. 30 seeds) for one dataset and one model
python -m uncertainty_eval.run_multigroup --setting a --dataset credit --model GCN --n_seeds 30

# Full grid: all 3 datasets x 3 models x 30 seeds (270 runs)
python -m uncertainty_eval.run_multigroup --setting a --datasets credit german bail --models GCN GAT GIN --n_seeds 30
```

Outputs: `all_runs.json`, `multigroup_a.log`.
