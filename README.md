# Joint feature and hyperparameter optimization with NSGA-II for lightweight intrusion detection

Course project (Emerging Topics in AI, Option 2, topic #12 Green AI Optimization). A decision-tree intrusion
detector on UNSW-NB15 is optimized for three objectives at once (1 − F1, number of features, inference time per
connection) under a false-alarm constraint (FPR ≤ 1% on validation). Four methods get the same budget of 5,000
evaluations and the same seeds (1–10):

| Method | What it searches | FPR limit |
|---|---|---|
| A, features only | 42 feature genes, default tree (NSGA-II) | none |
| B, sequential | A for 80 generations, then Optuna tunes 4 tree settings per subset (1,000 trials) | Optuna step |
| C, joint | 42 feature genes + max_depth, min_samples_leaf, criterion, normal_weight (NSGA-II) | yes |
| GA | 42 feature genes, default tree, one weighted score (single objective) | none |

Baselines: the all-features tree, and χ² / mutual-information top-k (k = 5…30).

## Setup

```
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` from the official UNSW-NB15 page
(research.unsw.edu.au/projects/unsw-nb15-dataset) into `data/raw/`. The data is not in this repository.

## Run, in this order

| Step | Script | Output |
|---|---|---|
| 1 | `src/health_checks.py` | the five data checks (leaks, duplicates, conflicts, unseen categories) |
| 2 | `src/prepare.py` | `data/prepared.pkl`: duplicates removed, one-hot, 80/20 split, scaling, 20k sample |
| 3 | `src/baselines.py` | `results/baselines_val.csv`, `figures/baselines_f1_vs_k.png` |
| 4 | `src/sanity_checks.py` | correctness tests for the problem definition |
| 5 | `src/experiments.py 1 10` | `results/{A,B,C,GA}_seed{1..10}.pkl` (start it in several terminals to run in parallel) |
| 6 | `src/analysis.py` | Table 2 (`results/table2_hypervolume.csv`), Figures 1–3, `results/analysis.json` |
| 7 | `src/final_eval.py` | Table 1 (`results/table1_test.csv`) and the validation–test gap; the only script that reads the test set |

A quick end-to-end check with tiny budgets (seconds per run, nothing touches the test set):

```
python src/experiments.py 1 3 --tiny
python src/analysis.py --results results/tiny --expect 3
python src/final_eval.py --results results/tiny --dry
```

## Rules the code follows

- The official test file is used once, by `final_eval.py`, after every choice was made on validation data.
- Scaling, χ² and mutual information are fitted on training data only.
- Every method scores candidates with the same `score()` in `src/scoring.py` and uses exactly 5,000 evaluations.
- Hypervolume uses one shared normalization for B and C and a reference point of 1.1 on each axis.
- Inference time is an objective, so runs with the same seed are not bit-for-bit identical (timing noise).

`notes/log.md` is the research log: every number, decision and problem, in order.
