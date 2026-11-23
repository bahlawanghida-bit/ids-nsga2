# Research log: ids-nsga2

Write one entry per day, at the end of each work session. The report is written from this file,
so copy numbers from the screen, never from memory.

Template (copy for each day):

## YYYY-MM-DD · Day N
Did:        
Numbers:    
Decisions:  
Problems:   
Next:       

---

## 2026-11-08 · Day 1
Did:        created project folder (data/raw, src, results, figures, report, notes); created .venv; installed packages; saved requirements.txt; added .gitignore; ran src/check_pymoo.py; downloaded UNSW-NB15 train/test CSVs
Numbers:    Python 3.13.13 · pymoo 0.6.2 · scikit-learn 1.9.1 · Optuna 5.0.0 · pandas 3.1.1
            ZDT1 check (NSGA-II, pop 50, 100 gen, seed 1): 50 non-dominated points
            train shape (175341, 45), attack share 68.1% · test shape (82332, 45), attack share 55.1%
            (official CSVs from research.unsw.edu.au, in data/raw; test set now locked until Day 9)
Decisions:  
Problems:   
Next:       

---

## 2026-11-09 · Day 2
Did:        2.1 loaded train/test from data/raw, inspected columns, class balance, attack_cat (src/explore.py)
            2.2 ran the five health checks (src/health_checks.py)
            fixed paths (scripts now build them from __file__); added .vscode/settings.json so VS Code uses .venv
Numbers:    train 175,341 rows (68.1% attack) · test 82,332 (55.1%) · Worms only 130 rows
            categorical values: proto 133 / service 13 / state 9 (train)
            id-only tree F1 = 0.967 -> id leaks the label
            attack_cat fully determines label -> label leak
            74,072 duplicate rows on 42 features + label (42.2% of train)
            229 groups of identical rows with conflicting labels
            state ACC, CLO appear only in test (proto, service: none)
Decisions:  drop id and attack_cat (leaks)
            Decision 1 (duplicates): pending, step 2.3
Problems:   relative path data/... failed when run from src/ -> Path(__file__)-based paths
            VS Code ran global Python 3.13 (no scikit-learn) instead of .venv -> settings.json + run from (.venv) terminal
Next:       2.3 decide on duplicates, then 2.4 encode/split/scale

---

## 2026-11-23 · Day 2 (continued)
Did:        2.3 Decision 1 (duplicates); 2.4 encode, split, scale, sample; 2.5 saved data/prepared.pkl (src/prepare.py)
            2.6 normal_weight probe on validation (src/probe_weight.py); Decision 2
            removed paper PDFs from git history, added papers/ to .gitignore; public GitHub repo ids-nsga2 created
Numbers:    after removing duplicates: 101,269 training rows, attack share 0.488
            194 columns after one-hot (proto 133 / service 13 / state 9 + 39 numeric); 42 gene groups, every column in exactly one
            X_tr 81,015 · X_val 20,254 · X_fit 20,000 · X_test 82,332 (194 columns each)
            attack share y_tr 0.4876 · y_val 0.4876 · y_fit 0.4876 -> stratification worked
            scaled range X_tr [0, 1] · X_val max 1.18 · X_test max 1.93 -> scaler fit on X_tr only
            5 test rows have state ACC/CLO (unseen in train) -> all-zero state columns
            prepared.pkl 318 MB (not in git)
            default tree, all features, validation: F1 0.8969 · FPR 0.0934 · recall 0.8928 (969 false alarms of 10,378 normal)
            probe (max_depth 12, min_leaf 10), normal weight w:
              w=1  FPR 0.1177  F1 0.9084 | w=5  FPR 0.0286  F1 0.8377
              w=20 FPR 0.0121  F1 0.7919 | w=50 FPR 0.0109  F1 0.7771
Decisions:  Decision 1: removed 74,072 exact duplicates from the training file BEFORE splitting, so validation
            measures rows the tree has not memorized; official test file left unchanged.
            229 conflicting groups remain (one row per label is kept) -> mention as unavoidable errors in the report
            Decision 2: add normal_weight gene (class weight of normal traffic, 1-50) to methods B and C, because
            FPR <= 1% is out of reach with default settings (9.3%) and with the weight alone (best 1.09% at w=50);
            method A unchanged (features only, default tree, no FPR limit)
            paper PDFs kept out of the public repo (copyright)
Problems:   git filter-branch also deleted papers/ from disk -> restored from backup (hashes checked)
            GitHub connection unreliable -> repo create and push needed retries
            git push run from AI-Research instead of ids-nsga2 -> "not a git repository"; run git inside ids-nsga2
Next:       send Dr. May the two-sentence update; Day 3 step 3.1: write score() in src/scoring.py

---

## 2026-11-24 · Days 3–6 (implementation, before the main runs)
Did:        3.1 score() in src/scoring.py (X arrays loaded as float32: trees cast to float32 anyway, results identical)
            3.2 filter baselines (src/baselines.py) -> results/baselines_val.csv, figures/baselines_f1_vs_k.png
            4.1–4.4, 5, 6.1–6.4 IDSProblem, FrontLog, run_nsga2, GA baseline, method B (src/problem.py), runner (src/experiments.py)
            correctness tests (src/sanity_checks.py) and a dry run with tiny budgets: all passed
Numbers:    score() on all 42 features, default tree (validation): F1 0.8969 · FPR 0.0934 (= Day 2)
            one evaluation ~0.20 s on this laptop -> one 5,000-evaluation run ~17 min
            baselines (validation): best chi2 F1 0.8951 (k=25, FPR 0.0928) · best MI F1 0.8978 (k=30, FPR 0.0889)
            no baseline meets FPR <= 1% (lowest 0.0889)
            Test 1: IDSProblem, all genes on, tune=False: F1 0.896857 = baseline exactly
            Test 2: no features -> F [1, 0, 1e6], G [1, 1] (violated)
            tiny C run (pop 10, 3 gen): feasible front, settings inside their ranges, lowest FPR 0.0078
            dry run: every method used exactly its budget (30 = 10 x 3)
Decisions:  B's Optuna step gets the FPR limit as a constraint (trial.set_constraint), so the sequential competitor
            is not handicapped against C; the deck's code tuned without it. Scoring is still the same IDSProblem.
            budget fixed at 5,000 evaluations for every method, whatever the convergence plot shows (equal budgets)
            machine: AMD Ryzen 5 5600H (6 cores / 12 threads), 15.3 GB RAM, Windows 11 Home; 4 runs in parallel
Problems:   Optuna 5.0 deprecates constraints_func -> trial.set_constraint
            same seed can give a slightly different front: time is an objective and timing is noisy (limitation)

### Analysis rules, fixed BEFORE any main result exists
Feasible run:  at least one final solution with validation FPR <= 1%.
Hypervolume:   B and C final fronts in (1 - F1, features, ms); ONE min-max normalization over all B and C fronts
               together; reference point 1.1 on each axis; an empty front has hypervolume 0.
Test (RQ1):    Mann-Whitney U, one-sided (C > B), alpha 0.05; effect size A12 = U / (10 x 10). One test, no Holm.
RQ2:           C's hypervolume per generation (same normalization), mean +- 1 std over 10 runs; B's final mean
               hypervolume as a line; first generation where C's mean reaches it (x 50 = evaluations).
Selection:     per run, from the validation front only: the knee (solutions with F1 >= 0.5, each objective scaled
               to 0-1 over that front, smallest distance to the ideal corner) and the best-F1 solution.
               A: its own knee and best-F1 · GA: its single best · chi2 and MI: best k by validation F1 · all-features tree.
Test set:      each chosen detector is refit on X_tr, y_tr and evaluated ONCE on X_test; time = median of 10
               predictions of the whole test set, one process, per connection.
Figures:       Figure 1 uses, for B and C, the run whose hypervolume is closest to that method's median.
