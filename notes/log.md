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

---

## 2026-11-24 · Days 7–9 (runs, analysis, test set)
Did:        7   40 runs (A, B, C, GA x seeds 1-10), 4 processes in parallel, 00:10-00:57 (47 min wall, 186 min of runs)
            8   src/analysis.py -> results/table2_hypervolume.csv, analysis.json, rq3_settings_spearman.csv,
                feature_frequency_front_C.csv, figures/fig1_fronts.png, fig2_convergence.png, fig3_settings_front_C.png
            9   src/final_eval.py, run ONCE -> results/table1_test.csv, test_eval_runs.csv, val_test_gap.csv
Numbers:    every run used exactly 5,000 evaluations; mean minutes per run A 4.9 · B 4.8 · C 5.1 · GA 3.8
            feasible runs (pre-registered rule): A 7/10 · GA 0/10 · B 10/10 · C 10/10
            feasible runs with F1 >= 0.5 (post hoc): A 0/10 · GA 0/10 · B 10/10 · C 10/10
            GA lowest FPR reached 0.2096
            hypervolume (validation): B 0.9532 +- 0.0401 (median 0.9599) · C 1.1234 +- 0.0163 (median 1.1271)
            RQ1: C's hypervolume was higher than B's in 10 of 10 runs (one-sided Mann-Whitney U = 100, p = 0.0001, A12 = 1.00)
            sensitivity, fronts without F1 < 0.5 (post hoc): C > B in 10 of 10, p = 0.0001, A12 = 1.00 (C 1.0626, B 0.8449)
            RQ2: C's mean hypervolume reaches B's final mean (0.953) at generation 38 = 1,900 of 5,000 evaluations
                 (C mean HV: gen 1 0.164 · gen 10 0.551 · gen 50 1.035 · gen 100 1.123)
            RQ3 Spearman along front C (326 solutions, F1 >= 0.5): max_depth vs ms +0.88, vs F1 +0.48, vs FPR +0.31;
                 min_leaf and normal_weight |rho| <= 0.14
            features front C keeps most: sbytes 77% · dbytes 64% · sttl 55% · ct_dst_src_ltm 43% · smean 42% · dmean 40%
            median-hypervolume runs (Figure 1): B seed 9 · C seed 2
            TEST SET (once; refit on X_tr; mean +- std over 10 runs):
              C knee     F1 0.8727 +- 0.0481 · FPR 0.0229 +- 0.0130 · 4.8 features · 0.040 us · FPR<=1% on test 2/10
              C best F1  F1 0.9182 +- 0.0074 · FPR 0.0274 +- 0.0086 · 11.8 features · 0.057 us · 0/10
              B knee     F1 0.7448 +- 0.1567 · FPR 0.0253 +- 0.0123 · 7.2 features · 0.048 us · 1/10
              B best F1  F1 0.8917 +- 0.0494 · FPR 0.0255 +- 0.0072 · 11.7 features · 0.055 us · 0/10
              A knee     F1 0.8403 · FPR 0.4505 · A best F1 F1 0.8804 · FPR 0.2622
              GA best    F1 0.8358 · FPR 0.4678 · 1 feature
              chi2 k=25  F1 0.8883 · FPR 0.2427 | MI k=30 F1 0.8869 · FPR 0.2425 | all 42 features F1 0.8868 · FPR 0.2458 · 0.277 us
            validation -> test FPR ratio (median): 2.1-3.3x for every method (distribution shift)
Decisions:  post hoc, reported as such: A's only "feasible" solutions were detect-nothing trees (F1 0-0.20), so
            feasibility is also reported for usable detectors (F1 >= 0.5, the selection rule's threshold), plus a
            sensitivity check of the hypervolume test without F1 < 0.5 solutions. The pre-registered numbers are kept.
Problems:   none in the runs; test set used exactly once
Next:       report Sections 6-9 from these numbers; send Dr. May the update

---

## 2026-11-25 · Days 12–13 (checks and package)
Did:        report drafted in Claude Docs (all sections, Tables 1-3, Figures 1-4, equations, references)
            12.2 every number in the report recomputed from the result files and raw data (81 checks)
            13.1 reproducibility check: fresh clone from GitHub, new venv, pip install -r requirements.txt,
                 every script in README order with tiny budgets
            13.4 package: Submission_Ghida_Bahlawan/ with Code_Ghida_Bahlawan.zip and the report PDF (draft)
Numbers:    number check: 81 of 81 match after fixes (6 Table 3 cells were rounded twice from the 4-decimal table;
            now rounded once from test_eval_runs.csv)
            repro check: pip install OK, 7 of 7 scripts exit 0, all outputs present;
            prepared.pkl arrays identical to the original; baseline F1, FPR, precision, recall identical
            code zip: 37 files, 516 KB, no data, .venv, papers or .pkl
            report PDF: 16 pages (A4)
Problems:   the PDF export printed equations as raw LaTeX -> rendered them as images (figures/equations/)
Next:       verify Section 2 against the papers; complete two references; write the AI-use statement;
            trim to the course's page limit; presentation slides; final PDF export
