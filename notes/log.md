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
