"""Day 9 — choose detectors on validation only, refit each on X_tr, evaluate it ONCE on the locked test set.
Writes Table 1 and the validation-test gap. Run it once, alone, after analysis.py.

    python src/final_eval.py          # the real, one-time test-set evaluation
    python src/final_eval.py --dry    # code check: validation data stands in for the test set
"""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score
from scoring import load_prepared, RESULTS
from results_io import load_runs, decode, knee

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, default=RESULTS)
parser.add_argument("--dry", action="store_true")
args = parser.parse_args()
out_dir = args.results / "dry" if args.dry else args.results
out_dir.mkdir(parents=True, exist_ok=True)

d = load_prepared()
groups, X_tr, y_tr = d["groups"], d["X_tr"], d["y_tr"]
X_eval, y_eval = (d["X_val"], d["y_val"]) if args.dry else (d["X_test"], d["y_test"])


def evaluate(cols, settings):
    tree = DecisionTreeClassifier(random_state=0, **settings).fit(X_tr[:, cols], y_tr)
    Xe = np.ascontiguousarray(X_eval[:, cols])
    times = []
    for _ in range(10):                                            # median of 10 predictions
        start = time.perf_counter()
        pred = tree.predict(Xe)
        times.append(time.perf_counter() - start)
    tn, fp, fn, tp = confusion_matrix(y_eval, pred, labels=[0, 1]).ravel()
    return {"f1": f1_score(y_eval, pred, zero_division=0), "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn), "fpr": fp / (fp + tn), "ms": float(np.median(times)) * 1000 / len(y_eval)}


# 9.1 Choose on validation only
picks = []
runs = load_runs(args.results)
for m in ["A", "B", "C"]:
    for r in runs[m]:
        F = r["F"]
        if len(F) == 0:
            continue
        for pick, i in [("knee", knee(F)), ("best F1", int(np.argmin(F[:, 0])))]:
            if i is not None:
                picks.append({"method": m, "pick": pick, "seed": r["seed"], "x": r["X"][i],
                              "val_f1": 1 - F[i, 0], "val_fpr": float(r["fpr"][i])})
for r in runs["GA"]:
    picks.append({"method": "GA", "pick": "best", "seed": r["seed"], "x": r["X"][0],
                  "val_f1": r["val"]["f1"], "val_fpr": r["val"]["fpr"]})
base = pd.read_csv(RESULTS / "baselines_val.csv")
for name in ["chi2", "MI", "all features"]:
    t = base[base.method == name]
    b = t.loc[t.f1.idxmax()]                                       # best k on validation
    genes = [int(g) for g in str(b.genes).split()]
    picks.append({"method": name, "pick": f"k={b.k}", "seed": None,
                  "x": {f"f{i}": i in genes for i in range(42)}, "val_f1": b.f1, "val_fpr": b.fpr})

# 9.2–9.3 Refit on X_tr, evaluate once
rows = []
for p in picks:
    on, cols, settings = decode(p["x"], groups)
    r = evaluate(cols, settings)
    rows.append({"method": p["method"], "pick": p["pick"], "seed": p["seed"], "features": len(on),
                 **{f"test_{k}": v for k, v in r.items()}, "val_f1": p["val_f1"], "val_fpr": p["val_fpr"],
                 "max_depth": settings.get("max_depth"), "min_leaf": settings.get("min_samples_leaf"),
                 "criterion": settings.get("criterion"),
                 "normal_weight": settings["class_weight"][0] if settings else None,
                 "genes": " ".join(map(str, on))})
raw = pd.DataFrame(rows)
raw.to_csv(out_dir / "test_eval_runs.csv", index=False)

# Table 1: mean ± std over the 10 runs for A, B, C and GA; single rows for the baselines
order = [("C", "knee"), ("C", "best F1"), ("B", "knee"), ("B", "best F1"), ("A", "knee"), ("A", "best F1"),
         ("GA", "best")] + [(n, None) for n in ["chi2", "MI", "all features"]]
cols = ["test_f1", "test_precision", "test_recall", "test_fpr", "features", "test_ms"]
table = []
for m, pick in order:
    g = raw[(raw.method == m) & ((raw.pick == pick) if pick else True)]
    row = {"method": m, "pick": pick or g.pick.iloc[0], "n": len(g),
           "meets FPR<=1% on test": f"{int((g.test_fpr <= 0.01).sum())} of {len(g)}"}
    for c in cols:
        v = g[c] * (1000 if c == "test_ms" else 1)                 # time in microseconds
        name = c.replace("test_ms", "µs").replace("test_", "")
        row[name] = f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(g) > 1 else f"{v.iloc[0]:.4f}"
    table.append(row)
table = pd.DataFrame(table)
table.to_csv(out_dir / "table1_test.csv", index=False)
print(("DRY RUN (validation as stand-in)\n" if args.dry else "") + table.to_string(index=False))

# 9.5 The validation-test gap in false alarms
gap = raw.assign(ratio=raw.test_fpr / raw.val_fpr.where(raw.val_fpr > 0))
gap = gap.groupby(["method", "pick"], sort=False).agg(val_fpr=("val_fpr", "mean"), test_fpr=("test_fpr", "mean"),
                                                       ratio_median=("ratio", "median"))
gap.to_csv(out_dir / "val_test_gap.csv")
print("\nValidation vs test FPR (mean), and median test/validation ratio:\n", gap.round(4).to_string())
