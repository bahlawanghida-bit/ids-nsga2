"""Day 3 · step 3.1 — the one scoring function every method and baseline uses."""
from pathlib import Path
import pickle
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def load_prepared():
    """prepared.pkl with X arrays as float32: scikit-learn trees cast to float32 anyway,
    so tree results are identical and no copy is made at every fit and predict."""
    with open(ROOT / "data" / "prepared.pkl", "rb") as f:
        d = pickle.load(f)
    for k in ("X_fit", "X_val", "X_tr", "X_test"):
        d[k] = np.ascontiguousarray(d[k], dtype=np.float32)
    return d


def score(cols, settings, X_a, y_a, X_b, y_b, repeats=3):
    """Fit a tree on (X_a, y_a) using columns `cols`, score it on (X_b, y_b)."""
    tree = DecisionTreeClassifier(random_state=0, **settings)
    tree.fit(X_a[:, cols], y_a)
    Xb = np.ascontiguousarray(X_b[:, cols])        # slice once, so only prediction is timed
    best = float("inf")
    for _ in range(repeats):                       # keep the fastest of `repeats`
        start = time.perf_counter()
        pred = tree.predict(Xb)
        best = min(best, time.perf_counter() - start)
    tn, fp, fn, tp = confusion_matrix(y_b, pred, labels=[0, 1]).ravel()
    return {"f1": f1_score(y_b, pred, zero_division=0),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn),
            "fpr": fp / (fp + tn),
            "ms": best * 1000 / len(y_b)}          # per connection
