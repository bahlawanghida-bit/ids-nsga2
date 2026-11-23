"""Day 2 · step 2.6 — can a heavier weight on normal traffic push FPR toward 1%? Validation data only."""
from pathlib import Path
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score

PREPARED = Path(__file__).resolve().parent.parent / "data" / "prepared.pkl"
d = pickle.load(open(PREPARED, "rb"))
X_fit, y_fit, X_val, y_val = d["X_fit"], d["y_fit"], d["X_val"], d["y_val"]

def report(name, tree):
    pred = tree.fit(X_fit, y_fit).predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    print(f"{name:22}  FPR = {fp / (fp + tn):.4f}   recall = {tp / (tp + fn):.4f}   F1 = {f1_score(y_val, pred):.4f}"
          f"   false alarms = {fp:,} of {fp + tn:,} normal")

report("default tree", DecisionTreeClassifier(random_state=0))                               # plan: FPR ~0.09
for w in [1, 5, 20, 50]:
    report(f"normal weight {w:>2}", DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=10, class_weight={0: w, 1: 1}, random_state=0))
# plan:  w=1 -> FPR 0.118, F1 0.908 | w=5 -> 0.029, 0.838 | w=20 -> 0.012, 0.792 | w=50 -> 0.011, 0.777
