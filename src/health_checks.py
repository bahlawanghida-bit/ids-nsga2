"""Day 2 · step 2.2 — the five health checks. Reads data/raw, changes nothing on disk."""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
train = pd.read_csv(DATA / "UNSW_NB15_training-set.csv")
test  = pd.read_csv(DATA / "UNSW_NB15_testing-set.csv")

features = [c for c in train.columns if c not in ("id", "attack_cat", "label")]
assert len(features) == 42

# 1. Leak: can the row id alone predict the label?
try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import f1_score
    t = DecisionTreeClassifier(max_depth=2, random_state=0)
    t.fit(train[["id"]], train["label"])
    print("1. id-only F1:", round(f1_score(train["label"], t.predict(train[["id"]])), 3))   # ~0.967 -> drop id
except ImportError:
    print("1. id-only F1: skipped (scikit-learn not installed)")

# 2. Label leak: attack_cat names the attack, so it fully determines label
leak = (train["attack_cat"] == "Normal").astype(int).eq(1 - train["label"]).all()
print("2. attack_cat fully determines label:", leak)                                   # True -> drop attack_cat

# 3. Duplicates on 42 features + label
dupes = train.duplicated(subset=features + ["label"]).sum()
print("3. duplicate rows:", dupes, f"({dupes / len(train):.1%})")                       # ~74,072

# 4. Conflicts: identical feature rows with different labels
conflicts = (train.groupby(features, dropna=False)["label"].nunique() > 1).sum()
print("4. conflicting groups:", conflicts)                                             # ~229

# 5. Categories in test that never appear in train
for col in ["proto", "service", "state"]:
    unseen = set(test[col]) - set(train[col])
    print(f"5. {col} test-only values:", unseen or "none")                              # state: ACC, CLO
