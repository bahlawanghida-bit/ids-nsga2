"""Day 2 · steps 2.3–2.5 — build the arrays every method will use."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
train = pd.read_csv(DATA / "UNSW_NB15_training-set.csv")
test  = pd.read_csv(DATA / "UNSW_NB15_testing-set.csv")

CAT = ["proto", "service", "state"]                                  # the 3 text features
features = [c for c in train.columns if c not in ("id", "attack_cat", "label")]
assert len(features) == 42

# 2.3 Decision 1: remove exact duplicates from the training file BEFORE splitting
before = len(train)
train = train.drop_duplicates(subset=features + ["label"]).reset_index(drop=True)
print(f"2.3 removed {before - len(train):,} duplicates -> {len(train):,} rows, "
      f"attack share {train['label'].mean():.3f}")                                    # 101,269 · ~0.488

# 2.4a Encode: one-hot the text features (proto=tcp -> proto_tcp = 1)
def encode(df):
    X = pd.get_dummies(df[features], columns=CAT, dtype=float)
    return X, df["label"].to_numpy()

X_all, y_all = encode(train)
X_test, y_test = encode(test)
X_test = X_test.reindex(columns=X_all.columns, fill_value=0)        # test gets exactly the training columns
cols = list(X_all.columns)

def columns_of(f):                                                  # gene -> its column numbers
    if f in CAT:
        return [j for j, c in enumerate(cols) if c.startswith(f + "_")]
    return [cols.index(f)]

groups = [columns_of(f) for f in features]                          # one group per gene, 42 in total
assert sorted(j for g in groups for j in g) == list(range(X_all.shape[1]))   # every column in exactly one gene
print("2.4a columns after one-hot:", X_all.shape[1],
      {f: len(columns_of(f)) for f in CAT})                                           # 194 · 133/13/9
print("     test rows with a state never seen in training:",
      test["state"].isin(set(test["state"]) - set(train["state"])).sum())

# 2.4b Split: 80% trains trees, 20% validation scores candidates; stratified keeps the attack share equal
X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=0)

# 2.4c Scale: learn min/max on X_tr only, then apply the same numbers to validation and test
scaler = MinMaxScaler().fit(X_tr)
X_tr, X_val, X_test = (scaler.transform(d) for d in (X_tr, X_val, X_test))

# 2.4d Sample: 20,000 stratified rows of X_tr, so each NSGA-II evaluation fits fast
X_fit, _, y_fit, _ = train_test_split(
    X_tr, y_tr, train_size=20000, stratify=y_tr, random_state=0)

print("2.4b shapes  X_tr", X_tr.shape, " X_val", X_val.shape, " X_fit", X_fit.shape, " X_test", X_test.shape)
print(f"     attack share  y_tr {y_tr.mean():.4f}  y_val {y_val.mean():.4f}  y_fit {y_fit.mean():.4f}")
print(f"2.4c range  X_tr [{X_tr.min():.2f}, {X_tr.max():.2f}]  X_val [{X_val.min():.2f}, {X_val.max():.2f}]"
      f"  X_test [{X_test.min():.2f}, {X_test.max():.2f}]")
