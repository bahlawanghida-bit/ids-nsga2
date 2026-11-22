from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"   # ids-nsga2/data/raw

train = pd.read_csv(DATA / "UNSW_NB15_training-set.csv")
test  = pd.read_csv(DATA / "UNSW_NB15_testing-set.csv")

# 1. Shape and columns
print("train:", train.shape, " test:", test.shape)      # (175341, 45)  (82332, 45)
print(list(train.columns))                               # 42 features + id, attack_cat, label

# 2. Attack share per file
print("attack share train:", round(train["label"].mean(), 3))   # ~0.681
print("attack share test: ", round(test["label"].mean(), 3))    # ~0.551

# 3. Attack categories
print(train["attack_cat"].value_counts())                # Worms ≈ 130 rows

# 4. Text (categorical) features: preview for step 2.4
for col in ["proto", "service", "state"]:
    print(col, "train:", train[col].nunique(), " test:", test[col].nunique())
print("state values only in test:", set(test["state"]) - set(train["state"]))  # {'ACC', 'CLO'}