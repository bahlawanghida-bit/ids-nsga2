"""Day 3 · steps 3.2–3.3 — all-features tree plus chi-square and mutual-information top-k baselines (validation)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, mutual_info_classif
from scoring import load_prepared, score, RESULTS, FIGURES

d = load_prepared()
X_fit, y_fit, X_val, y_val, groups = d["X_fit"], d["y_fit"], d["X_val"], d["y_val"], d["groups"]
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

all_cols = sorted(j for g in groups for j in g)
rows = [{"method": "all features", "k": 42, "features": 42,
         **score(all_cols, {}, X_fit, y_fit, X_val, y_val), "genes": " ".join(map(str, range(42)))}]

chi, _ = chi2(X_fit, y_fit)                                        # scores from X_fit only
mi = mutual_info_classif(X_fit, y_fit, random_state=0)
gene_score = lambda s: np.array([np.nan_to_num(s)[g].sum() for g in groups])   # gene = sum of its columns

for name, s in [("chi2", gene_score(chi)), ("MI", gene_score(mi))]:
    order = np.argsort(-s, kind="stable")                          # best gene first
    for k in [5, 10, 15, 20, 25, 30]:
        genes = sorted(order[:k].tolist())
        cols = sorted(j for i in genes for j in groups[i])
        r = score(cols, {}, X_fit, y_fit, X_val, y_val)
        rows.append({"method": name, "k": k, "features": k, **r, "genes": " ".join(map(str, genes))})

table = pd.DataFrame(rows)
table.to_csv(RESULTS / "baselines_val.csv", index=False)
print(table.drop(columns="genes").round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(5.5, 3.6))
for name, marker in [("chi2", "o"), ("MI", "s")]:
    t = table[table.method == name]
    ax.plot(t.k, t.f1, marker=marker, label=f"{'χ²' if name == 'chi2' else 'Mutual information'} top-k")
ax.axhline(rows[0]["f1"], color="grey", ls="--", label="All 42 features")
ax.set_xlabel("Features kept (k)")
ax.set_ylabel("F1 on validation")
ax.set_title("Filter baselines: F1 against number of features")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "baselines_f1_vs_k.png", dpi=200)
