"""Day 8 (+ Figures 1 and 3) — validation only: feasibility, hypervolume, Mann-Whitney U and A12 (Table 2),
convergence (Figure 2), fronts of B and C (Figure 1), settings along front C (Figure 3, RQ3)."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr
from pymoo.indicators.hv import HV
from scoring import RESULTS, FIGURES
from results_io import load_runs, feature_names, feasible, FPR_MAX

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, default=RESULTS)
parser.add_argument("--expect", type=int, default=10)
args = parser.parse_args()
out_dir = args.results
fig_dir = FIGURES if args.results == RESULTS else args.results
fig_dir.mkdir(parents=True, exist_ok=True)

runs = load_runs(args.results)
for m, rs in runs.items():                                         # never analyse incomplete results
    assert len(rs) == args.expect, f"{m}: {len(rs)} runs, expected {args.expect}"
    assert all(r["evals"] == r["budget"] for r in rs), f"{m}: a run did not use exactly its budget"

# 8.1 Feasibility
feas = {m: sum(feasible(r) for r in rs) for m, rs in runs.items()}
lowest_fpr = {m: float(min(r["fpr"].min() if len(r["fpr"]) else r.get("best_fpr_reached", np.nan) for r in rs))
              for m, rs in runs.items()}
# Post hoc (added after seeing results): A's only feasible solutions were "detect-nothing" trees (F1 = 0),
# so feasibility is also reported for usable detectors, with the F1 >= 0.5 threshold of the selection rule.
useful = {m: sum(bool(len(r["F"]) and np.any((r["fpr"] <= FPR_MAX + 1e-12) & (1 - r["F"][:, 0] >= 0.5)))
                 for r in rs) for m, rs in runs.items()}

# 8.2 Hypervolume: ONE normalization over all B and C fronts, reference point 1.1, empty front -> 0
REF = 1.1
every = np.vstack([r["F"] for m in "BC" for r in runs[m] if len(r["F"])])
lo, hi = every.min(axis=0), every.max(axis=0)
span = np.where(hi > lo, hi - lo, 1.0)
hv = HV(ref_point=np.full(3, REF))


def hv_of(F):
    if len(F) == 0:
        return 0.0
    Z = (np.asarray(F) - lo) / span
    Z = Z[np.all(Z < REF, axis=1)]                                  # points past the reference add nothing
    return float(hv(Z)) if len(Z) else 0.0


HVs = {m: np.array([hv_of(r["F"]) for r in runs[m]]) for m in "BC"}

# 8.3 One-sided Mann-Whitney U (C > B) and A12
test = mannwhitneyu(HVs["C"], HVs["B"], alternative="greater")
a12 = test.statistic / (len(HVs["C"]) * len(HVs["B"]))
wins = int(sum(c > b for c, b in zip(HVs["C"], HVs["B"])))         # same seed, side by side
sentence = (f"C's hypervolume was higher than B's in {wins} of {len(HVs['C'])} runs "
            f"(one-sided Mann-Whitney U = {test.statistic:.0f}, p = {test.pvalue:.4f}, A12 = {a12:.2f}).")
print(sentence)

# Post hoc sensitivity check: the same test with every F1 < 0.5 solution removed from the fronts
HVu = {m: np.array([hv_of(r["F"][1 - r["F"][:, 0] >= 0.5]) if len(r["F"]) else 0.0 for r in runs[m]]) for m in "BC"}
sens = mannwhitneyu(HVu["C"], HVu["B"], alternative="greater")
sens_a12 = sens.statistic / (len(HVu["C"]) * len(HVu["B"]))
sens_wins = int(sum(c > b for c, b in zip(HVu["C"], HVu["B"])))
print(f"Sensitivity (fronts without F1 < 0.5): C > B in {sens_wins} of 10, p = {sens.pvalue:.4f}, A12 = {sens_a12:.2f}; "
      f"HV mean C {HVu['C'].mean():.4f}, B {HVu['B'].mean():.4f}")

# 8.5 Table 2
rows = []
for m in ["A", "GA", "B", "C"]:
    row = {"method": m, "feasible runs": f"{feas[m]} of {len(runs[m])}",
           "feasible runs, F1 >= 0.5 (post hoc)": f"{useful[m]} of {len(runs[m])}", "lowest FPR": round(lowest_fpr[m], 4)}
    if m in HVs:
        row.update({"HV mean": HVs[m].mean(), "HV std": HVs[m].std(ddof=1), "HV median": np.median(HVs[m])})
    if m == "C":
        row.update({"p (C > B)": test.pvalue, "A12": a12})
    rows.append(row)
table2 = pd.DataFrame(rows)
table2.to_csv(out_dir / "table2_hypervolume.csv", index=False)
print(table2.round(4).to_string(index=False))

# 8.4 Figure 2: convergence of C, with B's final mean hypervolume as a line (RQ2)
curves = np.array([[hv_of(F) for F in r["history"]] for r in runs["C"]])
evals = np.array(runs["C"][0]["history_evals"])
mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1)
b_mean = HVs["B"].mean()
reach = np.where(mean >= b_mean)[0]
cross = None if len(reach) == 0 else {"generation": int(reach[0]) + 1, "evaluations": int(evals[reach[0]])}
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(evals, mean, color="teal", label=f"C (joint), mean of {len(curves)} runs")
ax.fill_between(evals, mean - std, mean + std, color="teal", alpha=0.2, label="C, ±1 std")
ax.axhline(b_mean, color="darkorange", ls="--", label="B (sequential), final mean")
if cross:
    ax.axvline(cross["evaluations"], color="grey", ls=":", lw=1)
    ax.annotate(f"reaches B after {cross['evaluations']:,} evaluations", (cross["evaluations"], b_mean),
                xytext=(8, 8), textcoords="offset points", fontsize=8)
ax.set_xlabel("Evaluations")
ax.set_ylabel("Hypervolume (normalized)")
ax.set_title("Convergence of the joint method (feasible fronts, FPR ≤ 1%)")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(fig_dir / "fig2_convergence.png", dpi=200)

# Figure 1: fronts of B and C (run with hypervolume closest to the median), baselines as markers
median_run = {m: int(np.argmin(np.abs(HVs[m] - np.median(HVs[m])))) for m in "BC"}
base = pd.read_csv(RESULTS / "baselines_val.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (j, xlabel, scale) in zip(axes, [(1, "Features kept", 1), (2, "Inference time (µs per connection)", 1000)]):
    for m, color, label in [("B", "darkorange", "B (sequential)"), ("C", "teal", "C (joint)")]:
        F = runs[m][median_run[m]]["F"]
        F = F[1 - F[:, 0] >= 0.5]
        ax.scatter(F[:, j] * scale, 1 - F[:, 0], color=color, s=22, label=f"{label}, FPR ≤ 1%", zorder=3)
    for name, marker, label in [("chi2", "^", "χ² top-k"), ("MI", "D", "MI top-k"), ("all features", "X", "All 42 features")]:
        t = base[base.method == name]
        x = t.features if j == 1 else t.ms * scale
        ax.scatter(x, t.f1, marker=marker, color="grey", s=30, label=f"{label} (FPR > 1%)", zorder=2)
    ga = np.vstack([r["F"] for r in runs["GA"]])
    ax.scatter(ga[:, j] * scale, 1 - ga[:, 0], marker="+", color="black", s=40, label="Single-objective GA (FPR > 1%)")
    ax.set_xlabel(xlabel)
axes[0].set_ylabel("F1 on validation")
axes[1].legend(fontsize=7, loc="lower right")
fig.suptitle("Pareto fronts of B and C (median-hypervolume run) and baselines; F1 < 0.5 not shown")
fig.tight_layout()
fig.savefig(fig_dir / "fig1_fronts.png", dpi=200)

# Figure 3 and RQ3: tree settings along front C, pooled over the 10 runs
pool = []
for r in runs["C"]:
    for F, x, fpr in zip(r["F"], r["X"], r["fpr"]):
        if 1 - F[0] >= 0.5:
            pool.append({"seed": r["seed"], "f1": 1 - F[0], "features": F[1], "ms": F[2], "fpr": fpr,
                         "max_depth": x["max_depth"], "min_leaf": x["min_leaf"],
                         "normal_weight": x["normal_weight"], "criterion": x["criterion"],
                         "genes": [i for i in range(42) if x[f"f{i}"]]})
pool = pd.DataFrame(pool)
rho = pd.DataFrame({s: {o: spearmanr(pool[s], pool[o]).statistic for o in ["f1", "fpr", "ms", "features"]}
                    for s in ["max_depth", "min_leaf", "normal_weight"]}).T
rho.to_csv(out_dir / "rq3_settings_spearman.csv")
print("RQ3 Spearman correlation, settings vs objectives along front C:\n", rho.round(2).to_string())

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
med_seed = runs["C"][median_run["C"]]["seed"]
for ax, s in zip(axes, ["max_depth", "min_leaf", "normal_weight"]):
    for crit, marker in [("gini", "o"), ("entropy", "s")]:
        t = pool[pool.criterion == crit]
        ax.scatter(t.f1, t[s], marker=marker, s=14, alpha=0.35, color="teal", label=f"{crit}, all runs")
    t = pool[pool.seed == med_seed]
    ax.scatter(t.f1, t[s], s=26, facecolors="none", edgecolors="black", label="median run")
    ax.set_xlabel("F1 on validation")
    ax.set_ylabel(s)
axes[0].legend(fontsize=7)
fig.suptitle(f"Tree settings along front C (FPR ≤ 1%), pooled over {len(runs['C'])} runs")
fig.tight_layout()
fig.savefig(fig_dir / "fig3_settings_front_C.png", dpi=200)

# Which traffic features the joint front keeps most often
names = feature_names()
freq = pd.Series(0.0, index=names)
for genes in pool.genes:
    freq.iloc[genes] += 1
freq = (freq / len(pool)).sort_values(ascending=False).rename("share of front-C solutions")
freq.to_csv(out_dir / "feature_frequency_front_C.csv")

json.dump({"sentence": sentence, "U": float(test.statistic), "p": float(test.pvalue), "A12": float(a12), "wins": wins,
           "HV": {m: v.tolist() for m, v in HVs.items()}, "feasible_runs": feas, "lowest_fpr": lowest_fpr,
           "feasible_runs_f1_0.5_post_hoc": useful,
           "sensitivity_without_f1_below_0.5": {"HV": {m: v.tolist() for m, v in HVu.items()}, "U": float(sens.statistic),
                                                "p": float(sens.pvalue), "A12": float(sens_a12), "wins": sens_wins},
           "normalization": {"lo": lo.tolist(), "hi": hi.tolist(), "ref": REF},
           "convergence_reaches_B": cross, "B_final_mean_HV": float(b_mean),
           "C_mean_HV_by_generation": mean.tolist(),
           "median_run_seed": {m: runs[m][median_run[m]]["seed"] for m in "BC"},
           "front_C_pooled_solutions": len(pool), "top_features_front_C": freq.head(10).round(3).to_dict(),
           "run_minutes": {m: [round(r["seconds"] / 60, 1) for r in rs] for m, rs in runs.items()}},
          open(out_dir / "analysis.json", "w"), indent=2)
