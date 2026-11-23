"""Days 4–5 — correctness tests for IDSProblem before any long run."""
import pandas as pd
from scoring import load_prepared, RESULTS
from problem import IDSProblem, run_nsga2

d = load_prepared()
groups, sets = d["groups"], (d["X_fit"], d["y_fit"], d["X_val"], d["y_val"])
baseline_f1 = pd.read_csv(RESULTS / "baselines_val.csv").set_index("method").loc["all features", "f1"]

# Test 1: all 42 genes on, tune=False -> F1 must equal the all-features baseline exactly
out = {}
IDSProblem(groups, sets, tune=False, fpr_max=1.0)._evaluate({f"f{i}": True for i in range(42)}, out)
print("Test 1  F1", round(1 - out["F"][0], 6), "baseline", round(baseline_f1, 6))
assert abs((1 - out["F"][0]) - baseline_f1) < 1e-12

# Test 2: no features -> worst objectives and a violated constraint
out = {}
IDSProblem(groups, sets, tune=True)._evaluate({f"f{i}": False for i in range(42)}, out)
print("Test 2  F", out["F"], "G", out["G"])
assert out["F"][0] == 1.0 and out["G"][0] > 0

# Tiny run, method C: settings inside their ranges?
res = run_nsga2(groups, sets, tune=True, seed=1, n_gen=3, pop_size=10)
assert res.X is not None, "no feasible solution in the tiny C run"
for x in res.X:
    assert 2 <= x["max_depth"] <= 30 and 1 <= x["min_leaf"] <= 50
    assert x["criterion"] in ("gini", "entropy") and 1 <= x["normal_weight"] <= 50
print("Tiny C  front", len(res.X), "| first:", {k: v for k, v in res.X[0].items() if not k.startswith("f")},
      "| F", res.F[0].round(4), "| FPR", round(res.G[0, 1] + 0.01, 4))
print("all checks passed")
