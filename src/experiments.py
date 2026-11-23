"""Days 6–7 — run every method for seeds first..last; each run is saved to results/{method}_seed{seed}.pkl.

    python src/experiments.py 1 10                 # all 40 runs; start it in several terminals to run in parallel
    python src/experiments.py 1 2 --tiny           # dry run with tiny budgets, saved to results/tiny/

Each process claims a run by creating {method}_seed{seed}.claim, so parallel processes never run the
same one. After a crash, delete leftover results/*.claim files, then start the script again.
"""
import argparse
import pickle
import time
from scoring import load_prepared, RESULTS
from problem import run_nsga2, nsga2_summary, sequential, run_ga, FPR_MAX

parser = argparse.ArgumentParser()
parser.add_argument("first", type=int)
parser.add_argument("last", type=int)
parser.add_argument("--methods", default="A,C,B,GA")
parser.add_argument("--tiny", action="store_true")
args = parser.parse_args()

pop, gen = (10, 3) if args.tiny else (50, 100)                    # budget = pop * gen evaluations
gen_b = int(0.8 * gen)                                             # B step 1: 80% of the budget ...
trials_b = pop * (gen - gen_b)                                     # ... and step 2 the rest, exactly
out_dir = RESULTS / "tiny" if args.tiny else RESULTS
out_dir.mkdir(parents=True, exist_ok=True)

d = load_prepared()
groups, sets = d["groups"], (d["X_fit"], d["y_fit"], d["X_val"], d["y_val"])

methods = {"A": lambda s: nsga2_summary(run_nsga2(groups, sets, False, s, gen, 1.0, pop), 1.0),
           "C": lambda s: nsga2_summary(run_nsga2(groups, sets, True, s, gen, FPR_MAX, pop), FPR_MAX),
           "B": lambda s: sequential(groups, sets, s, gen_b, trials_b, pop),
           "GA": lambda s: run_ga(groups, sets, s, gen, pop)}

runs = [(seed, m) for seed in range(args.first, args.last + 1) for m in args.methods.split(",")]
for seed, name in runs:
    path = out_dir / f"{name}_seed{seed}.pkl"
    claim = path.with_suffix(".claim")
    if path.exists():
        continue                                                   # finished earlier: skip
    try:
        claim.open("x").close()                                    # fails if another process has it
    except FileExistsError:
        continue
    start = time.time()
    result = methods[name](seed)
    result.update(seconds=time.time() - start, method=name, seed=seed, budget=pop * gen)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    claim.unlink()
    flag = "" if result["evals"] == pop * gen else f"  !! evals {result['evals']} != {pop * gen}"
    print(f"{name:2} seed {seed:2}  {result['seconds'] / 60:5.1f} min  evals {result['evals']}  "
          f"front {len(result['F'])}{flag}", flush=True)
