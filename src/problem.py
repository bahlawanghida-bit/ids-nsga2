"""Days 4–6 — the optimization problems and the four methods: A (features only), B (sequential),
C (joint) and the single-objective GA baseline."""
import warnings
import numpy as np
import optuna
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Binary, Integer, Choice
from pymoo.core.callback import Callback
from pymoo.core.mixed import MixedVariableSampling, MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scoring import score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
FPR_MAX = 0.01


def columns(groups, on):
    return sorted(j for i in on for j in groups[i])                # fixed column order


class IDSProblem(ElementwiseProblem):
    """42 feature genes (+ 4 setting genes when tune=True). Objectives: 1 - F1, features, ms.
    Constraints: at least 1 feature; FPR <= fpr_max (fpr_max=1.0 switches it off)."""

    def __init__(self, groups, sets, tune=True, fpr_max=FPR_MAX):
        vars = {f"f{i}": Binary() for i in range(len(groups))}
        if tune:
            vars["max_depth"] = Integer(bounds=(2, 30))
            vars["min_leaf"] = Integer(bounds=(1, 50))
            vars["criterion"] = Choice(options=["gini", "entropy"])
            vars["normal_weight"] = Integer(bounds=(1, 50))
        super().__init__(vars=vars, n_obj=3, n_ieq_constr=2)
        self.groups, self.sets = groups, sets                       # not self.data: pymoo overwrites it
        self.tune, self.fpr_max = tune, fpr_max

    def settings(self, x):
        if not self.tune:
            return {}                                               # method A: scikit-learn defaults
        return dict(max_depth=int(x["max_depth"]), min_samples_leaf=int(x["min_leaf"]),
                    criterion=str(x["criterion"]), class_weight={0: int(x["normal_weight"]), 1: 1})

    def _evaluate(self, x, out, *args, **kwargs):
        on = [i for i in range(len(self.groups)) if x[f"f{i}"]]
        if not on:                                                  # no features: worst, infeasible
            out["F"], out["G"] = [1.0, 0, 1e6], [1, 1]
            return
        r = score(columns(self.groups, on), self.settings(x), *self.sets)
        out["F"] = [1 - r["f1"], len(on), r["ms"]]
        out["G"] = [1 - len(on), r["fpr"] - self.fpr_max]


class FrontLog(Callback):
    """Replaces save_history=True: keeps only each generation's feasible front and the evaluations used."""

    def __init__(self):
        super().__init__()
        self.fronts, self.evals = [], []

    def notify(self, algorithm):
        opt = algorithm.opt
        self.fronts.append(opt.get("F")[opt.get("feasible")[:, 0]])
        self.evals.append(algorithm.evaluator.n_eval)


def plain(x):
    """Decision vector as plain Python values (safe to pickle and to reuse in another problem)."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in x.items()}


def run_nsga2(groups, sets, tune, seed, n_gen=100, fpr_max=FPR_MAX, pop_size=50):
    problem = IDSProblem(groups, sets, tune=tune, fpr_max=fpr_max)
    dedup = MixedVariableDuplicateElimination()
    algorithm = NSGA2(pop_size=pop_size, sampling=MixedVariableSampling(),
                      mating=MixedVariableMating(eliminate_duplicates=dedup),
                      eliminate_duplicates=dedup)
    log = FrontLog()
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, callback=log)
    res.fronts, res.front_evals = log.fronts, log.evals
    return res


def nsga2_summary(res, fpr_max):
    """What the analysis needs from a pymoo result (methods A and C)."""
    out = {"history": res.fronts, "history_evals": res.front_evals,
           "evals": res.algorithm.evaluator.n_eval}
    if res.X is None:                                               # no feasible solution found
        G = res.pop.get("G")
        return {**out, "F": np.empty((0, 3)), "X": [], "G": np.empty((0, 2)), "fpr": np.empty(0),
                "best_fpr_reached": float(G[:, 1].min() + fpr_max)}
    return {**out, "F": res.F, "X": [plain(x) for x in res.X], "G": res.G, "fpr": res.G[:, 1] + fpr_max}


def tune_subset(judge, genes, n_trials, seed):
    """Method B, step 2 for ONE feature subset: Optuna tunes the 4 settings with the exact same scoring as C."""
    def objective(trial):
        x = dict(genes,
                 max_depth=trial.suggest_int("max_depth", 2, 30),
                 min_leaf=trial.suggest_int("min_leaf", 1, 50),
                 criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
                 normal_weight=trial.suggest_int("normal_weight", 1, 50))
        out = {}
        judge._evaluate(x, out)
        trial.set_user_attr("r", {"F": [float(v) for v in out["F"]], "G": [float(v) for v in out["G"]], "x": x})
        trial.set_constraint("fpr", float(out["G"][1]))             # Optuna knows the FPR limit, as C does
        return out["F"][0], out["F"][2]                             # 1 - F1, time

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return [t.user_attrs["r"] for t in study.trials]


def sequential(groups, sets, seed, n_gen=80, trials=1000, pop_size=50):
    """Method B: NSGA-II on features only (no FPR limit), then Optuna tunes each subset on that front."""
    res = run_nsga2(groups, sets, tune=False, seed=seed, n_gen=n_gen, fpr_max=1.0, pop_size=pop_size)
    subsets = [plain(x) for x in res.X]
    k = len(subsets)
    share = [trials // k + (1 if i < trials % k else 0) for i in range(k)]   # remainder included
    judge = IDSProblem(groups, sets, tune=True)
    done = [r for g, n in zip(subsets, share) if n > 0 for r in tune_subset(judge, g, n, seed)]
    pool = [r for r in done if max(r["G"]) <= 0]                    # feasible only
    out = {"evals": res.algorithm.evaluator.n_eval + len(done), "step1_front": k, "trials": len(done),
           "history": [], "history_evals": []}
    if not pool:
        return {**out, "F": np.empty((0, 3)), "X": [], "G": np.empty((0, 2)), "fpr": np.empty(0)}
    F = np.array([r["F"] for r in pool])
    G = np.array([r["G"] for r in pool])
    keep = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return {**out, "F": F[keep], "X": [pool[i]["x"] for i in keep], "G": G[keep], "fpr": G[keep, 1] + FPR_MAX}


class GAProblem(ElementwiseProblem):
    """Single-objective GA baseline: 42 feature genes, default tree, one weighted score."""

    def __init__(self, groups, sets):
        vars = {f"f{i}": Binary() for i in range(len(groups))}
        super().__init__(vars=vars, n_obj=1, n_ieq_constr=1)
        self.groups, self.sets = groups, sets
        self.t_all = score(columns(groups, range(len(groups))), {}, *sets)["ms"]   # time of the all-features tree

    def _evaluate(self, x, out, *args, **kwargs):
        on = [i for i in range(len(self.groups)) if x[f"f{i}"]]
        if not on:
            out["F"], out["G"] = [1e6], [1]
            return
        r = score(columns(self.groups, on), {}, *self.sets)
        out["F"] = [0.6 * (1 - r["f1"]) + 0.2 * len(on) / len(self.groups) + 0.2 * r["ms"] / self.t_all]
        out["G"] = [1 - len(on)]


def run_ga(groups, sets, seed, n_gen=100, pop_size=50):
    problem = GAProblem(groups, sets)
    dedup = MixedVariableDuplicateElimination()
    algorithm = GA(pop_size=pop_size, sampling=MixedVariableSampling(),
                   mating=MixedVariableMating(eliminate_duplicates=dedup), eliminate_duplicates=dedup)
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed)
    x = plain(res.X)
    on = [i for i in range(len(groups)) if x[f"f{i}"]]
    val = score(columns(groups, on), {}, *sets)                     # its F1, FPR, time on validation, for reporting
    return {"F": np.array([[1 - val["f1"], len(on), val["ms"]]]), "X": [x], "weighted": float(res.F[0]),
            "fpr": np.array([val["fpr"]]), "val": val, "evals": res.algorithm.evaluator.n_eval,
            "history": [], "history_evals": []}
