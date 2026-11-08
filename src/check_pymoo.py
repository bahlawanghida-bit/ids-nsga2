"""Day 1, task 1.2: check that pymoo works and print versions for the log."""
import sys
import pymoo, sklearn, optuna, pandas
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize

res = minimize(get_problem("zdt1"), NSGA2(pop_size=50), ("n_gen", 100), seed=1, verbose=False)
print(f"ZDT1: {len(res.F)} non-dominated points (about 50 means pymoo works)")
print(f"Python {sys.version.split()[0]} | pymoo {pymoo.__version__} | scikit-learn {sklearn.__version__} "
      f"| Optuna {optuna.__version__} | pandas {pandas.__version__}")
