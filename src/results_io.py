"""Shared helpers for the analysis: load run files, decode a solution, pick the knee."""
import pickle
import numpy as np
import pandas as pd
from scoring import ROOT
from problem import IDSProblem, columns, FPR_MAX

METHODS = ["A", "B", "C", "GA"]


def load_runs(folder):
    """{method: [run dicts sorted by seed]}"""
    runs = {m: [] for m in METHODS}
    for p in folder.glob("*_seed*.pkl"):
        with open(p, "rb") as f:
            r = pickle.load(f)
        runs[r["method"]].append(r)
    return {m: sorted(v, key=lambda r: r["seed"]) for m, v in runs.items()}


def feature_names():
    head = pd.read_csv(ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv", nrows=0).columns
    return [c for c in head if c not in ("id", "attack_cat", "label")]   # same order as prepare.py


def decode(x, groups):
    """Solution dict -> (genes on, column list, tree settings), using IDSProblem's own decoding."""
    on = [i for i in range(len(groups)) if x[f"f{i}"]]
    return on, columns(groups, on), IDSProblem(groups, None, tune="max_depth" in x).settings(x)


def knee(F, min_f1=0.5):
    """Solution closest to the ideal corner after scaling each objective to 0-1; skips F1 < min_f1."""
    ok = np.where(1 - F[:, 0] >= min_f1)[0]
    if len(ok) == 0:
        return None
    G = F[ok]
    span = np.ptp(G, axis=0)
    Z = (G - G.min(axis=0)) / np.where(span > 0, span, 1)
    return int(ok[np.argmin(np.linalg.norm(Z, axis=1))])


def feasible(r):
    return bool(len(r["fpr"]) and r["fpr"].min() <= FPR_MAX + 1e-12)
