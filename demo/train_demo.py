"""Demo training script for nanbisect (sklearn logistic regression on .npz chunks)."""

import pickle

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def _load_chunks(chunk_paths):
    Xs, ys = [], []
    for p in chunk_paths:
        d = np.load(p)
        Xs.append(d["X"])
        ys.append(d["y"])
    return np.concatenate(Xs), np.concatenate(ys)


def train(chunk_paths, model_path):
    # Decision tree: unlike a global linear model, it has the capacity to
    # reflect *localized* data corruption (which is the point of the demo).
    # Deterministic via random_state; CPU-trivial.
    X, y = _load_chunks(chunk_paths)
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    return model_path


def accuracy(model_path, probe_path):
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    d = np.load(probe_path)
    return float(clf.score(d["X"], d["y"]))
