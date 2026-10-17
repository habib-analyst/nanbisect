"""nanbisect CLI: init / run / report."""

from __future__ import annotations

import importlib
import json
import os
import sys

import click
import yaml

from .core import Bisector, NoRegressionDetected
from .reporting import render_markdown, render_text


def _load_dotted(path: str):
    mod_name, _, attr = path.rpartition(".")
    if not mod_name:
        raise click.ClickException(f"Bad dotted path: {path!r} (need module.attr)")
    sys.path.insert(0, os.getcwd())
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


@click.group()
@click.version_option()
def main():
    """git bisect for your training data."""


@main.command()
@click.option("--dir", "target_dir", default=".", help="Where to scaffold the project.")
def init(target_dir):
    """Scaffold a nanbisect project (config + example training script)."""
    os.makedirs(target_dir, exist_ok=True)
    cfg_path = os.path.join(target_dir, "nanbisect.yaml")
    train_path = os.path.join(target_dir, "train_example.py")
    if os.path.exists(cfg_path):
        raise click.ClickException(f"{cfg_path} already exists; refusing to overwrite.")
    config = {
        "chunks": ["data/chunk_00.npz"],
        "good_chunks": 1,
        "probe": "data/probe.npz",
        "train_fn": "train_example.train",
        "metric_fn": "train_example.accuracy",
        "higher_is_better": True,
        "workdir": ".nanbisect",
        "regression_tol": 0.0,
    }
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    with open(train_path, "w") as f:
        f.write(TRAIN_EXAMPLE)
    click.echo(f"wrote {cfg_path}")
    click.echo(f"wrote {train_path}")
    click.echo("Next: point 'chunks' at your ordered data chunks, implement train/metric, then run:")
    click.echo("  nanbisect run --config nanbisect.yaml")


@main.command()
@click.option("--config", default="nanbisect.yaml", help="Path to nanbisect.yaml.")
@click.option("--out", default=".nanbisect/result.json", help="Where to write the result JSON.")
def run(config, out):
    """Run the bisection and write the result JSON + text report."""
    with open(config) as f:
        cfg = yaml.safe_load(f)
    train_fn = _load_dotted(cfg["train_fn"])
    metric_fn = _load_dotted(cfg["metric_fn"])
    bisector = Bisector(
        chunks=cfg["chunks"],
        train_fn=train_fn,
        metric_fn=metric_fn,
        probe_path=cfg["probe"],
        workdir=cfg.get("workdir", ".nanbisect"),
        good_chunks=cfg["good_chunks"],
        higher_is_better=cfg.get("higher_is_better", True),
        regression_tol=cfg.get("regression_tol", 0.0),
    )
    try:
        result = bisector.run()
    except NoRegressionDetected as e:
        click.echo(f"no regression detected: {e}")
        sys.exit(2)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result.as_dict(), f, indent=2)
    click.echo(render_text(result))
    click.echo(f"\nresult JSON: {out}")


@main.command()
@click.option("--result", default=".nanbisect/result.json", help="Result JSON from `nanbisect run`.")
@click.option("--format", "fmt", type=click.Choice(["text", "markdown"]), default="text")
def report(result, fmt):
    """Render a saved bisection result."""
    from .core import BisectResult, BisectStep

    with open(result) as f:
        d = json.load(f)
    res = BisectResult(
        culprit_index=d["culprit_index"],
        culprit_chunk=d["culprit_chunk"],
        good_prefix_len=d["good_prefix_len"],
        n_chunks=d["n_chunks"],
        m_good=d["m_good"],
        m_bad=d["m_bad"],
        boundary=d["boundary"],
        steps=[BisectStep(**s) for s in d["steps"]],
        train_runs=d["train_runs"],
        verified=d["verified"],
        monotonic=d["monotonic"],
        inconclusive_reason=d["inconclusive_reason"],
    )
    click.echo(render_markdown(res) if fmt == "markdown" else render_text(res))


TRAIN_EXAMPLE = '''"""Example training script for nanbisect.

Contract:
  train(chunk_paths, model_path) -> model_path
      Train on the given data chunks (ordered .npz files with X, y),
      save the model to model_path (pickle), return model_path.
  accuracy(model_path, probe_path) -> float
      Load the model, return accuracy on the probe set. Higher is better.
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression


def _load_chunks(chunk_paths):
    Xs, ys = [], []
    for p in chunk_paths:
        d = np.load(p)
        Xs.append(d["X"])
        ys.append(d["y"])
    return np.concatenate(Xs), np.concatenate(ys)


def train(chunk_paths, model_path):
    X, y = _load_chunks(chunk_paths)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    return model_path


def accuracy(model_path, probe_path):
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    d = np.load(probe_path)
    return float(clf.score(d["X"], d["y"]))
'''


if __name__ == "__main__":
    main()
