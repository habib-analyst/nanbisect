"""Unit tests for the bisection logic (mock training functions, no real models)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanbisect.core import Bisector, BisectStep, NoRegressionDetected


def make_mock(culprit_at=None, m_good=0.9, m_bad=0.4):
    """Mock train/metric where chunk `culprit_at` (0-indexed) flips the metric.

    Metric is  m_good  for prefixes not containing the culprit,
               m_bad   for prefixes containing it.
    """
    calls = {"n": 0}

    def train(chunk_paths, model_path):
        calls["n"] += 1
        # "model" = the prefix length; metric reads it back.
        with open(model_path, "w") as f:
            f.write(str(len(chunk_paths)))
        return model_path

    def metric(model_path, probe_path):
        with open(model_path) as f:
            prefix_len = int(f.read())
        if culprit_at is not None and prefix_len > culprit_at:
            return m_bad
        return m_good

    return train, metric, calls


def _bisector(culprit_at, n=16, good=8, **kw):
    train, metric, calls = make_mock(culprit_at=culprit_at, **kw)
    b = Bisector(
        chunks=[f"chunk_{i}" for i in range(n)],
        train_fn=train,
        metric_fn=metric,
        probe_path="probe",
        workdir="/tmp/nb_test",
        good_chunks=good,
        higher_is_better=True,
    )
    return b, calls


def test_finds_planted_culprit():
    b, calls = _bisector(culprit_at=11)
    res = b.run()
    assert res.verified
    assert res.culprit_index == 11
    assert res.culprit_chunk == "chunk_11"
    # 2 anchors + ceil(log2(8)) bisect steps + 2 verification
    assert res.train_runs == 2 + 3 + 2


def test_finds_culprit_at_boundary():
    # culprit is the first bisected chunk
    b, _ = _bisector(culprit_at=8)
    res = b.run()
    assert res.verified and res.culprit_index == 8
    # culprit is the last chunk
    b, _ = _bisector(culprit_at=15)
    res = b.run()
    assert res.verified and res.culprit_index == 15


def test_no_regression_raises():
    b, _ = _bisector(culprit_at=None)  # metric never drops
    with pytest.raises(NoRegressionDetected):
        b.run()


def test_lower_is_better():
    train, metric, calls = make_mock(culprit_at=11, m_good=0.1, m_bad=0.9)
    b = Bisector(
        chunks=[f"chunk_{i}" for i in range(16)],
        train_fn=train,
        metric_fn=metric,
        probe_path="probe",
        workdir="/tmp/nb_test",
        good_chunks=8,
        higher_is_better=False,
    )
    res = b.run()
    assert res.verified and res.culprit_index == 11


def test_non_monotonic_flagged_not_silent():
    # Training noise can break monotonicity: the same prefix scores
    # differently across runs. The tool must report INCONCLUSIVE (never a
    # silent wrong culprit) when the verification bracket fails.
    # This mock is deterministic (fixed seed) and was verified to trigger
    # the inconclusive path.
    import random

    rng = random.Random(7)
    calls = {"n": 0}

    def train(chunk_paths, model_path):
        calls["n"] += 1
        with open(model_path, "w") as f:
            f.write(str(len(chunk_paths)))
        return model_path

    def metric(model_path, probe_path):
        with open(model_path) as f:
            n = int(f.read())
        base = 0.4 if n > 11 else 0.9
        # heavy training noise: can flip either side across the 0.65 boundary
        return base + rng.uniform(-0.35, 0.35)

    b = Bisector(
        chunks=[f"chunk_{i}" for i in range(16)],
        train_fn=train,
        metric_fn=metric,
        probe_path="probe",
        workdir="/tmp/nb_test",
        good_chunks=8,
        higher_is_better=True,
    )
    res = b.run()
    # Either it got lucky and verified a correct culprit, or (expected) it
    # flagged inconclusive. It must NEVER silently return a wrong culprit.
    if not res.verified:
        assert res.culprit_index is None
        assert res.inconclusive_reason is not None
        assert "not monotonic" in res.inconclusive_reason
    else:
        assert res.culprit_index == 11


def test_invalid_good_chunks():
    train, metric, _ = make_mock(culprit_at=11)
    with pytest.raises(ValueError):
        Bisector(["a", "b"], train, metric, "p", "/tmp", good_chunks=0)
    with pytest.raises(ValueError):
        Bisector(["a", "b"], train, metric, "p", "/tmp", good_chunks=2)


def test_step_log_structure():
    b, _ = _bisector(culprit_at=11)
    res = b.run()
    assert len(res.steps) == 3
    for s in res.steps:
        assert isinstance(s, BisectStep)
        assert s.side in ("good", "bad")
        assert s.lo < s.mid <= s.hi
    d = res.as_dict()
    assert d["culprit_index"] == 11
    assert len(d["steps"]) == 3
