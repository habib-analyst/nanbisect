"""End-to-end test: real tiny model + planted regression.

Generates a small demo dataset (seed 123, culprit chunk fixed), runs the real
Bisector with the sklearn decision-tree train/metric functions, and asserts
the planted culprit is exactly isolated with a passing verification bracket.
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo.make_demo import make_demo, GOOD_CHUNKS, CULPRIT_CHUNK
from demo.train_demo import train, accuracy
from nanbisect.core import Bisector

BASE = os.path.dirname(os.path.abspath(__file__))


def test_end_to_end_planted_regression():
    tmp = os.path.join(BASE, ".e2e_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        info = make_demo(os.path.join(tmp, "data"), seed=123, culprit_chunk=CULPRIT_CHUNK)
        b = Bisector(
            chunks=info["chunk_paths"],
            train_fn=train,
            metric_fn=accuracy,
            probe_path=info["probe_path"],
            workdir=os.path.join(tmp, "work"),
            good_chunks=GOOD_CHUNKS,
            higher_is_better=True,
        )
        res = b.run()
        assert res.verified, f"verification failed: {res.inconclusive_reason}"
        assert res.culprit_index == CULPRIT_CHUNK, (
            f"expected chunk {CULPRIT_CHUNK}, got {res.culprit_index}"
        )
        # Sanity: the regression must be measurable (dropped metric).
        assert res.m_bad < res.m_good - 0.05, (
            f"regression too weak to be meaningful: {res.m_good:.3f} -> {res.m_bad:.3f}"
        )
        # Bisect efficiency: ~log2 steps, not exhaustive.
        assert res.train_runs <= 10, f"too many training runs: {res.train_runs}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
