"""Core bisection engine.

The protocol mirrors `git bisect`, but over training-data prefixes instead of
code commits:

  good = model trained on chunks[0:good_chunks]      (known-good, e.g. old data)
  bad  = model trained on chunks[0:N]                (full data, regressed)

Binary-search the boundary in [good_chunks, N): the smallest prefix length L
such that training on chunks[0:L] lands on the *bad* side of the metric.
The culprit chunk is chunks[L-1] — the chunk whose inclusion flips the metric.

Decision boundary: midpoint between M_good and M_bad. A prefix is "bad-side"
iff its metric is at least as far toward M_bad as the midpoint (direction
aware: works for higher-is-better and lower-is-better metrics).

Monotonicity assumption (documented limitation): once the culprit chunk is
included, every larger prefix must stay bad-side. The engine verifies the
final bracket explicitly and flags non-monotonic runs instead of silently
returning a wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence


# A training function: given chunk identifiers and an output path, trains a
# model and returns the path to the saved model artifact.
TrainFn = Callable[[List[str], str], str]
# A metric function: given a saved model path and a probe path, returns a
# scalar behavior score.
MetricFn = Callable[[str, str], float]


@dataclass
class BisectStep:
    step: int
    lo: int            # current known-good prefix length
    hi: int            # current known-bad prefix length
    mid: int           # prefix length tested this step
    n_chunks_tested: int
    metric: float
    side: str          # "good" | "bad"
    decision: str      # human-readable, e.g. "bad -> hi = 12"

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "lo": self.lo,
            "hi": self.hi,
            "mid": self.mid,
            "n_chunks_tested": self.n_chunks_tested,
            "metric": self.metric,
            "side": self.side,
            "decision": self.decision,
        }


@dataclass
class BisectResult:
    culprit_index: int | None     # index into chunks, None if inconclusive
    culprit_chunk: str | None     # chunk identifier, None if inconclusive
    good_prefix_len: int
    n_chunks: int
    m_good: float
    m_bad: float
    boundary: float
    steps: List[BisectStep] = field(default_factory=list)
    train_runs: int = 0           # total training runs used (incl. anchors+verify)
    verified: bool = False        # bracket re-check passed
    monotonic: bool = True        # False if any step contradicted the bracket
    inconclusive_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "culprit_index": self.culprit_index,
            "culprit_chunk": self.culprit_chunk,
            "good_prefix_len": self.good_prefix_len,
            "n_chunks": self.n_chunks,
            "m_good": self.m_good,
            "m_bad": self.m_bad,
            "boundary": self.boundary,
            "train_runs": self.train_runs,
            "verified": self.verified,
            "monotonic": self.monotonic,
            "inconclusive_reason": self.inconclusive_reason,
            "steps": [s.as_dict() for s in self.steps],
        }


class NoRegressionDetected(Exception):
    """Raised when the full-data metric is not worse than the good-prefix metric."""


class Bisector:
    """Binary-search over ordered data-chunk prefixes for a behavior regression."""

    def __init__(
        self,
        chunks: Sequence[str],
        train_fn: TrainFn,
        metric_fn: MetricFn,
        probe_path: str,
        workdir: str,
        good_chunks: int,
        higher_is_better: bool = True,
        regression_tol: float = 0.0,
    ) -> None:
        if good_chunks < 1:
            raise ValueError("good_chunks must be >= 1 (need a non-empty known-good prefix)")
        if good_chunks >= len(chunks):
            raise ValueError("good_chunks must be < len(chunks) (need something to bisect)")
        self.chunks = list(chunks)
        self.train_fn = train_fn
        self.metric_fn = metric_fn
        self.probe_path = probe_path
        self.workdir = workdir
        self.good_chunks = good_chunks
        self.higher_is_better = higher_is_better
        self.regression_tol = regression_tol
        self._run_counter = 0

    # -- internals ------------------------------------------------------
    def _train_and_score(self, prefix_len: int, tag: str) -> tuple[str, float]:
        import os

        os.makedirs(self.workdir, exist_ok=True)
        model_path = os.path.join(self.workdir, f"model_{tag}.pkl")
        self.train_fn(self.chunks[:prefix_len], model_path)
        self._run_counter += 1
        score = self.metric_fn(model_path, self.probe_path)
        return model_path, score

    def _is_bad_side(self, score: float, boundary: float) -> bool:
        if self.higher_is_better:
            return score <= boundary
        return score >= boundary

    def _is_regression(self, m_good: float, m_bad: float) -> bool:
        if self.higher_is_better:
            return m_bad < m_good - self.regression_tol
        return m_bad > m_good + self.regression_tol

    # -- public ----------------------------------------------------------
    def run(self) -> BisectResult:
        n = len(self.chunks)

        # Anchor measurements.
        _, m_good = self._train_and_score(self.good_chunks, "anchor_good")
        _, m_bad = self._train_and_score(n, "anchor_bad")

        if not self._is_regression(m_good, m_bad):
            raise NoRegressionDetected(
                f"No regression: good-prefix metric={m_good:.4f} vs "
                f"full-data metric={m_bad:.4f} (higher_is_better={self.higher_is_better}). "
                f"nanbisect needs a measurable behavior drop to bisect."
            )

        boundary = (m_good + m_bad) / 2.0
        lo, hi = self.good_chunks, n
        steps: List[BisectStep] = []
        monotonic = True
        step_no = 0

        while hi - lo > 1:
            step_no += 1
            mid = (lo + hi) // 2
            step_lo, step_hi = lo, hi  # range being halved this step
            _, score = self._train_and_score(mid, f"step{step_no}")
            bad_side = self._is_bad_side(score, boundary)
            side = "bad" if bad_side else "good"
            if bad_side:
                decision = f"bad-side -> hi = {mid}"
                hi = mid
            else:
                decision = f"good-side -> lo = {mid}"
                lo = mid
            steps.append(
                BisectStep(
                    step=step_no,
                    lo=step_lo,
                    hi=step_hi,
                    mid=mid,
                    n_chunks_tested=mid,
                    metric=score,
                    side=side,
                    decision=decision,
                )
            )

        culprit_index = hi - 1  # chunk whose inclusion flips the metric

        # Verification bracket: [0..hi-1) must be good-side, [0..hi) bad-side.
        _, m_lo = self._train_and_score(hi - 1, "verify_lo")
        _, m_hi = self._train_and_score(hi, "verify_hi")
        lo_good_side = not self._is_bad_side(m_lo, boundary)
        hi_bad_side = self._is_bad_side(m_hi, boundary)
        verified = bool(lo_good_side and hi_bad_side)
        if not verified:
            monotonic = False

        result = BisectResult(
            culprit_index=culprit_index if verified else None,
            culprit_chunk=self.chunks[culprit_index] if verified else None,
            good_prefix_len=self.good_chunks,
            n_chunks=n,
            m_good=m_good,
            m_bad=m_bad,
            boundary=boundary,
            steps=steps,
            train_runs=self._run_counter,
            verified=verified,
            monotonic=monotonic,
            inconclusive_reason=(
                None
                if verified
                else (
                    "verification bracket failed: prefix without the culprit "
                    f"scored {m_lo:.4f} ({'bad' if not lo_good_side else 'good'}-side) and "
                    f"prefix with the culprit scored {m_hi:.4f} "
                    f"({'good' if not hi_bad_side else 'bad'}-side). The regression is "
                    "not monotonic in the data order — the culprit cannot be "
                    "attributed to a single contiguous chunk."
                )
            ),
        )
        return result
