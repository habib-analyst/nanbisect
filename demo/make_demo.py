"""Generate the nanbisect demo: synthetic data with a planted regression.

Realistic scenario: the team retrains on old data + new data covering a NEW
REGION of the input space. One new-data slice (chunk 11) has systematically
mislabeled points in that region (broken labeling pipeline). The team's
region-specific probe metric tanks. nanbisect must find chunk 11.

Layout:
  demo/data/chunk_00.npz ... chunk_15.npz   (16 ordered data commits)
  demo/data/probe.npz                        (region-R points, true labels)

  chunks 0-7  ("old data"): 500 pts each, region R EXCLUDED (never seen)
  chunks 8-10, 12-15 ("new data"): 500 pts each, natural region-R coverage
  chunk 11 (culprit): 500 pts ALL in region R, ALL mislabeled
  probe: 1000 pts in region R with true labels

Each .npz holds X (n,2) float64 and y (n,) int64.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.datasets import make_classification

N_CHUNKS = 16
CHUNK_SIZE = 500
PROBE_SIZE = 1000
CULPRIT_CHUNK = 11  # 0-indexed
GOOD_CHUNKS = 8     # chunks [0,8) = old data, known good


def _region_threshold(X: np.ndarray) -> float:
    """Region R = top 15% by x0 (quantile-based, stable across seeds)."""
    return float(np.percentile(X[:, 0], 85))


def _region_mask(X: np.ndarray, threshold: float) -> np.ndarray:
    """Region R: the new slice of input space."""
    return X[:, 0] > threshold


def make_demo(outdir: str, seed: int = 0, culprit_chunk: int = CULPRIT_CHUNK) -> dict:
    rng = np.random.RandomState(seed)
    os.makedirs(outdir, exist_ok=True)

    # Large clean pool to draw from.
    X_pool, y_pool = make_classification(
        n_samples=60000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.2,
        flip_y=0.02,
        random_state=rng,
    )
    in_R = _region_mask(X_pool, _region_threshold(X_pool))
    XR, yR = X_pool[in_R], y_pool[in_R]
    XnR, ynR = X_pool[~in_R], y_pool[~in_R]
    assert len(XR) >= 2500, f"R pool too small: {len(XR)}"

    def take(X_src, y_src, n):
        idx = rng.choice(len(X_src), size=n, replace=False)
        mask = np.ones(len(X_src), dtype=bool)
        mask[idx] = False
        return X_src[idx], y_src[idx], X_src[mask], y_src[mask]

    chunk_paths = []
    for i in range(N_CHUNKS):
        if i < GOOD_CHUNKS:
            # Old data: region R excluded.
            xc, yc, XnR, ynR = take(XnR, ynR, CHUNK_SIZE)
        elif i == culprit_chunk:
            # Culprit: region R only, ALL labels flipped (broken pipeline).
            xc, yc, XR, yR = take(XR, yR, CHUNK_SIZE)
            yc = 1 - yc
        else:
            # New data: natural coverage (mix, region R included).
            n_r = int(CHUNK_SIZE * 0.08)
            xr, yr, XR, yR = take(XR, yR, n_r)
            xn, yn, XnR, ynR = take(XnR, ynR, CHUNK_SIZE - n_r)
            xc = np.concatenate([xr, xn])
            yc = np.concatenate([yr, yn])
            perm = rng.permutation(len(xc))
            xc, yc = xc[perm], yc[perm]
        p = os.path.join(outdir, f"chunk_{i:02d}.npz")
        np.savez(p, X=xc, y=yc)
        chunk_paths.append(p)

    xp, yp, _, _ = take(XR, yR, PROBE_SIZE)
    probe_path = os.path.join(outdir, "probe.npz")
    np.savez(probe_path, X=xp, y=yp)

    return {
        "chunk_paths": chunk_paths,
        "probe_path": probe_path,
        "culprit_chunk": culprit_chunk,
        "n_chunks": N_CHUNKS,
        "seed": seed,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="demo/data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    info = make_demo(args.outdir, args.seed)
    print(f"wrote {info['n_chunks']} chunks + probe to {args.outdir} "
          f"(culprit: chunk {info['culprit_chunk']}, seed {args.seed})")
