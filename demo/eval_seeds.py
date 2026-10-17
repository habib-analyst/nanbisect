"""Multi-seed evaluation: planted-culprit recovery rate for nanbisect.

For each seed: generate fresh demo data (culprit chunk fixed at 11), run the
full bisection via the Bisector API, record whether the planted culprit was
exactly isolated, whether verification passed, and training runs used.

Prints a summary table and writes results/eval_seeds.json.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo.make_demo import make_demo, GOOD_CHUNKS, CULPRIT_CHUNK  # noqa: E402
from demo.train_demo import train, accuracy  # noqa: E402
from nanbisect.core import Bisector, NoRegressionDetected  # noqa: E402

N_SEEDS = 10
BASE = os.path.dirname(os.path.abspath(__file__))


def run_seed(seed: int) -> dict:
    datadir = os.path.join(BASE, f".eval_seed{seed}", "data")
    workdir = os.path.join(BASE, f".eval_seed{seed}", "work")
    shutil.rmtree(os.path.join(BASE, f".eval_seed{seed}"), ignore_errors=True)
    info = make_demo(datadir, seed=seed, culprit_chunk=CULPRIT_CHUNK)
    chunks = info["chunk_paths"]
    try:
        b = Bisector(
            chunks=chunks,
            train_fn=train,
            metric_fn=accuracy,
            probe_path=info["probe_path"],
            workdir=workdir,
            good_chunks=GOOD_CHUNKS,
            higher_is_better=True,
        )
        res = b.run()
    except NoRegressionDetected as e:
        return {"seed": seed, "ok": False, "reason": f"no-regression: {e}"}
    finally:
        shutil.rmtree(os.path.join(BASE, f".eval_seed{seed}"), ignore_errors=True)
    return {
        "seed": seed,
        "ok": True,
        "culprit_index": res.culprit_index,
        "expected": CULPRIT_CHUNK,
        "recovered": res.culprit_index == CULPRIT_CHUNK,
        "verified": res.verified,
        "train_runs": res.train_runs,
        "m_good": round(res.m_good, 4),
        "m_bad": round(res.m_bad, 4),
        "drop_pp": round((res.m_good - res.m_bad) * 100, 2),
    }


def main():
    results = [run_seed(s) for s in range(N_SEEDS)]
    ok = [r for r in results if r["ok"]]
    recovered = [r for r in ok if r["recovered"]]
    verified = [r for r in ok if r["verified"]]
    runs = [r["train_runs"] for r in ok]
    drops = [r["drop_pp"] for r in ok]

    print(f"seeds: {N_SEEDS} | completed: {len(ok)}")
    print(f"recovered culprit exactly: {len(recovered)}/{len(ok)}")
    print(f"verification passed: {len(verified)}/{len(ok)}")
    if runs:
        print(f"train runs: median={sorted(runs)[len(runs)//2]} min={min(runs)} max={max(runs)}")
    if drops:
        print(f"metric drop (pp): median={sorted(drops)[len(drops)//2]:.1f} "
              f"min={min(drops):.1f} max={max(drops):.1f}")
    for r in results:
        if r["ok"]:
            flag = "OK " if r["recovered"] else "MISS"
            print(f"  seed {r['seed']}: {flag} culprit={r['culprit_index']} "
                  f"runs={r['train_runs']} drop={r['drop_pp']}pp verified={r['verified']}")
        else:
            print(f"  seed {r['seed']}: FAIL {r['reason']}")

    outdir = os.path.join(BASE, "results")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "eval_seeds.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {outdir}/eval_seeds.json")


if __name__ == "__main__":
    main()
