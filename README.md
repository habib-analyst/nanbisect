# nanbisect — git bisect for your training data

**Your retrained model got worse. Which data slice caused it?**

`nanbisect` binary-searches over your ordered training-data chunks ("data commits"): it trains on progressively halved prefixes, measures a behavior metric on your probe set, and isolates the culprit chunk — in ~log2(N) training runs instead of N.

```
$ nanbisect run --config nanbisect.yaml
================================================================
nanbisect report
================================================================
chunks            : 16 (good prefix: 8)
metric(good)      : 0.7400
metric(bad, full) : 0.4210
decision boundary : 0.5805 (midpoint)
training runs     : 7

bisection log:
  step        range  tested    metric  side   decision
     1  [  8, 16)       12    0.3400  bad    bad-side -> hi = 12
     2  [  8, 12)       10    0.7340  good   good-side -> lo = 10
     3  [ 10, 12)       11    0.7800  good   good-side -> lo = 11

CULPRIT: chunk #11 (data/chunk_11.npz)
verified: prefix without it is good-side, with it is bad-side.
================================================================
```

## 30-second demo

```bash
pip install -r requirements.txt
cd demo
python make_demo.py --outdir data --seed 0   # generate the planted demo data (~5s)
PYTHONPATH=.. python -m nanbisect.cli run --config nanbisect.yaml
```

(The demo ships 16 synthetic data chunks; chunk 11 has a planted labeling corruption. The tool finds it in 7 training runs. See `demo/README.md` for the story.)

## How it works

1. **Anchor**: train on your known-good prefix (`good_chunks`) → `M_good`; train on all chunks → `M_bad`. If `M_bad` isn't worse, it refuses to run (no regression, nothing to bisect).
2. **Bisect**: binary-search the range `[good_chunks, N)`. Decision boundary = midpoint between `M_good` and `M_bad`. Each step trains on a prefix, scores the probe set, halves the range.
3. **Verify**: re-trains the final bracket (prefix without vs. with the culprit). If the bracket doesn't hold — non-monotonic regression — it reports **INCONCLUSIVE** instead of a wrong answer.

## Usage

```bash
nanbisect init                      # scaffold nanbisect.yaml + example train script
nanbisect run --config nanbisect.yaml
nanbisect report --result .nanbisect/result.json --format markdown
```

`nanbisect.yaml`:

```yaml
chunks: [data/chunk_00.npz, ..., data/chunk_15.npz]  # ordered data commits
good_chunks: 8        # prefix [0,8) is known-good (e.g. your old training data)
probe: data/probe.npz # held-out probe set measuring the regressed capability
train_fn: my_train.train    # (chunk_paths, model_path) -> model_path
metric_fn: my_train.accuracy # (model_path, probe_path) -> float
higher_is_better: true
```

The tool is model-agnostic: you bring `train_fn`/`metric_fn` (sklearn, torch, XGBoost, anything). It just orchestrates the experiments.

## Measured results

10-seed planted-regression eval (synthetic, decision tree, 16 chunks):

| metric | value |
|---|---|
| culprit exactly recovered | **10/10 seeds** |
| verification bracket passed | 10/10 |
| training runs used (median) | **7** (vs 16 exhaustive) |
| probe metric drop (median) | 55.8pp (min 17.6, max 62.1) |

Unit + end-to-end: `pytest tests/` → **8/8 green**.

## Honest limitations

- **Monotonicity**: like `git bisect`, the regression must persist once the culprit chunk is included. Non-monotonic regressions are flagged INCONCLUSIVE, not silently misattributed.
- **Granularity**: it finds the culprit *chunk*, not the culprit *sample*. Your "data commits" define the resolution. (Sample-level is Cleanlab's job.)
- **Retraining cost**: each step trains a model. ~log2(N) runs of *your* training cost. Cheap for sklearn, pricey for LLMs — size your chunks accordingly.
- **Confounded changes**: hold your code fixed. If code *and* data changed, bisect the code with `git bisect` first.
- **Probe quality**: the tool is only as good as your probe set. It finds what broke *the metric you gave it*.

See [PAPER.md](PAPER.md) for the full method and numbers.
