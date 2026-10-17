# nanbisect: Binary Search over Training-Data Subsets for Behavior-Regression Attribution

## Problem

A team retrains a model on old data plus new data. A capability metric drops. Which data slice caused it? Current practice is manual: guess a suspect commit, revert, retrain, compare — O(N) retrains with no systematic search. Data versioning tools (DVC, LakeFS) provide the versions; attribution methods (TracIn, datamodels, influence functions) score individual examples for single predictions; slice-finders (Domino) locate underperforming input regions. None automates the causal experiment over data versions. (Full prior-art survey: `KILLGATE.md`.)

## Method

`nanbisect` applies the `git bisect` protocol to training-data prefixes.

**Inputs.** Ordered data chunks `c[0..N)` ("data commits"), a training function `train(chunks) -> model`, a probe set, and a scalar behavior metric `m(model)`. The user declares a known-good prefix length `G` (typically the old training data).

**Protocol.**
1. **Anchors.** Train on `c[0..G)` → `M_good`; train on `c[0..N)` → `M_bad`. If `M_bad` is not worse than `M_good` (direction-aware, optional tolerance), abort with `NoRegressionDetected`. There must be a measurable drop to bisect.
2. **Search.** Binary-search `[G, N)`. Decision boundary `B = (M_good + M_bad)/2`. At midpoint `mid`, train on `c[0..mid)`; if `m <= B` (higher-is-better) the bad side is `[G, mid)`, else `[mid, N)`. Halve until one chunk remains: the culprit is `c[hi-1]`.
3. **Verification.** Re-train `c[0..hi-1)` (must score good-side) and `c[0..hi)` (must score bad-side). If the bracket fails, report **INCONCLUSIVE** with the reason — never a silent wrong answer. This is the monotonicity guard: the regression must persist once the culprit is included.

**Cost.** `2` anchors + `ceil(log2(N-G))` search steps + `2` verification runs.

## Evaluation (all numbers measured, 2026-10-17, CPU-only)

**Setup.** Synthetic 2D binary classification (sklearn `make_classification`). 16 ordered chunks × 500 samples. Realistic corruption: chunks 0–7 ("old data") exclude a region R of input space; chunks 8–15 ("new data") cover R; chunk 11 (planted culprit) contains 500 region-R points with 100% flipped labels (broken labeling pipeline). Probe: 1000 region-R points with true labels. Model: `DecisionTreeClassifier` (deterministic; a global linear model cannot reflect localized corruption — see Limitations). Metric: probe accuracy, higher-is-better.

**Results (10 seeds, fresh data per seed):**

| metric | measured |
|---|---|
| planted culprit exactly recovered | 10/10 (100%) |
| verification bracket passed | 10/10 |
| training runs used | 7 every seed (2 + 3 + 2) |
| probe accuracy drop, median | 55.8pp (min 17.6, max 62.1) |
| pytest suite | 8/8 green (7 unit + 1 end-to-end) |

**Worked example (seed 0).** `M_good = 0.740`, `M_bad = 0.421`, boundary `0.581`. Step 1 tests prefix 12 → `0.340` (bad) → hi=12. Step 2 tests prefix 10 → `0.734` (good) → lo=10. Step 3 tests prefix 11 → `0.780` (good) → lo=11. Culprit: chunk 11. Verified.

## Limitations (not hidden)

1. **Monotonicity assumption.** Like `git bisect`, if the regression does not persist once the culprit chunk is included (e.g., dilution by later clean data flips it back), the search can mislead — which is why the verification bracket exists and why failure is loud, not silent.
2. **Model capacity.** The demo uses a decision tree deliberately: a global linear model (logistic regression) showed only ~0.1pp drops on the same corruption because it cannot represent localized damage. The tool finds what the *model+metric* reflect; a model class blind to the corruption yields no anchors and the tool correctly refuses to run.
3. **Chunk granularity.** Resolution is the user's data commits, not individual samples.
4. **Retraining cost.** ~log2(N) full training runs. The demo's runs take seconds (sklearn, CPU); LLM-scale retraining bounds practical N.
5. **Confounded changes.** Code must be held fixed across the bisect; otherwise attribute to code first (`git bisect`), then data.
6. **Probe dependence.** The tool attributes the regression *of the given metric on the given probe*. A probe that doesn't measure the regressed capability will find nothing.

## Reproducibility

```bash
pip install -r requirements.txt
cd demo
python make_demo.py --outdir data --seed 0   # regenerates planted data
PYTHONPATH=.. python -m nanbisect.cli run --config nanbisect.yaml
python eval_seeds.py                          # 10-seed recovery eval
```
