# nanbisect — Kill-Gate Report
**Date:** 2026-10-17 | **Type:** research-only kill-gate (no build, no repo, no downloads)
**Concept:** "git bisect for your training data" — a developer tool that binary-searches over
training-data subsets ("data commits") to find which data slice introduced a capability
regression or behavior change in a trained model. Model got worse at X after retraining →
which data commit caused it?

**Hunt rank:** #3 / 63/70 (novelty 8, narrative 9, demo 9, $0 10, skill 9, legal 10, timebox 8)

---

## Angle 1 — Data attribution methods (influence functions, TracIn, datamodels)
**Verdict: CLEAR (adjacent, different question)**

- Influence functions (Koh & Liang 2017), TracIn (Pruthi et al. 2020, https://research.google/blog/tracin-a-simple-method-to-estimate-training-data-influence/), datamodels (Ilyas et al. 2022) all answer: *"which training examples most influenced this specific prediction?"* — per-example influence scores via gradients or subset-retraining ensembles.
- None offers bisect-style search over data subsets, none frames the workflow as regression debugging, none is packaged as a developer CLI. TracIn needs checkpoint-per-epoch gradient tracing; datamodels needs thousands of subset models as a research harness, not a debugging tool.
- Structural difference: attribution scores *individual examples* for *one prediction*; nanbisect searches *data versions* for *a behavior regression*. Different question, different user, different output.
- Evidence: https://research.google/blog/tracin-a-simple-method-to-estimate-training-data-influence/ ; https://deepwiki.com/frederick0329/TracIn/2.1-core-concepts

## Angle 2 — Training-data debugging / slice-finding tools (Domino et al.)
**Verdict: CLEAR (adjacent, different question)**

- Domino (Eyuboglu et al. 2022, https://arxiv.org/pdf/2203.14960) discovers *underperforming slices of the input distribution* in embedding space and describes them in natural language. It audits model behavior on data slices; it never asks which *training-data change* caused the behavior.
- Spotlight, GEORGE, MultiAccuracy Boost: same family — find where the model is bad, not which training data made it bad.
- Evidence: https://hazyresearch.stanford.edu/blog/2022-04-02-domino

## Angle 3 — Checkpoint-diffing / behavior-regression tools
**Verdict: CLEAR (adjacent, monitors instead of attributing)**

- Evidently AI, Arize, Fiddler, Alibi Detect, whylogs: detect drift and performance degradation, diff model behavior across versions. They answer *"did behavior change?"* — never *"which training-data subset caused it?"*
- LLM regression-testing vendors (e.g. http://www.bards.ai/services/llm-regression-testing): diff-based paired eval, behavior clustering — again detection/diffing, no data-cause search.
- The CUBIG analysis of data-versioning platforms states the boundary explicitly: versioning/diffing tools "do not measure live model accuracy, determine whether a shift is harmful" — the causal step is left manual (https://cubig.ai/articles/model-performance-degradation-revalidation/).

## Angle 4 — "git bisect" analogues for ML
**Verdict: CLEAR (gap confirmed — only tutorials, no tool)**

- All "git bisect for ML" content found is *tutorials on using git bisect on code commits* (e.g. scikit-learn CI debugging: https://www.python4data.science/en/latest/productive/git/advanced/bisect.html ; knowledgebase https://knowledgebase.batistalab.com/git/intro/). The ML example given is always "accuracy dropped, bisect the *code* commits."
- The hunt's original finding stands: the *data* variant exists only as a documented manual procedure (Jan 2026 Medium article). Zero packaged tools on GitHub/PyPI frame or implement bisection over training-data subsets.
- PyPI scan (nullsec-datapoisoning, mlsec, mldebug, dataset-inspector, dqdetect): all are static dataset auditors or poisoning detectors — none does subset-retraining bisection.

## Angle 5 — Dataset versioning + lineage (DVC, LakeFS, Delta Lake, Pachyderm)
**Verdict: CLEAR (gives the versions, not the search)**

- DVC, LakeFS, Delta Lake, Pachyderm provide versioning, branching, lineage, rollback. The CUBIG lakeFS analysis describes the *manual* workflow: identify the suspect commit, revert, rerun, inspect the diff — and notes the conclusion "still depends on whether the rerun held the other conditions constant." No automation of the bisect loop; no behavioral test harness; no binary search over versions.
- Structural gap: versioning tools version the *data*; nanbisect would automate the *experiment* (train on subset A vs B, compare behavior, halve the range). Nobody closes that loop.
- Evidence: https://cubig.ai/articles/lakefs-data-versioning-vs-syntitan-ai-readiness/ ; https://blogwolf.com/4-ai-dataset-versioning-platforms-that-help-you-manage-data-lifecycles/

## Angle 6 — 2026 papers on training-data bisection / regression attribution
**Verdict: CLEAR (nothing found)**

- Web/arXiv search for 2026 papers on training-data bisection, regression attribution to data subsets, or "which data caused the capability drop" returned no matching paper. Closest hits were unrelated (a dev.to post on flagged-datasets registries, ML-Yarning diagnostics chapters).
- No Papers-with-Code task, no benchmark, no method paper for data-subset bisection.

## Bonus angle — Poison/label-error detectors (Cleanlab et al.)
**Verdict: CLEAR (adjacent, static audit)**

- Cleanlab (12k+ stars, https://docs.cleanlab.ai/) finds mislabeled samples via confident learning; nullsec-datapoisoning / mlsec detect poisoning via spectral signatures / activation clustering. All are *static audits of one dataset* answering "which samples look bad" — not "which data commit introduced the regression," and none retrains on subsets or binary-searches.

---

## Feasibility assessment ($0, CPU)

- **Core loop is tractable:** with N ordered data commits, binary search needs ~log2(N) training runs. Tiny models (logistic regression / small CNN on MNIST-scale) train in seconds–minutes on CPU. A 16-commit demo → ~4–5 short runs. Pure Python + torch/sklearn, no GPU needed.
- **Minimal honest demo:** inject a synthetic regression (e.g., flip labels of one data slice, or add a corrupted subset at a known "commit"), then show the tool converging on the culprit slice in ~log2 steps and rendering the offending samples. The injected ground truth makes the eval honest and reproducible.
- **Honest limitations to document:** (1) monotonicity assumption — like git bisect, the regression must persist once introduced; (2) granularity is the user's "data commits," not individual samples (sample-level pinpointing is Cleanlab's job); (3) retraining cost bounds the practical N; (4) confounded changes (code + data changing together) need the user to hold code fixed.
- **No $0 blockers.** No API costs, no GPU needed, no datasets to download (synthetic demo suffices for v1).

## Differentiation

If partial prior art existed, the structural gap would still hold: every adjacent tool answers a *different* question (which sample is bad / which version exists / did behavior change?). nanbisect is the only artifact that would automate the *causal experiment* — train on subset A vs B, compare on a probe set, halve the range — packaged as a one-line developer tool (`nanbisect run --data-commits ... --probe ...`). The "unbuilt tool" novelty from the hunt is confirmed, not weakened, by this gate.

---

## FINAL VERDICT: **GO**

**Load-bearing reason:** No packaged tool exists that binary-searches over training-data subsets to attribute a behavior regression — every adjacent system (TracIn/datamodels, Domino, Evidently, DVC/LakeFS, Cleanlab, poisoning detectors) answers a different question (which example influenced this prediction / where is the model bad / which data version exists / which samples look dirty), and the technique is documented only as a manual procedure. The shelf is empty, the $0 CPU path is concrete, and the differentiation is structural, not cosmetic.
