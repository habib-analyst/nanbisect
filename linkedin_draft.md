# LinkedIn draft: nanbisect (project #42)

Your retrained model got worse. Which data slice caused it?

Versioning tools give you data commits. Attribution papers score individual examples. Nobody automates the actual debugging experiment. So I built the missing tool.

nanbisect is git bisect for training data. Give it your ordered data chunks, a training function, and a probe metric. It trains on halved prefixes, compares behavior, and isolates the culprit chunk in about log2(N) runs instead of N. Then it re-verifies the final bracket, and if the regression isn't monotonic it tells you INCONCLUSIVE instead of silently pointing at the wrong slice.

Measured, not claimed: on a planted-regression benchmark (one chunk with systematically mislabeled data, 10 fresh seeds), it recovered the exact culprit chunk 10 out of 10 times, using 7 training runs where exhaustive search needs 16. 8 out of 8 tests green. CPU-only, model-agnostic, zero cost.

One honest finding from building it: a global linear model barely registers localized data corruption (0.1pp drop on the same poison that craters a tree by 56pp). Your model has to be able to see the damage before any attribution tool can find it. The tool refuses to run when there's no measurable regression, which is the correct behavior.

If you retrain models on growing datasets, this is the debugging loop you've been doing by hand.

github.com/habib-analyst/nanbisect
