# nanbisect demo

**The story.** Your team retrains a classifier on old data plus new data. The new data covers a *new region* of the input space (top 15% by x0). Your region-specific probe accuracy tanks from ~80% to ~40%. One of the 8 new data slices has a broken labeling pipeline. Which one?

**Run it** (from this directory):

```bash
pip install -r ../requirements.txt
PYTHONPATH=.. python -m nanbisect.cli run --config nanbisect.yaml
```

The tool trains 7 tiny decision trees (~seconds, CPU), bisects the 8 new-data chunks in 3 steps, and names the culprit: `data/chunk_11.npz`.

**What's planted.** `make_demo.py` generates 16 chunks × 500 samples (2D binary classification). Chunks 0–7 ("old data") exclude region R. Chunks 8–15 ("new data") cover R — except chunk 11, whose 500 region-R points are 100% mislabeled. The probe (`probe.npz`) is 1000 region-R points with true labels.

**Regenerate / re-evaluate:**

```bash
python make_demo.py --outdir data --seed 0   # fresh planted data, any seed
python eval_seeds.py                          # 10-seed recovery eval -> results/eval_seeds.json
```

Measured: 10/10 seeds recover chunk 11 exactly, 7 training runs each.
