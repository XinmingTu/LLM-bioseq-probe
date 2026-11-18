"""Open-weight models' next-token probabilities on the main set (scripts/probe/raw_logprobs.py)
-> results/raw_logprobs_summary.tsv. Free.

Each token in the top 5 counts toward the residue it starts with (whitespace stripped); a residue's
probability is the sum over its tokens, so it is a lower bound when some of its tokens fall outside
the top 5. Per question: top 1 (the most likely residue), top-5 hit (the true residue among the top
5), and p(true) (0 when it is not among them). Per model: real minus shuffled for each, as a mean
over families, with a 95% bootstrap CI over families, as for every other result.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from bioseq_probe.probe import STANDARD

rows = []
for path in sorted(Path("results/raw_logprobs").glob("*.jsonl")):
    for line in path.open():
        d = json.loads(line)
        if "top" not in d:
            continue
        p = {}
        for tok, lp in d["top"]:
            t = tok.strip().upper()
            if t and t[0] in STANDARD:
                p[t[0]] = p.get(t[0], 0) + math.exp(lp)
        if not p:
            continue
        true = d["target"][0]
        rows.append({"model": d["model"], "pfam": d["pfam"], "condition": d["condition"], "top1": max(p, key=p.get) == true,
                     "top5": true in p, "p_true": p.get(true, 0.0)})
df = pd.DataFrame(rows)
rng = np.random.default_rng(2026)
out = []
for model, d in df.groupby("model"):
    row = {"model": model, "n": len(d)}
    for metric in ("top1", "top5", "p_true"):
        fam = d.groupby(["pfam", "condition"])[metric].mean().unstack().dropna()
        gap = (fam.real - fam.shuffled) * 100
        boot = gap.to_numpy()[rng.integers(0, len(gap), (10_000, len(gap)))].mean(1)
        row |= {f"{metric}_real": fam.real.mean() * 100, f"{metric}_shuffled": fam.shuffled.mean() * 100,
                f"{metric}_gap": gap.mean(), f"{metric}_lo": np.percentile(boot, 2.5), f"{metric}_hi": np.percentile(boot, 97.5)}
    out.append(row)
out = pd.DataFrame(out).sort_values("top5_gap", ascending=False)
out.to_csv("results/raw_logprobs_summary.tsv", sep="\t", index=False, float_format="%.3f")
print(out.round(1).to_string(index=False))
