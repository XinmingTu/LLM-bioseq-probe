"""Protein language models on the main set -> results/plm_summary.tsv. Free.

Same statistics as the LLMs (scripts/analysis/confirm.py): per-family real-minus-shuffled accuracy
of the residue after the cut, mean over families, 95% bootstrap CI over families. Reads the
replies written by scripts/probe/protein_lm.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

NAMES = {"protxlstm-102M": "Prot-xLSTM 102M", "progen3-3b": "ProGen3-3B", "protgpt3-10B": "ProtGPT3-10B", "protgpt3-1.3B": "ProtGPT3-1.3B", "protgpt2": "ProtGPT2",
         "esmc-600m": "ESM C 600M", "esm3-open": "ESM3-open 1.4B", "esm2-650M": "ESM-2 650M"}
size = pd.read_csv("data/confirm_sequences.tsv", sep="\t").groupby("pfam").family_proteins.first()
rng = np.random.default_rng(2026)

rows = []
for path in sorted(Path("results/plm_confirm").glob("*__*.jsonl")):
    model, mode = path.stem.split("__")
    d = pd.DataFrame(map(json.loads, path.open()))
    acc = d.groupby("condition").first_correct.mean()
    fam = d.groupby(["pfam", "condition"]).first_correct.mean().unstack().dropna()
    gap = (fam.real - fam.shuffled) * 100
    boot = gap.to_numpy()[rng.integers(0, len(gap), (10_000, len(gap)))].mean(1)
    rho, _ = spearmanr(gap, np.log10(size.reindex(gap.index)))
    rows.append({"model": model, "name": NAMES[model], "mode": mode, "n": len(d), "real_acc": acc.real, "shuffled_acc": acc.shuffled,
                 "d": gap.mean(), "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5), "rho_family_size": rho})
out = pd.DataFrame(rows).sort_values("d", ascending=False)
out.to_csv("results/plm_summary.tsv", sep="\t", index=False, float_format="%.4f")
print(out.to_string(index=False))
