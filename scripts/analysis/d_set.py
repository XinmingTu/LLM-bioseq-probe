"""The D set (scripts/data/07_d_set.py): proteins first public after the assumed training cutoff,
with no earlier relative at 50% identity or more -> results/d_set_summary.tsv. Free.

Same statistics as the main set: per-family real-minus-shuffled accuracy of the residue after the
cut, mean over the 31 families, 95% bootstrap CI over families. For reference, each model's gap on
the main set restricted to the same 31 families (proteins public before the cutoff, no relative at
90% identity or more): a lower gap on D shows that having seen the protein or close relatives helps.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bioseq_probe.probe import cut_off, parse, shown_prefix

D = pd.read_csv("data/d_set_sequences.tsv", sep="\t")
MAIN = pd.read_csv("data/confirm_sequences.tsv", sep="\t")
SEQ = pd.concat([D, MAIN]).set_index("accession").sequence
FAMILIES = set(D.pfam)
NAMES = {"claude-fable-5-2": "Claude Fable 5.2", "claude-opus-5-5": "Claude Opus 5.5", "gpt-6.2-sol": "GPT-6.2 Sol",
         "progen3-3b": "ProGen3-3B", "protgpt3-10B": "ProtGPT3-10B"}
rng = np.random.default_rng(2026)


def llm(path):
    rows = []
    for line in open(path):
        d = json.loads(line)
        pred = parse(d.get("text", ""), shown_prefix(SEQ[d["accession"]], d["accession"], d["condition"], d["cut"]), cut=cut_off(d))
        rows.append({"pfam": d["pfam"], "condition": d["condition"], "valid": bool(pred) and d.get("stop") != "refusal",
                     "ok": bool(pred) and pred[0] == d["target"][0]})
    return pd.DataFrame(rows)


def plm(path):
    d = pd.DataFrame(map(json.loads, open(path)))
    return d.assign(valid=True, ok=d.first_correct)[["pfam", "condition", "valid", "ok"]]


def gap(d):
    d = d[d.valid & d.pfam.isin(FAMILIES)]
    fam = d.groupby(["pfam", "condition"]).ok.mean().unstack().dropna()
    g = (fam.real - fam.shuffled) * 100
    boot = g.to_numpy()[rng.integers(0, len(g), (10_000, len(g)))].mean(1)
    return g.mean(), *np.percentile(boot, [2.5, 97.5]), fam.real.mean() * 100, fam.shuffled.mean() * 100, len(g)


rows = []
for key, name in NAMES.items():
    is_llm = key in ("claude-fable-5-2", "claude-opus-5-5", "gpt-6.2-sol")
    dpath = Path(f"results/d_set/{key}.jsonl") if is_llm else Path(f"results/plm_d/{key}__prefix.jsonl")
    mpath = Path(f"results/confirm/{key}.jsonl") if is_llm else Path(f"results/plm_confirm/{key}__prefix.jsonl")
    if not dpath.exists():
        continue
    load = llm if is_llm else plm
    dg, dlo, dhi, dreal, dshuf, nfam = gap(load(dpath))
    mg, mlo, mhi, *_ = gap(load(mpath))
    rows.append({"model": name, "D_gap": dg, "D_lo": dlo, "D_hi": dhi, "D_real": dreal, "D_shuffled": dshuf, "D_families": nfam,
                 "main_same_families_gap": mg, "main_lo": mlo, "main_hi": mhi})
out = pd.DataFrame(rows)
out.to_csv("results/d_set_summary.tsv", sep="\t", index=False, float_format="%.2f")
print(out.round(1).to_string(index=False))
