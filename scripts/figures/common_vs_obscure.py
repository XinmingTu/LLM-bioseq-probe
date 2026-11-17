"""Pilot: real-minus-shuffled on well-known vs obscure proteins, per model -> results/fig_common_vs_obscure.png. Free.

well-known: vertebrate members of well-studied families (>= 100 PDB structures)
obscure:    non-vertebrate members of DUF families (no structure, almost no papers)
Models that declined the plain prompt use their best-guess run.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioseq_probe.probe import cut_off, parse, shown_prefix

LABEL = {"gpt-4o-2024-11-20": "GPT-4o", "gpt-4.1-2025-04-14": "GPT-4.1", "gpt-5.5": "GPT-5.5", "gpt-6.2-sol": "GPT-6.2 Sol", "gpt-6.2-astra": "GPT-6.2 Astra", "gpt-6.1-sol": "GPT-6.1 Sol", "gpt-6-astra": "GPT-6 Astra",
         "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5", "claude-opus-4-6": "Claude Opus 4.6", "claude-opus-4-8": "Claude Opus 4.8",
         "claude-opus-5": "Claude Opus 5", "claude-opus-5-5": "Claude Opus 5.5", "claude-fable-5-1": "Claude Fable 5.1", "claude-fable-5-2": "Claude Fable 5.2", "gemini-2.5-pro": "Gemini 2.5 Pro", "gemini-3.5-flash": "Gemini 3.5 Flash",
         "gemini-3.8-flash": "Gemini 3.8 Flash", "gemini-4-carbon": "Gemini 4 Carbon", "deepseek-v4p1-flash": "DeepSeek V4.1 Flash",
         "qwen3p8-2p4t-a95b": "Qwen 3.8", "kimi-k3p1-pro": "Kimi K3.1 Pro", "glm-5p3": "GLM-5.3", "minimax-m3": "MiniMax M3",
         "gpt-oss-120b": "gpt-oss-120b", "nemotron-3-ultra-nvfp4": "Nemotron 3 Ultra"}
rng = np.random.default_rng(0)
seqs = pd.read_csv("data/sequences_pilot.tsv", sep="\t").set_index("accession")
group = pd.read_csv("data/sequence_groups.tsv", sep="\t").set_index("accession").group


def subset_of(acc):
    vert = group[acc] == "Vertebrata"
    lit = seqs.loc[acc, "lit_cell"]
    return "well-known" if vert and lit == "rich" else ("obscure" if not vert and lit == "poor" else None)


def answer(r):
    prefix = shown_prefix(seqs.loc[r["accession"]].sequence, r["accession"], r["condition"], r["cut"])
    return parse(r.get("text", ""), prefix, cut=cut_off(r))


def boot(g):
    t = g.groupby(["pfam", "condition"])["first"].mean().unstack().dropna()
    D = (t["real"] - t["shuffled"]).to_numpy() * 100
    b = D[rng.integers(0, len(D), (4000, len(D)))].mean(1)
    return D.mean(), *np.percentile(b, [2.5, 97.5])


rows = []
for m, name in LABEL.items():
    path = Path(f"results/probe_guess/{m}.jsonl")
    path = path if path.exists() else Path(f"results/probe/{m}.jsonl")
    recs = [json.loads(line) for line in path.open()]
    answers = [(r, answer(r)) for r in recs if "error" not in r and r.get("stop") != "refusal"]
    df = pd.DataFrame([{"pfam": r["pfam"], "condition": r["condition"], "sub": subset_of(r["accession"]),
                        "first": p[0] == r["target"][0]} for r, p in answers if p])
    for sub in ("well-known", "obscure"):
        d, lo, hi = boot(df[df["sub"] == sub])
        rows.append({"model": name, "sub": sub, "d": d, "lo": lo, "hi": hi, "guess": "probe_guess" in str(path)})
res = pd.DataFrame(rows)
res.to_csv("results/common_vs_obscure.tsv", sep="\t", index=False)

names = list(LABEL.values())
fig, ax = plt.subplots(figsize=(8.6, 7.4))
for i, n in enumerate(names):
    y = len(names) - 1 - i
    for sub, dy, c in (("well-known", 0.17, "#eb6834"), ("obscure", -0.17, "#2a78d6")):
        r = res[(res.model == n) & (res["sub"] == sub)].iloc[0]
        ax.plot([r.lo, r.hi], [y + dy, y + dy], color=c, lw=1.8, solid_capstyle="round")
        ax.plot(r.d, y + dy, "o", color=c, ms=5, mec="white", mew=0.6)
labels = [n + (" *" if res[res.model == n].guess.iloc[0] else "") for n in names]
ax.set_yticks(range(len(names)), labels[::-1])
ax.axvline(0, color="#6b6a66", lw=0.8)
ax.grid(axis="x", color="#e6e5e1", lw=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Next amino acid: real minus shuffled accuracy (percentage points), 95% CI over families")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#eb6834", label="well-known: vertebrate proteins in well-studied families (16 families)"),
                   plt.Line2D([], [], marker="o", ls="", color="#2a78d6", label="obscure: non-vertebrate DUF proteins (17 families)")],
          loc="lower right", frameon=False, fontsize=8)
ax.set_title("PROTEIN (pilot, 37 Pfam families): well-known proteins vs obscure ones, every model\n"
             "many models know famous proteins; only Claude Opus 4.8 and later carry it to obscure ones   (* best-guess prompt)",
             loc="left", fontsize=10, fontweight="bold")
Path("results/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("results/figures/fig_common_vs_obscure.png", dpi=150, bbox_inches="tight")
print(res.pivot(index="model", columns="sub", values="d").round(1).loc[names].to_string())
