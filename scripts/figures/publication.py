"""The eight figures used in the README and the write-up -> figures/fig1..fig8 (.png, .pdf). Free.

Reads the summary tables written by scripts/analysis and scripts/figures:
results/confirm_tests.tsv, results/raw_summary.tsv, results/dna_confirm_tests.tsv,
results/common_vs_obscure.tsv, results/recall_length_summary.tsv, and for Fig 5 the replies
in results/confirm/ with results/confirm_family_opus55.tsv (Europe PMC hits per family).
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HEROES = {"Claude Opus 5.5", "Claude Fable 5.2"}

from bioseq_probe.figstyle import ACCENT, ACCENT_TEXT, GREY, GRID, INK, LIGHT_GREY, MUTED, PROTEIN_LM, color, header, save

LABEL = {"claude-opus-4-5-20251101": "Claude Opus 4.5", "claude-opus-4-6": "Claude Opus 4.6", "claude-opus-4-7": "Claude Opus 4.7", "claude-opus-4-8": "Claude Opus 4.8", "claude-opus-5": "Claude Opus 5",
         "claude-opus-5-5": "Claude Opus 5.5", "claude-fable-5-2": "Claude Fable 5.2", "claude-fable-5-1": "Claude Fable 5.1", "gpt-5.5": "GPT-5.5", "gpt-6-astra": "GPT-6 Astra", "gemini-4-carbon": "Gemini 4 Carbon",
         "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5", "gemini-3.5-flash": "Gemini 3.5 Flash", "gemini-3.8-flash": "Gemini 3.8 Flash",
         "gpt-6.1-sol": "GPT-6.1 Sol", "gpt-6.2-sol": "GPT-6.2 Sol", "gpt-6.2-astra": "GPT-6.2 Astra"}


def forest(ax, rows, y_of, xlabel):
    """rows: dicts with name, d, lo, hi, raw (bool)."""
    for r in rows:
        y, c = y_of[r["name"]], color(r["name"])
        hero = r["name"] in HEROES
        ax.plot([r["lo"], r["hi"]], [y, y], color=c, lw=2.2 if hero else 1.4, solid_capstyle="round", zorder=4 if hero else 2)
        mk, ms = ("D", 3.8) if r.get("plm") else ("o", 6.0 if hero else 4.2)
        ax.plot(r["d"], y, mk, ms=ms, color=c, mfc="white" if r["raw"] else c, mec=c, mew=1.0, zorder=5 if hero else 3)
        ax.text(max(r["hi"], 0) + 0.8, y, f"{r['d']:+.1f}", va="center", fontsize=7.5 if hero else 6.3,
                color=INK if hero else MUTED, fontweight="bold" if hero else "normal")
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    ax.set_xlabel(xlabel)


# Fig 1: every model on the obscure proteins ---------------------------------------------------
c = pd.read_csv("results/confirm_tests.tsv", sep="\t").set_index("model")
raw = pd.read_csv("results/raw_summary.tsv", sep="\t")
plm = pd.read_csv("results/plm_summary.tsv", sep="\t").query("mode == 'prefix'").set_index("model")
groups = [
    ("Anthropic", [("claude-fable-5-2", "chat"), ("claude-opus-5-5", "chat"), ("claude-fable-5-1", "chat"), ("claude-opus-5", "chat"), ("claude-opus-4-8", "chat"), ("claude-opus-4-7", "chat"),
                   ("claude-opus-4-6", "chat"), ("claude-opus-4-5-20251101", "chat")]),
    ("OpenAI", [("gpt-6.2-sol", "chat"), ("gpt-6.2-astra", "chat"), ("gpt-6.1-sol", "chat"), ("gpt-6-astra", "chat"), ("gpt-5.5", "chat")]),
    ("Google", [("gemini-4-carbon", "chat"), ("Gemini 2.5 Pro (raw)", "raw")]),
    ("Open-weight", [(n, "raw") for n in raw.name if n.endswith("(raw)") and not n.startswith(("Gemini", "Claude"))]),
]
rows, y_of, ticks, y = [], {}, [], 0.0
for g, members in groups:
    ticks.append((g, y))
    y -= 0.85
    for key, how in members:
        if how == "chat":
            r = c.loc[key]
            name = LABEL[key]
            rows.append({"name": name, "d": r.D_points, "lo": r.D_ci_low, "hi": r.D_ci_high, "raw": False})
        elif how == "plm":
            r = plm.loc[key]
            name = r["name"]
            rows.append({"name": name, "d": r.d, "lo": r.lo, "hi": r.hi, "raw": False, "plm": True})
        else:
            r = raw[raw.name == key].iloc[0]
            name = key.replace(" (raw)", "")
            rows.append({"name": name, "d": r.d, "lo": r.lo, "hi": r.hi, "raw": True})
        y_of[name] = y
        y -= 1
    y -= 0.35
fig, ax = plt.subplots(figsize=(4.6, 5.35))
forest(ax, rows, y_of, "Next amino acid: real minus shuffled accuracy (percentage points)")
ax.set_yticks(list(y_of.values()), list(y_of.keys()))
for t, m in zip(ax.get_yticklabels(), y_of):
    if m in HEROES:
        t.set_fontweight("bold")
for g, gy in ticks:
    ax.text(-0.02, gy - 0.25, g.upper(), transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6, color=MUTED, fontweight="bold")
for hero in HEROES:
    ax.axhspan(y_of[hero] - 0.48, y_of[hero] + 0.48, color=ACCENT, alpha=0.09, lw=0, zorder=0)
ax.set_ylim(min(y_of.values()) - 0.8, 0.5)
ax.set_xlim(-3, 52)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=GREY, ms=4.2, label="LLM, chat prompt"),
                   Line2D([], [], marker="o", ls="", color=GREY, mfc="white", ms=4.2, mew=1.0, label="LLM, raw continuation"),
                   ],
          loc="lower right", handletextpad=0.4)
header(fig, ax, "Of 22 language models, only the latest Claude models predict little-studied proteins",
       "668 bacterial and archaeal proteins from 167 unknown-function (DUF) families, no relative at ≥90% identity;\n"
       "points: mean over families, bars: 95% bootstrap CI. Raw continuation: completion endpoint or prefilled reply.\n"
       "Claude Sonnet 4.5 refused 97% of prefill requests (not shown).")
save(fig, "fig1_protein")

# Fig 6: Claude Opus 5.5 against protein language models, same input ---------------------------
cmp = [{"name": LABEL[k], "d": c.loc[k].D_points, "lo": c.loc[k].D_ci_low, "hi": c.loc[k].D_ci_high, "kind": "LLM, chat prompt"}
       for k in ("claude-fable-5-2", "claude-opus-5-5")]
for key, r in plm.iterrows():
    kind = "reads left to right" if key.startswith(("protgpt", "progen")) else "masked model, reads both sides"
    cmp.append({"name": r["name"], "d": r.d, "lo": r.lo, "hi": r.hi, "kind": kind})
cmp = sorted(cmp, key=lambda r: -r["d"])
fig, ax = plt.subplots(figsize=(4.6, 2.6))
for i, r in enumerate(cmp):
    y, hero = -i, r["name"] in HEROES
    col = color(r["name"]) if hero else PROTEIN_LM
    if hero:
        ax.axhspan(y - 0.45, y + 0.45, color=ACCENT, alpha=0.09, lw=0, zorder=0)
    ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=2.2 if hero else 1.4, solid_capstyle="round", zorder=2)
    masked = r["kind"].startswith("masked")
    ax.plot(r["d"], y, "o" if hero else "D", ms=6 if hero else 4, color=col, mfc="white" if masked else col, mec=col, mew=1.0, zorder=3)
    ax.text(r["hi"] + 0.8, y, f"{r['d']:+.1f}", va="center", fontsize=7.5 if hero else 6.3, color=INK if hero else MUTED,
            fontweight="bold" if hero else "normal")
ax.set_yticks([-i for i in range(len(cmp))], [r["name"] for r in cmp])
for t in ax.get_yticklabels():
    if t.get_text() in HEROES:
        t.set_fontweight("bold")
ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
ax.set_xlim(-1, 52)
ax.set_xlabel("Next amino acid: real minus shuffled accuracy (percentage points)")
ax.legend(handles=[Line2D([], [], marker="D", ls="", color=PROTEIN_LM, ms=4, label="protein LM, reads left to right"),
                   Line2D([], [], marker="D", ls="", color=PROTEIN_LM, mfc="white", mew=1.0, ms=4, label="protein LM, masked (normally sees both sides)")],
          loc="lower right", handletextpad=0.4)
header(fig, ax, "Claude Fable 5.2 beats every open protein language model we ran",
       "Same 668 proteins and same input for every model: the residues before the cut. Points: mean over families,\n"
       "bars: 95% bootstrap CI. ESM-2 shown the residues on both sides of the cut reaches +34.6 (Appendix C).")
save(fig, "fig6_protein_lm")

# Fig 7 and 8: every position of the 10 residues asked for -------------------------------------
from bioseq_probe.probe import cut_off, parse, parse_raw, shown_prefix  # noqa: E402

CHAT = ["claude-fable-5-2", "claude-opus-5-5", "claude-fable-5-1", "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5-20251101",
        "gpt-6.2-sol", "gpt-6.2-astra", "gpt-6.1-sol", "gpt-6-astra", "gpt-5.5", "gemini-4-carbon"]
RAW = {"gemini-2.5-pro": "Gemini 2.5 Pro", "deepseek-v4p1-flash": "DeepSeek V4.1 Flash", "qwen3p8-2p4t-a95b": "Qwen 3.8",
       "kimi-k3p1-pro": "Kimi K3.1 Pro", "glm-5p3": "GLM-5.3", "minimax-m3": "MiniMax M3", "gpt-oss-120b": "gpt-oss-120b",
       "nemotron-3-ultra-nvfp4": "Nemotron 3 Ultra"}
LABEL.update(RAW)
confirm_seq = pd.read_csv("data/confirm_sequences.tsv", sep="\t").set_index("accession").sequence


def prefix_of(d):
    return shown_prefix(confirm_seq[d["accession"]], d["accession"], d["condition"], d["cut"])


replies = []  # (model, pfam, condition, hits at positions 1..10)
for model in CHAT + list(RAW):
    for line in open(f"results/{'raw' if model in RAW else 'confirm'}/{model}.jsonl"):
        d = json.loads(line)
        if "error" in d or d.get("stop") == "refusal":
            continue
        pred = parse_raw(d.get("text", "") or "") if model in RAW else parse(d.get("text", "") or "", prefix_of(d), cut=cut_off(d))
        if pred:
            replies.append((model, d["pfam"], d["condition"], [k < len(pred) and pred[k] == d["target"][k] for k in range(10)]))
pos = pd.DataFrame([(m, f, c, k + 1, h[k]) for m, f, c, h in replies for k in range(10)], columns=["model", "pfam", "condition", "k", "hit"])
fam = pos.groupby(["model", "k", "pfam", "condition"]).hit.mean().unstack().dropna()
fam = ((fam.real - fam.shuffled) * 100).rename("gap").reset_index()
rng7 = np.random.default_rng(2026)
summ = []
for (model, k), g in fam.groupby(["model", "k"]):
    x = g.gap.to_numpy()
    b = x[rng7.integers(0, len(x), (10_000, len(x)))].mean(1)
    summ.append({"model": model, "position": k, "d": x.mean(), "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5)})
summ = pd.DataFrame(summ)
summ.to_csv("results/position_gaps.tsv", sep="\t", index=False, float_format="%.4f")
runs = {(m, c): np.mean([[all(h[: k + 1]) for k in range(10)] for mm, _, cc, h in replies if mm == m and cc == c], axis=0) * 100
        for m in CHAT + list(RAW) for c in ("real", "shuffled")}
pd.DataFrame({f"{m}|{c}": v for (m, c), v in runs.items()}, index=range(1, 11)).rename_axis("k").to_csv(
    "results/exact_runs.tsv", sep="\t", float_format="%.4f")
ORDER = CHAT + list(RAW)


def style(model):
    hero = LABEL[model] in HEROES
    return dict(color=color(LABEL[model]), lw=1.8 if hero else (0.8 if model in RAW else 1.0), ms=3.2 if hero else 2.2,
                mfc="white" if model in RAW else color(LABEL[model]), zorder=3 if hero else (1 if model in RAW else 2))


def legend_handles(models):
    hs = [Line2D([], [], marker="o", **{k: v for k, v in style(m).items() if k != "zorder"}, label=LABEL[m]) for m in models]
    hs.append(Line2D([], [], marker="o", color=GREY, lw=0.8, ms=2.2, mfc="white", label="7 open-weight models"))
    return hs


fig, ax = plt.subplots(figsize=(4.0, 3.3))
for model in ORDER[::-1]:
    r = summ[summ.model == model].sort_values("position")
    if LABEL[model] in HEROES:
        ax.fill_between(r.position, r.lo, r.hi, color=color(LABEL[model]), alpha=0.12, lw=0)
    ax.plot(r.position, r.d, "-o", **style(model))
ax.axhline(0, color=MUTED, lw=0.6)
ax.set_xticks(range(1, 11))
ax.set_ylim(-6, 52)
ax.set_xlabel("Position in the reply (1 = the next amino acid)")
ax.set_ylabel("Real minus shuffled (points)")
ax.grid(axis="y", color=GRID, lw=0.5)
ax.grid(axis="x", visible=False)
ax.legend(handles=legend_handles(CHAT + ["gemini-2.5-pro"]), loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=5.8,
          handletextpad=0.3, labelspacing=0.32)
header(fig, ax, "The latest Claude models keep their signal across all 10 residues they write",
       "668 little-studied proteins; each position scored on its own (earlier positions need not be right); mean over 167 families,\n"
       "bands: 95% bootstrap CI for Fable 5.2 and Opus 5.5. Hollow points: raw continuation of the FASTA text (Gemini 2.5 Pro, open-weight models).")
save(fig, "fig7_by_position")

fig, ax = plt.subplots(figsize=(4.6, 3.3))
for model in ORDER[::-1]:
    ax.plot(range(1, 11), runs[(model, "real")], "-o", **style(model))
ax.plot(range(1, 11), runs[("claude-opus-5-5", "shuffled")], "--o", color=ACCENT, lw=1.2, ms=2.8, mfc="white", zorder=3)
hs = legend_handles(CHAT + ["gemini-2.5-pro"])
hs.append(Line2D([], [], marker="o", ls="--", color=ACCENT, lw=1.2, ms=2.8, mfc="white", label="Claude Opus 5.5, shuffled proteins"))
ax.legend(handles=hs, loc="upper right", ncol=2, fontsize=5.6, handletextpad=0.3, labelspacing=0.3, columnspacing=0.8)
ax.set_xticks(range(1, 11))
ax.set_ylim(-1.5, 58)
ax.set_xlabel("Number of residues, starting from the next one")
ax.set_ylabel("All of them right (%)")
ax.grid(axis="y", color=GRID, lw=0.5)
ax.grid(axis="x", visible=False)
header(fig, ax, "The latest Claude models write runs of correct residues; on shuffled proteins they cannot",
       "668 little-studied proteins. Share of replies whose first k residues are all right. On real proteins, Fable 5.2 gets\n"
       "11% to 5 in a row and 2.6% to all 10, Opus 5.5 3.0% and 0.4%; on shuffled proteins neither gets past 3.\n"
       "Hollow points: raw continuation of the FASTA text (Gemini 2.5 Pro, open-weight models).")
save(fig, "fig8_runs")

# Fig 2: DNA by region ------------------------------------------------------------------------
t = pd.read_csv("results/dna_confirm_tests.tsv", sep="\t")
cells = [("human cds", "Human coding (CDS)"), ("mouse cds", "Mouse coding (CDS)"), ("bacteria/archaea cds", "Bacterial coding (CDS)"),
         ("human promoter", "Human promoter"), ("mouse promoter", "Mouse promoter"), ("human intron", "Human intron"),
         ("mouse intron", "Mouse intron"), ("human intergenic", "Human intergenic"), ("mouse intergenic", "Mouse intergenic")]
models = [("claude-opus-5-5", 0.22), ("claude-opus-5", 0.0), ("gpt-5.5", -0.22)]
fig, ax = plt.subplots(figsize=(4.6, 3.6))
for i, (cell, name) in enumerate(cells):
    y = -i - (0.5 if i >= 3 else 0)
    for m, dy in models:
        r = t[(t.model == m) & (t.cell == cell)].iloc[0]
        col = color(LABEL[m])
        hollow = m == "claude-opus-5"
        ax.plot([r.lo, r.hi], [y + dy, y + dy], color=col, lw=1.0 if hollow else 1.3, solid_capstyle="round", alpha=0.75 if hollow else 1)
        ax.plot(r.D, y + dy, "o", ms=3.6, color=col, mfc="white" if hollow else col, mec=col if hollow else "white", mew=0.9 if hollow else 0.4)
        if m == "claude-opus-5-5" and r.exact20_real:
            ax.text(r.hi + 1, y + dy, f"{int(r.exact20_real)} exact 20-nt runs", va="center", fontsize=6, color=ACCENT_TEXT)
ax.set_yticks([-i - (0.5 if i >= 3 else 0) for i in range(len(cells))], [n for _, n in cells])
ax.axvline(0, color=MUTED, lw=0.6)
ax.set_xlim(-12, 46)
ax.set_xlabel("Next nucleotide: real minus dinucleotide-shuffled accuracy (percentage points)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=color(LABEL[m]), mfc="white" if m == "claude-opus-5" else color(LABEL[m]),
                          mew=0.9, ms=4, label=LABEL[m]) for m, _ in models],
          loc="lower right", handletextpad=0.3)
header(fig, ax, "On DNA, Claude Opus 5.5 predicts human and mouse coding sequence only",
       "500 repeat-free windows per region (hg38, mm39, and the genes of the proteins in Fig. 1); 200 nt shown, next nucleotide scored;\n"
       "95% bootstrap CI over windows. An exact 20-nt continuation has a chance probability of about 1e-12.")
save(fig, "fig2_dna")

# Fig 3: well-known vs obscure proteins -------------------------------------------------------
k = pd.read_csv("results/common_vs_obscure.tsv", sep="\t")
wide = k.pivot(index="model", columns="sub", values="d")
pilot = pd.read_csv("data/sequences_pilot.tsv", sep="\t").set_index("accession")
vertebrate = pd.read_csv("data/sequence_groups.tsv", sep="\t").set_index("accession").group == "Vertebrata"


def pilot_subset(acc):  # as in scripts/figures/common_vs_obscure.py
    lit = pilot.loc[acc, "lit_cell"]
    return "well-known" if vertebrate[acc] and lit == "rich" else ("obscure" if not vertebrate[acc] and lit == "poor" else None)


for name, path in (("ProtGPT3-10B", "results/plm_pilot/protgpt3-10B__prefix.jsonl"), ("ProtGPT2", "results/plm/protgpt2.jsonl")):
    d = pd.DataFrame(json.loads(line) for line in open(path))
    d["sub"] = d.accession.map(pilot_subset)
    fam = d.groupby(["sub", "pfam", "condition"]).first_correct.mean().unstack().dropna()
    wide.loc[name] = ((fam.real - fam.shuffled) * 100).groupby("sub").mean()
order = wide.sort_values("well-known").index.tolist()
fig, ax = plt.subplots(figsize=(4.6, 5.0))
for y, m in enumerate(order):
    a, b = wide.loc[m, "well-known"], wide.loc[m, "obscure"]
    col, mk, ms = color(m), ("D" if m.startswith("Prot") else "o"), (3.9 if m.startswith("Prot") else 4.2)
    ax.plot([b, a], [y, y], color=LIGHT_GREY, lw=1.2, zorder=1)
    ax.plot(a, y, mk, ms=ms, color="white", mec=col, mew=1.1, zorder=2)
    ax.plot(b, y, mk, ms=ms, color=col, mec=col, zorder=3)
guess = dict(zip(k.model, k.guess))
ax.set_yticks(range(len(order)), [m + (" *" if guess.get(m) else "") for m in order])
for tl in ax.get_yticklabels():
    if tl.get_text().startswith("Prot"):
        tl.set_color(PROTEIN_LM)
for hero in ("Claude Opus 5.5", "Claude Fable 5.2"):
    ax.axhspan(order.index(hero) - 0.45, order.index(hero) + 0.45, color=ACCENT, alpha=0.09, lw=0, zorder=0)
for tl in ax.get_yticklabels():
    if tl.get_text() in ("Claude Opus 5.5", "Claude Fable 5.2"):
        tl.set_fontweight("bold")
ax.axvline(0, color=MUTED, lw=0.6)
ax.set_xlabel("Next amino acid: real minus shuffled accuracy (percentage points)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color="white", mec=MUTED, mew=1.1, ms=4.2, label="well-known: vertebrate proteins, well-studied families"),
                   Line2D([], [], marker="o", ls="", color=MUTED, ms=4.2, label="obscure: non-vertebrate DUF proteins")],
          loc="lower right", handletextpad=0.3)
header(fig, ax, "On obscure proteins, the latest Claude models hold up like a protein language model",
       "Pilot set, 37 Pfam families (16 well-known, 17 obscure); means over families. * best-guess prompt (the model declined\n"
       "the plain one). Purple diamonds: protein language models given the same residues before the cut.")
save(fig, "fig3_known_vs_obscure")

# Fig 4: letter-for-letter recitation of well-known proteins ----------------------------------
s = pd.read_csv("results/recall_length_summary.tsv", sep="\t").set_index("model")
s.index = [LABEL.get(m, m) for m in s.index]
s = s.sort_values(">= 100")
fig, ax = plt.subplots(figsize=(4.6, 2.9))
shades = {">= 50": "#E6E9EF", ">= 100": "#C5CBD6", ">= 200": "#8B93A3"}
ys = range(len(s))
for col_name, shade in shades.items():
    ax.barh(list(ys), s[col_name], height=0.62, color=[color(m) if col_name == ">= 200" and color(m) != GREY else shade for m in s.index],
            edgecolor="white", linewidth=0.4, label=f"exact run {col_name.replace('>=', '≥')} residues", zorder=2)
for y, m in zip(ys, s.index):
    ax.text(s.loc[m, ">= 50"] + 0.8, y, f"longest {s.loc[m, 'longest']}", va="center", fontsize=6, color=MUTED)
ax.set_yticks(list(ys), list(s.index))
ax.set_xlabel("Cut points continued letter for letter (of 158)")
ax.legend(loc="lower right", handletextpad=0.4, reverse=True)
header(fig, ax, "Several models recite hundreds of residues of famous proteins",
       "Cut points that any model continued 10/10 in the pilot; asked for the next 200 residues. Bars overlap: each darker bar is a subset; the \u2265200 segment in each lab's colour.")
save(fig, "fig4_recall")

# Fig 5: the gap by family size, 20 to 30,000+ proteins -------------------------------------------
from pathlib import Path  # noqa: E402

seqs = pd.read_csv("data/confirm_sequences.tsv", sep="\t")
small = pd.read_csv("data/small_family_sequences.tsv", sep="\t")
bins = ["20-60", "60-150", "150-300", "300-1000", "1000-3000", "3000-10000", "10000-max"]
bin_of = pd.concat([small.groupby("pfam").size_bin.first(), seqs.groupby("pfam").size_bin.first()])
papers = pd.read_csv("results/confirm_family_opus55.tsv", sep="\t").set_index("pfam").epmc
SMALL_N = 2 * 5 * len(small)  # questions in a finished small-family run


sequence_of = pd.concat([seqs.set_index("accession").sequence, small.set_index("accession").sequence])


def family_gaps(*paths):
    """Per-family gap. LLM replies are scored from their text with the current parser; protein
    language model records (results/plm_*) keep their stored scores."""
    rows = []
    for path in paths:
        if not Path(path).exists() or ("small" in path and sum(1 for _ in open(path)) < SMALL_N):
            continue  # a small-family run still going is left out
        recs = [json.loads(line) for line in open(path)]
        if "/plm" not in path:
            recs = [{**r, "valid": True, "first_correct": p[0] == r["target"][0]} for r in recs
                    if "error" not in r and r.get("stop") != "refusal"
                    and (p := parse(r.get("text", "") or "", shown_prefix(sequence_of[r["accession"]], r["accession"], r["condition"], r["cut"]),
                                    cut=cut_off(r)))]
        rows += recs
    d = pd.DataFrame(rows)
    if "valid" in d:
        d = d[d.valid.astype(bool) & (d.get("stop") != "refusal")]
    fam = d.groupby(["pfam", "condition"]).first_correct.mean().unstack().dropna()
    return (fam.real - fam.shuffled) * 100


def by_bin(g, rng=np.random.default_rng(0)):
    out = []
    for b in bins:
        x = g[bin_of.reindex(g.index) == b].to_numpy()
        if len(x) < 5:
            out.append((np.nan, np.nan, np.nan))
            continue
        boot = x[rng.integers(0, len(x), (10_000, len(x)))].mean(1)
        out.append((x.mean(), *np.percentile(boot, [2.5, 97.5])))
    return np.array(out)


LABEL.update({"protgpt3-10B": "ProtGPT3-10B", "progen3-3b": "ProGen3-3B"})
PLM5 = {"protgpt3-10B": PROTEIN_LM, "progen3-3b": "#C39BD3"}  # protein language models, purple diamonds
HERO5 = ("claude-opus-5-5", "claude-fable-5-2", *PLM5)


def family_size_figure(models, name, title, subtitle, size, scale=1.0, faint=(), background=(), background_label=""):
    fig, ax = plt.subplots(figsize=size)
    xs = np.arange(len(bins))
    ends = []
    bg_ends = []
    for m in background:  # thin grey lines under one shared label
        r = by_bin(family_gaps(f"results/confirm/{m}.jsonl", f"results/small_families/{m}.jsonl"))
        ax.plot(xs, r[:, 0], "-", color=GREY, lw=0.8, alpha=0.6, zorder=1)
        bg_ends.append(r[-1, 0])
    if background:
        LABEL[background_label] = background_label
        ends.append([float(np.nanmean(bg_ends)), background_label])
    for m in models:
        if m in PLM5:
            r = by_bin(family_gaps(f"results/plm_confirm/{m}__prefix.jsonl", f"results/plm_small/{m}__prefix.jsonl"))
            ax.plot(xs, r[:, 0], "-D", color=PLM5[m], lw=1.6 * scale, ms=3.6 * scale, zorder=4)
        else:
            r = by_bin(family_gaps(f"results/confirm/{m}.jsonl", f"results/small_families/{m}.jsonl"))
            col, top = color(LABEL[m]), m in HERO5
            if top:
                ok = ~np.isnan(r[:, 0])
                ax.fill_between(xs[ok], r[ok, 1], r[ok, 2], color=col, alpha=0.12, lw=0)
            if m in faint:
                ax.plot(xs, r[:, 0], "-o", color=col, lw=0.9, ms=2.2, alpha=0.75, zorder=2)
            else:
                ax.plot(xs, r[:, 0], "-o", color=col, lw=(1.6 if top else 1.0) * scale, ms=(3.6 if top else 2.8) * scale, zorder=3 if top else 2)
        ends.append([r[-1, 0], m])
    ends.sort()
    gap = 2.6 * scale
    orig = [y for y, _ in ends]
    for i in range(1, len(ends)):  # keep end labels apart ...
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    start = 0
    for i in range(1, len(ends) + 1):  # ... and centre each pushed-apart run on its lines
        if i == len(ends) or ends[i][0] - ends[i - 1][0] > gap + 0.01:
            shift = np.mean(orig[start:i]) - np.mean([y for y, _ in ends[start:i]])
            for e in ends[start:i]:
                e[0] += shift
            start = i
    for y, m in ends:
        top = m in HERO5
        ax.text(xs[-1] + 0.15, y, LABEL[m], va="center", fontsize=6.0 * (scale if top else 1.0), color=(PROTEIN_LM if m in PLM5 else INK) if top else MUTED,
                fontweight="bold" if top else "normal", bbox={"fc": "white", "ec": "none", "pad": 0.4})
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(2.5, color=LIGHT_GREY, lw=0.8, ls=":", zorder=0)
    ax.text(2.42, 54, "small-family set", ha="right", va="top", fontsize=5.8, color=MUTED)
    ax.text(2.58, 54, "main set", ha="left", va="top", fontsize=5.8, color=MUTED)
    ax.set_ylim(-9.5 if len(models) > 3 else -3, 56)
    ax.set_xticks(xs, ["20–60", "60–150", "150–300", "300–1k", "1k–3k", "3k–10k", ">10k"])
    ax.set_xlim(-0.3, len(bins) - 1 + 1.6)
    ax.set_xlabel("Proteins in the family (Pfam)")
    ax.set_ylabel("Real minus shuffled (points)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.grid(axis="x", visible=False)
    header(fig, ax, title, subtitle)
    save(fig, name)


SETS5 = ("Next amino acid on 257 DUF families: 90 from 20 to 300 proteins, 30 per bin, and the 167 of Fig. 1; mean over families,\n"
         "bands: 95% bootstrap CI for Fable 5.2 and Opus 5.5. Purple diamonds: protein language models (ProtGPT3-10B, ProGen3-3B) on the same residues.")
EARLIER = ("claude-opus-4-5-20251101", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-fable-5-1")
family_size_figure([*EARLIER, "claude-opus-5-5", "claude-fable-5-2", *PLM5], "fig5_family_size",
                   "The bigger the protein family, the better the latest Claude models, as for a protein language model",
                   SETS5 + "\nThin lines: earlier Claude models; grey: GPT-6.2 Sol and Astra, GPT-6.1 Sol, GPT-6 Astra, GPT-5.5 and Gemini 4 Carbon.", (5.2, 3.4), 1.25,
                   faint=EARLIER, background=("gpt-6.2-sol", "gpt-6.2-astra", "gpt-6.1-sol", "gpt-6-astra", "gpt-5.5", "gemini-4-carbon"), background_label="GPT and Gemini")
print("figures/fig1_protein, fig2_dna, fig3_known_vs_obscure, fig4_recall, fig5_family_size, fig6_protein_lm, fig7_by_position, fig8_runs")

# Fig 9: proteins first public after the assumed training cutoff (the D set) ------------------
dset = pd.read_csv("results/d_set_summary.tsv", sep="\t")
order9 = ["Claude Fable 5.2", "Claude Opus 5.5", "ProGen3-3B", "ProtGPT3-10B", "GPT-6.2 Sol"]
dset = dset.set_index("model").reindex([m for m in order9 if m in set(dset.model)])
fig, ax = plt.subplots(figsize=(4.6, 0.42 * len(dset) + 1.1))
for i, (m, r) in enumerate(dset.iterrows()):
    y, col, mk = -i, color(m), ("D" if m.startswith("Pro") else "o")
    if m in HEROES:
        ax.axhspan(y - 0.45, y + 0.45, color=ACCENT, alpha=0.09, lw=0, zorder=0)
    ax.plot([r.D_gap, r.main_same_families_gap], [y, y], color=LIGHT_GREY, lw=1.2, zorder=1)
    ax.plot([r.D_lo, r.D_hi], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2)
    ax.plot(r.main_same_families_gap, y, mk, ms=4.6, color="white", mec=col, mew=1.1, zorder=3)
    ax.plot(r.D_gap, y, mk, ms=4.8, color=col, mec=col, zorder=4)
    ax.text(r.D_lo - 1.2, y, f"{r.D_gap:+.1f}", va="center", ha="right", fontsize=6.6, color=INK if m in HEROES else MUTED,
            fontweight="bold" if m in HEROES else "normal", bbox=None if m in HEROES else {"fc": "white", "ec": "none", "pad": 0.3})
ax.set_yticks([-i for i in range(len(dset))], list(dset.index))
for tl in ax.get_yticklabels():
    if tl.get_text() in HEROES:
        tl.set_fontweight("bold")
    if tl.get_text().startswith("Pro"):
        tl.set_color(PROTEIN_LM)
ax.axvline(0, color=MUTED, lw=0.6)
ax.set_xlim(-13, 50)
ax.set_ylim(-len(dset) - 0.5, 0.6)
ax.set_xlabel("Next amino acid: real minus shuffled accuracy (percentage points)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, ms=4.6, label="56 proteins first public in August 2026, no earlier relative at 50%+"),
                   Line2D([], [], marker="o", ls="", color="white", mec=MUTED, mew=1.1, ms=4.6, label="the same 31 families in the main set")],
          loc="lower right", handletextpad=0.3, fontsize=6)
header(fig, ax, "The latest Claude models predict proteins published after their training cutoff",
       "D set (Leo Tianlai Chen): RefSeq proteins first public 2 to 27 August 2026, after the assumed 1 July cutoff; closest earlier\n"
       "relative 34% to 49% identical. Mean over 31 families; bars: 95% bootstrap CI. Both protein language models drop as much from the main set.")
save(fig, "fig9_after_cutoff")
