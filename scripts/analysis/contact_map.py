"""Score the contact-map pilot (scripts/probe/contact_map.py) -> results/contact_map_summary.tsv and
figures/contact_map_pilot.png. Free.

A reply's map is its lines that are only 0 and 1. It is well formed if there are exactly L of them,
each L long; otherwise it is recorded as malformed and not scored (nothing is padded or cropped).
Maps are not symmetrized: only the upper triangle (j > i) is scored, at separations |i-j| >= 6, in
ranges short 6-11, medium 12-23 and long >= 24. For each range: precision of the predicted 1s,
recall, numbers predicted and true, and the random baseline, the true contact density in that range
(the precision of 1s placed at random). The standard metric is long-range precision against it.
Proteins come in two sets, famous (structure likely seen in training) and recent (released after
the assumed training cutoff); pooled rows ("ALL famous", "ALL recent") sum counts over each set.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from bioseq_probe.figstyle import INK, LIGHT_GREY, MUTED, color

MODELS = {"claude-opus-5-5": "Claude Opus 5.5", "claude-fable-5-2": "Claude Fable 5.2", "gpt-6.2-sol": "GPT-6.2 Sol"}
RANGES = {"short": (6, 11), "medium": (12, 23), "long": (24, 10**6), "all": (6, 10**6)}
OUT = Path("results/contact_map")
TRUTH = json.loads((OUT / "truth.json").read_text())


def grid(text: str, n: int) -> tuple[np.ndarray | None, str]:
    """(LxL 0/1 array or None, note on what was wrong)."""
    lines = [s.strip() for s in text.splitlines()]
    rows = [s for s in lines if s and re.fullmatch(r"[01]+", s)]
    lengths = sorted({len(r) for r in rows})
    if len(rows) == n and lengths == [n]:
        return np.array([[int(c) for c in r] for r in rows]), ""
    if not rows:
        return None, "declined: no 0/1 lines"
    return None, f"{len(rows)} 0/1 lines (want {n}), lengths {lengths[:3]}{'...' if len(lengths) > 3 else ''}"


def as_written(text: str, n: int) -> np.ndarray | None:
    """For the figure only: a malformed reply's 0/1 lines as written, rows left-aligned, if they
    are close to a map (within 3 rows and 10 columns of L); None otherwise. Never scored."""
    rows = [s.strip() for s in text.splitlines() if re.fullmatch(r"[01]+", s.strip())]
    if abs(len(rows) - n) > 3 or max(map(len, rows), default=10**6) > n + 10:
        return None
    out = np.zeros((len(rows), max(map(len, rows))), int)
    for k, r in enumerate(rows):
        out[k, : len(r)] = [int(c) for c in r]
    return out


def score(pred: np.ndarray, true: np.ndarray, lo: int, hi: int) -> dict:
    n = len(true)
    i, j = np.triu_indices(n, lo)
    keep = (j - i) <= hi
    p, t = pred[i[keep], j[keep]].astype(bool), true[i[keep], j[keep]].astype(bool)
    tp = int((p & t).sum())
    return {"n_pred": int(p.sum()), "n_true": int(t.sum()), "n_correct": tp, "n_pairs": int(keep.sum()),
            "precision": tp / p.sum() if p.sum() else np.nan, "recall": tp / t.sum() if t.sum() else np.nan,
            "baseline": t.mean()}


rows, maps, notes, drawn = [], {}, {}, {}
for model in MODELS:
    for line in (OUT / f"{model}.jsonl").open():
        r = json.loads(line)
        t = TRUTH[r["protein"]]
        true = np.array([[int(c) for c in s] for s in t["map"]])
        n = len(true)
        pred, note = grid(r.get("text", ""), n) if "error" not in r else (None, "error: " + r["error"][:80])
        maps[model, r["protein"]], notes[model, r["protein"]] = pred, note
        drawn[model, r["protein"]] = pred if pred is not None else as_written(r.get("text", ""), n)
        base = {"set": t["set"], "model": model, "protein": r["protein"], "pdb": r["pdb"], "length": n, "stop": r.get("stop"),
                "well_formed": pred is not None, "note": note,
                "symmetric": bool((pred == pred.T).all()) if pred is not None else None}
        for rng, (lo, hi) in RANGES.items():
            s = score(pred if pred is not None else np.zeros_like(true), true, lo, hi)
            if pred is None:
                s |= {k: np.nan for k in ("n_pred", "n_correct", "precision", "recall")}
            rows.append(base | {"range": rng} | s)
df = pd.DataFrame(rows)
pooled = []
for (group, model, rng), d in df[df.well_formed].groupby(["set", "model", "range"], sort=False):
    c = d[["n_pred", "n_true", "n_correct", "n_pairs"]].sum()
    pooled.append({"set": group, "model": model, "protein": f"ALL {group}", "pdb": "", "length": np.nan,
                   "well_formed": True, "note": f"{len(d)} well-formed of 6", "range": rng} | c.to_dict()
                  | {"precision": c.n_correct / c.n_pred if c.n_pred else np.nan, "recall": c.n_correct / c.n_true,
                     "baseline": c.n_true / c.n_pairs})
df = pd.concat([df, pd.DataFrame(pooled)], ignore_index=True)
df["enrichment"] = df.precision / df.baseline
df.to_csv("results/contact_map_summary.tsv", sep="\t", index=False, float_format="%.4g")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
cols = ["set", "model", "pdb", "length", "stop", "well_formed", "symmetric", "n_pred", "n_true", "n_correct", "precision",
        "recall", "baseline", "enrichment"]
for rng in RANGES:
    print(f"\n== {rng} ==")
    print(df[df.range == rng][cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
for (model, name), note in notes.items():
    if note:
        print(f"not scored: {model} {TRUTH[name]['pdb']}: {note}")

# Figure: famous proteins on the left, recent on the right. In each, a row per protein: the true map,
# then each model's map. Model panels show the true contacts in light grey under the prediction.
SETS = {"famous": "Famous proteins: the model may recall the structure", "recent": "Recent proteins: released after 2026-08-01"}
cols_ = [None, *MODELS]
fig = plt.figure(figsize=(9.6, 7.0))
subs = fig.subfigures(1, 2, wspace=0.03)
for sub, (group, title) in zip(subs, SETS.items()):
    names = [k for k, v in TRUTH.items() if v["set"] == group]
    axes = sub.subplots(len(names), len(cols_))
    sub.subplots_adjust(left=0.13, right=0.99, top=0.925, bottom=0.03, wspace=0.08, hspace=0.3)
    sub.suptitle(title, x=0.02, y=0.995, ha="left", va="top", fontsize=8, fontweight="bold", color=INK)
    for r, name in enumerate(names):
        true = np.array([[int(c) for c in s] for s in TRUTH[name]["map"]])
        n = len(true)
        for c, model in enumerate(cols_):
            ax = axes[r, c]
            ax.imshow(true, cmap=ListedColormap(["white", MUTED if model is None else LIGHT_GREY]), interpolation="nearest",
                      extent=(-0.5, n - 0.5, n - 0.5, -0.5))
            if model is None:
                ax.set_xlabel(f"L = {n}", fontsize=5.6, color=MUTED, labelpad=2)
            elif drawn[model, name] is not None:
                pred = drawn[model, name]
                h, w = pred.shape
                ax.imshow(np.ma.masked_where(pred == 0, pred), cmap=ListedColormap([color(MODELS[model])]), interpolation="nearest",
                          extent=(-0.5, w - 0.5, h - 0.5, -0.5))
                ax.set_xlim(-0.5, max(n, w) - 0.5)
                ax.set_ylim(max(n, h) - 0.5, -0.5)
                if maps[model, name] is None:
                    wide = notes[model, name].split("lengths ")[1].strip("[]").replace(", ", "/")
                    ax.set_xlabel(f"malformed: {h} rows, {wide} wide", fontsize=5.6, color=MUTED, labelpad=2)
                else:
                    lr = df[(df.model == model) & (df.protein == name) & (df.range == "long")].iloc[0]
                    ax.set_xlabel(f"long P {lr.precision:.2f} / {lr.baseline:.2f}" if lr.n_pred else "no long-range 1s",
                                  fontsize=5.6, color=MUTED, labelpad=2)
            else:
                note = notes[model, name]
                label = "declined" if note.startswith("declined") else "error" if note.startswith("error") else "malformed"
                ax.text(0.5, 0.5, label, transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=INK,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(LIGHT_GREY)
                sp.set_linewidth(0.5)
            if r == 0:
                ax.set_title("True" if model is None else MODELS[model], fontsize=6.5, fontweight="bold",
                             color=INK if model is None else color(MODELS[model]), pad=3)
            if c == 0:
                label = name.replace(", ", "\n") if len(name) > 22 else name
                ax.set_ylabel(f"{label}\n{TRUTH[name]['pdb']}", fontsize=5.8, color=INK, labelpad=3)
fig.text(0.01, -0.005, "Grey: true contacts (C-beta within 8 Å). Colour: the model's predicted contacts. Under each map, long P: "
         "precision of the predicted contacts at |i-j| >= 24 / the precision of random guesses (true contact density at that range).\n"
         "Malformed replies (rows of the wrong length or number) are drawn as written, rows left-aligned, and not scored.",
         fontsize=5.8, color=MUTED, ha="left", va="top", linespacing=1.4)
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/contact_map_pilot.png", bbox_inches="tight", pad_inches=0.08)
print("wrote results/contact_map_summary.tsv, figures/contact_map_pilot.png")
