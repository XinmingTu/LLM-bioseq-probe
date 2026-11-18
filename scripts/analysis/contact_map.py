"""Score the contact-map runs (scripts/probe/contact_map.py, scripts/probe/esm_contacts_modal.py) ->
results/contact_map_summary.tsv, results/contact_map_groups.tsv, results/contact_map_shift.tsv,
results/contact_map_stability.tsv and figures contact_map_pilot.png, contact_map_fable_vs_esm.png,
contact_map_clean_examples.png. Free.

Replies. A grid reply's map is its lines that are only 0 and 1; it is well formed if there are
exactly L of them, each L long, and is otherwise not scored (nothing is padded or cropped). A list
reply's map is its lines of two integers "i j"; pairs out of range or closer than 6 are dropped
(and counted); it is not scored if it has no pair. Maps are not symmetrized: only the upper
triangle (j > i) is scored, at separations |i-j| >= 6, in ranges short 6-11, medium 12-23 and long
>= 24. For each range: precision of the predicted contacts, recall, numbers predicted and true, and
the random baseline, the true contact density in that range (the precision of contacts placed at
random). The headline number is long-range precision against it.

ESM-2 gives probabilities, so it is scored on its top-ranked long-range pairs: P@L/5, P@L, and P@N
with N the number of long-range contacts a given LLM reply predicted for that protein (matched N).
The LLMs' contacts are unranked, so they are scored at their own count. A sequence-only straw man,
every pair of hydrophobic residues called a contact ("hydrophobic"), is scored the same way.

Proteins: the pilot's six famous ones (positive control) and the recent ones, the pilot's six and
the expanded set's 40 (scripts/data/08_contact_map_set.py, 09_contact_map_organism.py), each in a
novelty group: "clean" (no earlier PDB relative at e < 1e-3, sequence first public on or after
2026-07-01 in UniProtKB, UniParc and its organism's UniParc records), "new sequence,
PDB relative", "old sequence" (no PDB relative, sequence public before July 2026) and "old
sequence, PDB relative". Pooled rows sum counts over a group's well-formed replies.

Chance. Per protein, the long-range hits of n random pairs are hypergeometric; per group, the
p-value is P(sum >= observed) under the exact distribution of the sum (convolution). A stricter
null keeps each predicted pair's separation: a pair at separation s is a hit with probability equal
to the true contact density at separation s in that protein (Poisson-binomial). Shift test: the
predicted long-range pairs are moved along the diagonal, (i+k, j+k), and scored again; if the
precision stays up when the map is moved by several residues, the hits come from a generic pattern
rather than from where the contacts are in this sequence.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.stats import hypergeom

from bioseq_probe.figstyle import INK, LIGHT_GREY, MUTED, PROTEIN_LM, color

OUT = Path("results/contact_map")
TRUTH = json.loads((OUT / "truth.json").read_text())
PROT = pd.read_csv("data/contact_map_proteins.tsv", sep="\t").set_index("pdb")
RANGES = {"short": (6, 11), "medium": (12, 23), "long": (24, 10**6), "all": (6, 10**6)}
SOURCES = {  # key: (file, prompt, display name)
    "fable_grid": ("claude-fable-5-2", "grid", "Claude Fable 5.2, grid"),
    "fable_grid_run2": ("claude-fable-5-2__grid_run2", "grid", "Claude Fable 5.2, grid, run 2"),
    "fable_list": ("claude-fable-5-2__list", "list", "Claude Fable 5.2, list"),
    "opus_grid": ("claude-opus-5-5", "grid", "Claude Opus 5.5, grid"),
    "opus_list": ("claude-opus-5-5__list", "list", "Claude Opus 5.5, list"),
    "gpt_grid": ("gpt-6.2-sol", "grid", "GPT-6.2 Sol, grid"),
}
ESM = {"esm2_650M": ("esm2_t33_650M", "ESM-2 650M"), "esm2_3B": ("esm2_t36_3B", "ESM-2 3B")}
GROUPS = ["famous", "clean", "new sequence, PDB relative", "old sequence", "old sequence, PDB relative"]
POOLS = {g: {g} for g in GROUPS} | {"no earlier PDB relative (clean + old sequence)": {"clean", "old sequence"},
                                      "all recent": set(GROUPS) - {"famous"}}
GROUP_COLOR = {"famous": "#6B7385", "clean": "#C2410C", "new sequence, PDB relative": "#E9A23B", "old sequence": "#2563EB",
               "old sequence, PDB relative": "#93C5FD"}
SHIFTS = [-20, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 20]


def group_of(name: str) -> str:
    t = TRUTH[name]
    return "famous" if t["set"] == "famous" else PROT.loc[t["pdb"], "group"]


def grid(text: str, n: int) -> tuple[np.ndarray | None, str]:
    """(LxL 0/1 array or None, note on what was wrong)."""
    rows = [s for s in (x.strip() for x in text.splitlines()) if s and re.fullmatch(r"[01]+", s)]
    lengths = sorted({len(r) for r in rows})
    if len(rows) == n and lengths == [n]:
        return np.array([[int(c) for c in r] for r in rows]), ""
    if not rows:
        return None, "declined: no 0/1 lines"
    return None, f"{len(rows)} 0/1 lines (want {n}), lengths {lengths[:3]}{'...' if len(lengths) > 3 else ''}"


def pairs(text: str, n: int) -> tuple[np.ndarray | None, str]:
    """A list reply as an upper-triangle 0/1 map, or None, and a note on what was dropped."""
    got = [tuple(map(int, m)) for m in re.findall(r"^\s*(\d+)\s*[ ,\t-]\s*(\d+)\s*$", text, re.M)]
    other = sum(1 for s in text.splitlines() if s.strip() and not re.fullmatch(r"\s*\d+\s*[ ,\t-]\s*\d+\s*", s))
    if not got:
        return None, "declined: no pairs" if not other else f"no pairs; {other} other lines"
    out = np.zeros((n, n), int)
    bad = close = 0
    for i, j in got:
        i, j = min(i, j), max(i, j)
        if not 1 <= i < j <= n:
            bad += 1
        elif j - i < 6:
            close += 1
        else:
            out[i - 1, j - 1] = 1
    note = ", ".join(f"{k} {w}" for k, w in ((bad, "out of range"), (close, "closer than 6"), (other, "other lines"),
                                             (len(got) - bad - close - int(out.sum()), "repeated")) if k)
    return out, note


def as_written(text: str, n: int) -> np.ndarray | None:
    """For the pilot figure only: a malformed grid's 0/1 lines as written, rows left-aligned, if
    they are close to a map (within 3 rows and 10 columns of L); None otherwise. Never scored."""
    rows = [s.strip() for s in text.splitlines() if re.fullmatch(r"[01]+", s.strip())]
    if abs(len(rows) - n) > 3 or max(map(len, rows), default=10**6) > n + 10:
        return None
    out = np.zeros((len(rows), max(map(len, rows))), int)
    for k, r in enumerate(rows):
        out[k, : len(r)] = [int(c) for c in r]
    return out


def band(n: int, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    i, j = np.triu_indices(n, lo)
    keep = (j - i) <= hi
    return i[keep], j[keep]


def score(pred: np.ndarray, true: np.ndarray, lo: int, hi: int) -> dict:
    i, j = band(len(true), lo, hi)
    p, t = pred[i, j].astype(bool), true[i, j].astype(bool)
    tp, npred, ntrue = int((p & t).sum()), int(p.sum()), int(t.sum())
    sep = j - i
    density = {s: true[i[sep == s], j[sep == s]].mean() for s in np.unique(sep[p])}  # true density at each predicted separation
    return {"n_pred": npred, "n_true": ntrue, "n_correct": tp, "n_pairs": len(i),
            "precision": tp / npred if npred else np.nan, "recall": tp / ntrue if ntrue else np.nan, "baseline": t.mean(),
            "expected_sep": float(sum(density[s] for s in sep[p])), "p_hyper": hypergeom.sf(tp - 1, len(i), ntrue, npred) if npred else np.nan,
            "sep_probs": [density[s] for s in sep[p]]}


def shifted(pred: np.ndarray, true: np.ndarray, k: int) -> tuple[int, int]:
    """(hits, pairs) of the predicted long-range pairs moved by k along the diagonal; pairs moved off the map are dropped."""
    i, j = band(len(true), 24, 10**6)
    sel = pred[i, j].astype(bool)
    i, j = i[sel] + k, j[sel] + k
    ok = (i >= 0) & (j < len(true))
    return int(true[i[ok], j[ok]].sum()), int(ok.sum())


def esm_scores(probs: np.ndarray, true: np.ndarray, ns: dict[str, int]) -> dict:
    """ESM's long-range precision at its top L/5, L, and at each matched N."""
    i, j = band(len(true), 24, 10**6)
    order = np.argsort(-probs[i, j], kind="stable")
    hits = true[i, j][order]
    n = len(true)
    out = {"P@L/5": hits[: max(1, n // 5)].mean(), "P@L": hits[:n].mean()}
    for key, k in ns.items():
        out[f"P@N {key}"] = hits[:k].mean() if k else np.nan
        out[f"hits@N {key}"] = int(hits[:k].sum()) if k else 0
    return out


def sum_pvalue(parts: list[tuple[int, int, int]], observed: int) -> float:
    """P(sum of independent hypergeometric hits >= observed); parts are (pairs, true, predicted)."""
    dist = np.array([1.0])
    for m, t, k in parts:
        dist = np.convolve(dist, hypergeom.pmf(np.arange(min(t, k) + 1), m, t, k))
    return float(dist[observed:].sum())


def poisson_binomial_pvalue(probs: list[float], observed: int) -> float:
    dist = np.array([1.0])
    for p in probs:
        dist = np.convolve(dist, [1 - p, p])
    return float(dist[observed:].sum())


# --- score every reply -------------------------------------------------------------------------
true_maps = {name: np.array([[int(c) for c in s] for s in t["map"]]) for name, t in TRUTH.items()}
rows, maps, notes, drawn, sep_probs = [], {}, {}, {}, {}
for key, (fname, prompt, label) in SOURCES.items():
    path = OUT / f"{fname}.jsonl"
    if not path.exists():
        continue
    for line in path.open():
        r = json.loads(line)
        if r["protein"] not in TRUTH:
            continue
        true = true_maps[r["protein"]]
        n = len(true)
        if "error" in r:
            pred, note = None, "error: " + r["error"][:80]
        else:
            pred, note = (grid if prompt == "grid" else pairs)(r.get("text", ""), n)
        maps[key, r["protein"]], notes[key, r["protein"]] = pred, note
        drawn[key, r["protein"]] = pred if pred is not None or prompt == "list" else as_written(r.get("text", ""), n)
        base = {"source": key, "model": r["model"], "prompt": prompt, "group": group_of(r["protein"]), "set": TRUTH[r["protein"]]["set"],
                "protein": r["protein"], "pdb": r["pdb"], "length": n, "stop": r.get("stop"), "scored": pred is not None, "note": note}
        for rng, (lo, hi) in RANGES.items():
            s = score(pred if pred is not None else np.zeros_like(true), true, lo, hi)
            sep_probs[key, r["protein"], rng] = s.pop("sep_probs")
            if pred is None:
                s |= {k: np.nan for k in ("n_pred", "n_correct", "precision", "recall", "expected_sep", "p_hyper")}
            rows.append(base | {"range": rng} | s)
# A sequence-only straw man: every pair of hydrophobic residues (AVILMFWC) is called a contact.
for name, t in TRUTH.items():
    true = true_maps[name]
    h = np.array([c in "AVILMFWC" for c in t["sequence"]])
    pred = np.triu(np.outer(h, h).astype(int), 1)
    maps["hydrophobic", name], notes["hydrophobic", name] = pred, ""
    base = {"source": "hydrophobic", "model": "hydrophobic pairs", "prompt": "baseline", "group": group_of(name), "set": t["set"],
            "protein": name, "pdb": t["pdb"], "length": len(true), "scored": True, "note": ""}
    for rng, (lo, hi) in RANGES.items():
        sc = score(pred, true, lo, hi)
        sep_probs["hydrophobic", name, rng] = sc.pop("sep_probs")
        rows.append(base | {"range": rng} | sc)
df = pd.DataFrame(rows)

# ESM-2: per protein, long-range P@L/5, P@L, and P@N matched to each LLM reply's long-range count.
long = df[(df.range == "long") & df.scored].set_index(["source", "protein"])
esm_rows = []
for key, (fname, label) in ESM.items():
    path = OUT / f"{fname}.jsonl"
    if not path.exists():
        continue
    for line in path.open():
        r = json.loads(line)
        if r["protein"] not in TRUTH:
            continue
        true = true_maps[r["protein"]]
        probs = np.array(r["probs"])
        ns = {s: int(long.loc[(s, r["protein"]), "n_pred"]) for s in SOURCES if (s, r["protein"]) in long.index}
        e = esm_scores(probs, true, ns)
        b = score(np.zeros_like(true), true, 24, 10**6)
        esm_rows.append({"source": key, "model": fname, "prompt": "", "group": group_of(r["protein"]), "set": TRUTH[r["protein"]]["set"],
                         "protein": r["protein"], "pdb": r["pdb"], "length": len(true), "scored": True, "range": "long",
                         "n_true": b["n_true"], "n_pairs": b["n_pairs"], "baseline": b["baseline"]} | e)
        maps[key, r["protein"]] = probs
esm = pd.DataFrame(esm_rows)

# --- pooled by group ---------------------------------------------------------------------------
pooled, group_rows = [], []
for (grp, key, rng), d in df[df.scored].groupby(["group", "source", "range"], sort=False):
    c = d[["n_pred", "n_true", "n_correct", "n_pairs"]].sum()
    pooled.append({"source": key, "model": d.model.iloc[0], "prompt": d.prompt.iloc[0], "group": grp, "protein": f"ALL {grp}",
                   "scored": True, "note": f"{len(d)} scored of {len(df[(df.group == grp) & (df.source == key) & (df.range == rng)])}",
                   "range": rng} | c.to_dict() | {"precision": c.n_correct / c.n_pred if c.n_pred else np.nan, "recall": c.n_correct / c.n_true if c.n_true else np.nan,
                                                   "baseline": c.n_true / c.n_pairs, "expected_sep": d.expected_sep.sum()})
summary = pd.concat([df, pd.DataFrame(pooled), esm], ignore_index=True)
summary["enrichment"] = summary.precision / summary.baseline
summary.to_csv("results/contact_map_summary.tsv", sep="\t", index=False, float_format="%.4g")

for grp, members in POOLS.items():
    n_prot = sum(group_of(p) in members for p in TRUTH)
    for key in [*SOURCES, "hydrophobic"]:
        d = df[df.group.isin(members) & (df.source == key) & (df.range == "long")]
        if d.empty:
            continue
        s = d[d.scored]
        k, npred, ntrue, npairs = s.n_correct.sum(), s.n_pred.sum(), s.n_true.sum(), s.n_pairs.sum()
        probs = [p for name in s.protein for p in sep_probs[key, name, "long"]]
        row = {"group": grp, "proteins": n_prot, "source": key, "asked": len(d), "scored": len(s),
               "with_long_range_pred": int((s.n_pred > 0).sum()), "long_pred": int(npred), "long_correct": int(k),
               "long_precision": k / npred if npred else np.nan, "baseline": ntrue / npairs if npairs else np.nan,
               "expected_random": float((s.n_pred * s.n_true / s.n_pairs).sum()), "expected_sep_matched": float(s.expected_sep.sum()),
               "p_random": sum_pvalue(list(zip(s.n_pairs, s.n_true, s.n_pred)), int(k)) if npred else np.nan,
               "p_sep_matched": poisson_binomial_pvalue(probs, int(k)) if npred else np.nan,
               "mean_protein_precision": s[s.n_pred > 0].precision.mean()}
        for ekey in ESM:
            e = esm[(esm.source == ekey) & esm.protein.isin(s.protein)] if len(esm) else esm
            if e.empty or f"hits@N {key}" not in e:
                continue
            m = e[f"hits@N {key}"].sum()
            row |= {f"{ekey}_matched_precision": m / npred if npred else np.nan, f"{ekey}_matched_correct": int(m)}
        group_rows.append(row)
    for ekey in ESM:
        e = esm[(esm.source == ekey) & esm.group.isin(members)] if len(esm) else esm
        if len(e):
            group_rows.append({"group": grp, "proteins": n_prot, "source": ekey, "asked": len(e), "scored": len(e),
                               "baseline": e.n_true.sum() / e.n_pairs.sum(), "P@L/5": e["P@L/5"].mean(), "P@L": e["P@L"].mean()})
groups = pd.DataFrame(group_rows)
groups.to_csv("results/contact_map_groups.tsv", sep="\t", index=False, float_format="%.4g")

# Shift test: the LLMs' predicted long-range pairs moved along the diagonal, pooled per group.
shift_rows = []
for key in [*SOURCES, "hydrophobic"]:
    for grp, members in POOLS.items():
        names = [p for p in TRUTH if group_of(p) in members and maps.get((key, p)) is not None]
        if not names:
            continue
        for k in SHIFTS:
            hits, tot = map(sum, zip(*(shifted(maps[key, p], true_maps[p], k) for p in names)))
            shift_rows.append({"source": key, "group": grp, "proteins": len(names), "shift": k, "pairs": tot, "hits": hits,
                               "precision": hits / tot if tot else np.nan})
shift = pd.DataFrame(shift_rows)
shift.to_csv("results/contact_map_shift.tsv", sep="\t", index=False, float_format="%.4g")

# Stability: Fable's grid prompt asked twice on the provisional clean group (before the
# organism-wide UniParc check moved some of it to "old sequence").
stab_rows = []
for p in TRUTH:
    if ("fable_grid_run2", p) not in notes:
        continue
    a, b = maps.get(("fable_grid", p)), maps.get(("fable_grid_run2", p))
    row = {"protein": p, "pdb": TRUTH[p]["pdb"], "group": group_of(p), "run1_scored": a is not None, "run2_scored": b is not None}
    for run, m in (("run1", a), ("run2", b)):
        if m is not None:
            s = score(m, true_maps[p], 24, 10**6)
            row |= {f"{run}_long_pred": s["n_pred"], f"{run}_long_correct": s["n_correct"], f"{run}_long_precision": s["precision"]}
    if a is not None and b is not None:
        for rng, (lo, hi) in (("long", RANGES["long"]), ("all", RANGES["all"])):
            i, j = band(len(a), lo, hi)
            x, y = a[i, j].astype(bool), b[i, j].astype(bool)
            row[f"jaccard_{rng}"] = (x & y).sum() / (x | y).sum() if (x | y).sum() else np.nan
    stab_rows.append(row)
stab = pd.DataFrame(stab_rows)
stab.to_csv("results/contact_map_stability.tsv", sep="\t", index=False, float_format="%.4g")

# --- print ---------------------------------------------------------------------------------------
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 500)
print(groups.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
if len(shift):
    print(shift.pivot_table(index=["source", "group"], columns="shift", values="precision").to_string(float_format=lambda x: f"{x:.2f}"))
print(stab.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
cols = ["group", "source", "pdb", "length", "scored", "n_pred", "n_true", "n_correct", "precision", "baseline", "expected_sep", "p_hyper", "note"]
print(df[(df.range == "long") & df.group.isin(["clean", "old sequence"])][cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
if len(esm):
    print(esm[esm.group.isin(["clean", "old sequence"])][["source", "pdb", "length", "P@L/5", "P@L", "P@N fable_grid", "P@N fable_list"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
for (key, name), note in notes.items():
    if note and maps.get((key, name)) is None:
        print(f"not scored: {key} {TRUTH[name]['pdb']}: {note}")

# --- figures ---------------------------------------------------------------------------------------
Path("figures").mkdir(exist_ok=True)


def frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(LIGHT_GREY)
        sp.set_linewidth(0.5)


# The pilot figure, unchanged: famous proteins on the left, the six recent pilot proteins on the right.
PILOT = {"opus_grid": "Claude Opus 5.5", "fable_grid": "Claude Fable 5.2", "gpt_grid": "GPT-6.2 Sol"}
SETS = {"famous": "Famous proteins: the model may recall the structure", "recent": "Recent proteins: released after 2026-08-01"}
cols_ = [None, *PILOT]
fig = plt.figure(figsize=(9.6, 7.0))
subs = fig.subfigures(1, 2, wspace=0.03)
for sub, (grp, title) in zip(subs, SETS.items()):
    names = [k for k, v in TRUTH.items() if v["set"] == grp]
    axes = sub.subplots(len(names), len(cols_))
    sub.subplots_adjust(left=0.13, right=0.99, top=0.925, bottom=0.03, wspace=0.08, hspace=0.3)
    sub.suptitle(title, x=0.02, y=0.995, ha="left", va="top", fontsize=8, fontweight="bold", color=INK)
    for r, name in enumerate(names):
        true = true_maps[name]
        n = len(true)
        for c, key in enumerate(cols_):
            ax = axes[r, c]
            ax.imshow(true, cmap=ListedColormap(["white", MUTED if key is None else LIGHT_GREY]), interpolation="nearest",
                      extent=(-0.5, n - 0.5, n - 0.5, -0.5))
            if key is None:
                ax.set_xlabel(f"L = {n}", fontsize=5.6, color=MUTED, labelpad=2)
            elif drawn.get((key, name)) is not None:
                pred = drawn[key, name]
                h, w = pred.shape
                ax.imshow(np.ma.masked_where(pred == 0, pred), cmap=ListedColormap([color(PILOT[key])]), interpolation="nearest",
                          extent=(-0.5, w - 0.5, h - 0.5, -0.5))
                ax.set_xlim(-0.5, max(n, w) - 0.5)
                ax.set_ylim(max(n, h) - 0.5, -0.5)
                if maps[key, name] is None:
                    wide = notes[key, name].split("lengths ")[1].strip("[]").replace(", ", "/")
                    ax.set_xlabel(f"malformed: {h} rows, {wide} wide", fontsize=5.6, color=MUTED, labelpad=2)
                else:
                    lr = df[(df.source == key) & (df.protein == name) & (df.range == "long")].iloc[0]
                    ax.set_xlabel(f"long P {lr.precision:.2f} / {lr.baseline:.2f}" if lr.n_pred else "no long-range 1s",
                                  fontsize=5.6, color=MUTED, labelpad=2)
            else:
                note = notes[key, name]
                label = "declined" if note.startswith("declined") else "error" if note.startswith("error") else "malformed"
                ax.text(0.5, 0.5, label, transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=INK,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
            frame(ax)
            if r == 0:
                ax.set_title("True" if key is None else PILOT[key], fontsize=6.5, fontweight="bold",
                             color=INK if key is None else color(PILOT[key]), pad=3)
            if c == 0:
                label = name.replace(", ", "\n") if len(name) > 22 else name
                ax.set_ylabel(f"{label}\n{TRUTH[name]['pdb']}", fontsize=5.8, color=INK, labelpad=3)
fig.text(0.01, -0.005, "Grey: true contacts (C-beta within 8 Å). Colour: the model's predicted contacts. Under each map, long P: "
         "precision of the predicted contacts at |i-j| >= 24 / the precision of random guesses (true contact density at that range).\n"
         "Malformed replies (rows of the wrong length or number) are drawn as written, rows left-aligned, and not scored.",
         fontsize=5.8, color=MUTED, ha="left", va="top", linespacing=1.4)
fig.savefig("figures/contact_map_pilot.png", bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# (a) Fable vs ESM-2 650M at matched N, per protein, one panel per prompt.
if len(esm):
    e650 = esm[esm.source == "esm2_650M"].set_index("protein")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, key in zip(axes, ["fable_grid", "fable_list"]):
        d = df[(df.source == key) & (df.range == "long") & df.scored & (df.n_pred > 0)].set_index("protein")
        ax.plot([0, 1], [0, 1], color=LIGHT_GREY, lw=0.8, zorder=0)
        for grp in GROUPS:
            g = d[d.group == grp]
            if g.empty:
                continue
            x = e650.loc[g.index, f"P@N {key}"]
            ax.scatter(x, g.precision, s=10 + 4 * np.sqrt(g.n_pred), color=GROUP_COLOR[grp], edgecolor="white", linewidth=0.4,
                       label=f"{grp} ({len(g)})", zorder=3 if grp == "clean" else 2, alpha=0.95)
            if grp == "clean":
                for name, row in g.iterrows():
                    ax.annotate(row.pdb, (x[name], row.precision), xytext=(3, 2), textcoords="offset points", fontsize=5.4, color=INK)
        skipped = len(df[(df.source == key) & (df.range == "long")]) - len(d)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.grid(True, axis="both", color="#ECEEF2", lw=0.5)
        ax.spines["left"].set_visible(True)
        ax.tick_params(axis="y", length=2.5, color=MUTED)
        ax.set_xlabel("ESM-2 650M: precision of its top N long-range pairs", fontsize=6.6)
        ax.set_ylabel(f"Fable 5.2 ({SOURCES[key][1]} prompt): precision of its N", fontsize=6.6)
        total = len(df[(df.source == key) & (df.range == "long")])
        ax.set_title(f"{SOURCES[key][1].capitalize()} prompt: {total - skipped} of {total} proteins", fontsize=7.5, fontweight="bold",
                     loc="left", color=INK)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=5.8, title="Protein group", title_fontsize=6.2, handletextpad=0.3)
    fig.suptitle("Long-range contacts (|i-j| ≥ 24): Claude Fable 5.2 vs ESM-2 650M, at Fable's own number of predictions N",
                 x=0.01, ha="left", fontsize=8, fontweight="bold")
    fig.text(0.01, -0.02, "Each point is a protein. N = the long-range contacts Fable predicted for it; ESM-2 is scored on its N highest-probability "
             "long-range pairs. Point size grows with N.\nProteins whose reply was not scored or had no long-range contact are left out. clean: no earlier PDB relative (e < 1e-3) and sequence first public on or after "
             "2026-07-01.\nfamous: positive control (structures long public). Grey line: equal precision.",
             fontsize=5.6, color=MUTED, ha="left", va="top", linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig("figures/contact_map_fable_vs_esm.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

# (b) True vs Fable vs ESM-2 maps for proteins with no earlier PDB relative: every clean one, then
# proteins from the "old sequence" group spanning Fable's grid precision (best to worst), six in all.
clean = [p for p in TRUTH if group_of(p) == "clean"]
if clean and len(esm):
    lr = df[(df.source == "fable_grid") & (df.range == "long")].set_index("protein")

    def ranked(ps):
        return sorted(ps, key=lambda p: -np.nan_to_num(lr.precision.get(p, np.nan), nan=-1))

    old = ranked([p for p in TRUTH if group_of(p) == "old sequence"])
    k_old = max(0, 6 - len(clean))
    picks = ranked(clean)[:6] + [old[k] for k in np.linspace(0, len(old) - 1, k_old).round().astype(int)]
    cols_b = ["true", "fable_grid", "fable_list", "esm2_650M"]
    heads = {"true": "True", "fable_grid": "Fable 5.2, grid", "fable_list": "Fable 5.2, list", "esm2_650M": "ESM-2 650M, top N"}
    fig, axes = plt.subplots(len(picks), len(cols_b), figsize=(5.6, 1.45 * len(picks) + 0.5), squeeze=False)
    for r, name in enumerate(picks):
        true = true_maps[name]
        n = len(true)
        for c, key in enumerate(cols_b):
            ax = axes[r, c]
            ax.imshow(true, cmap=ListedColormap(["white", MUTED if key == "true" else LIGHT_GREY]), interpolation="nearest")
            if key == "true":
                ax.set_xlabel(f"L = {n}", fontsize=5.6, color=MUTED, labelpad=2)
            elif key.startswith("esm"):
                pred_f = maps.get(("fable_grid", name))
                k = int(np.triu(pred_f, 6).sum()) if pred_f is not None else n
                i, j = band(n, 6, 10**6)
                top = np.argsort(-maps[key, name][i, j], kind="stable")[:k]
                pred = np.zeros_like(true)
                pred[i[top], j[top]] = 1
                ax.imshow(np.ma.masked_where(pred == 0, pred), cmap=ListedColormap([PROTEIN_LM]), interpolation="nearest")
                e = esm[(esm.source == key) & (esm.protein == name)].iloc[0]
                ax.set_xlabel(f"top {k} shown; long P@L/5 {e['P@L/5']:.2f}", fontsize=5.6, color=MUTED, labelpad=2)
            elif maps.get((key, name)) is not None:
                pred = maps[key, name]
                ax.imshow(np.ma.masked_where(pred == 0, pred), cmap=ListedColormap([color("Claude Fable 5.2")]), interpolation="nearest")
                s = df[(df.source == key) & (df.protein == name) & (df.range == "long")].iloc[0]
                ax.set_xlabel(f"long {int(s.n_correct)}/{int(s.n_pred)} = {s.precision:.2f}" if s.n_pred else "no long-range contacts",
                              fontsize=5.6, color=MUTED, labelpad=2)
            else:
                ax.text(0.5, 0.5, "not scored", transform=ax.transAxes, ha="center", va="center", fontsize=6.2, color=INK)
            frame(ax)
            if r == 0:
                ax.set_title(heads[key], fontsize=6.5, fontweight="bold", pad=3,
                             color=INK if key == "true" else PROTEIN_LM if key.startswith("esm") else color("Claude Fable 5.2"))
            if c == 0:
                row = PROT.loc[TRUTH[name]["pdb"]]
                desc = str(row.description)[:26]
                ax.set_ylabel(f"{TRUTH[name]['pdb']}, {group_of(name)}\n{desc} ({row.fold})\nbaseline {lr.baseline.get(name, np.nan):.2f}",
                              fontsize=5.4, color=GROUP_COLOR[group_of(name)] if group_of(name) == "clean" else INK, labelpad=3)
    fig.suptitle(f"No earlier PDB relative: the {len(clean)} clean proteins (sequence first public after 2026-07-01)\n"
                 f"and {k_old} whose sequence was public before (AlphaFold DB may hold a model)", x=0.01, ha="left",
                 fontsize=7.2, fontweight="bold")
    fig.text(0.01, 0.0, "Grey: true contacts (C-beta within 8 Å). Colour: predicted contacts (upper triangle). Under each map: long-range "
             "(|i-j| ≥ 24) hits / predicted.\nESM-2: its N highest-probability pairs at |i-j| ≥ 6 are drawn, N = the contacts Fable predicted there "
             "(grid prompt; L if that reply was not scored);\nunder it, ESM-2's precision on its top L/5 long-range pairs. Baseline: true long-range contact density (precision of random guesses). "
             "Rows within each group ordered by Fable's grid precision.", fontsize=5.4, color=MUTED, ha="left", va="top", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig("figures/contact_map_clean_examples.png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
print("wrote results/contact_map_{summary,groups,shift,stability}.tsv and figures/contact_map_*.png")
