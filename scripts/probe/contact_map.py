"""Pilot: can a language model predict a protein's residue contact map from its sequence?
-> results/contact_map/<model>.jsonl. Dry by default; --run makes paid calls.

The prompt is Leo Tianlai Chen's: the sequence and its length, nothing naming the protein. True maps come
from PDB coordinates: residues i and j are in contact if their C-beta atoms (C-alpha for glycine)
are within 8 Angstrom. The sequence given is the modelled residues of the chain, so every residue
has coordinates. Scoring (scripts/analysis/contact_map.py) counts pairs at least 6 apart.

Two sets of six: famous proteins whose structures a model may simply recall, and recent ones, X-ray
structures first released after 2026-08-01 (after the models' assumed training cutoff), 60-130
modelled residues in one contiguous chain, with no earlier PDB entry at 90% sequence identity or
more (RCSB sequence search, 2026-11-18). A rerun keeps the replies already on file and asks only for
the missing ones and those that errored.

The expanded set (scripts/data/08_contact_map_set.py, data/contact_map_proteins.tsv) adds about
40 recent X-ray structures with each protein's novelty recorded; their true maps come from the
mmCIF file, on the gap-free modelled chain listed there. A second prompt, "list", asks for the
contacts as "i j" lines instead of a grid (Leo's grid prompt is kept unchanged as "grid").
Replies go to results/contact_map/<model>.jsonl for the grid prompt (the pilot's file, extended),
<model>__list.jsonl for the list prompt, and <model>__grid_run2.jsonl for a repeat of the grid prompt.

    uv run --with biopython --with gemmi python scripts/probe/contact_map.py --models claude-opus-5-5 --run
    uv run --with biopython --with gemmi python scripts/probe/contact_map.py --models claude-fable-5-2 --prompt list --run
    uv run --with biopython --with gemmi python scripts/probe/contact_map.py --models claude-fable-5-2 --proteins clean --tag run2 --run
"""

import argparse
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from bioseq_probe.providers import call

PROTEINS = {  # name: (PDB id, chain, set)
    "barrier-to-autointegration factor": ("1CI4", "A", "famous"),
    "ubiquitin": ("1UBQ", "A", "famous"),
    "protein G B1 domain": ("1PGA", "A", "famous"),
    "BPTI": ("5PTI", "A", "famous"),
    "spectrin SH3 domain": ("1SHG", "A", "famous"),
    "thioredoxin": ("2TRX", "A", "famous"),
    "HicA toxin, Leptotrichia wadei": ("9WYY", "A", "recent"),  # released 2026-10-07, 1.52 A
    "YbaB homolog, Streptococcus mutans": ("9YMS", "A", "recent"),  # 2026-10-21, 1.5 A
    "anti-CRISPR AcrIIA17": ("23AA", "A", "recent"),  # 2026-08-12, 2.03 A
    "lanmodulin, Ancylobacter novellus": ("39RI", "A", "recent"),  # 2026-11-11, 1.5 A
    "peptidyl-tRNA hydrolase 2, Lokiarchaeum": ("24BT", "A", "recent"),  # 2026-09-02, 2.12 A
    "VnOptE beta-barrel domain": ("37TF", "A", "recent"),  # 2026-11-04, 2.05 A
}
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
         "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def structure(pdb: str, chain: str) -> tuple[str, np.ndarray]:
    """(sequence of the modelled standard residues, contact map) for one chain."""
    from Bio.PDB import PDBParser

    text = urllib.request.urlopen(f"https://files.rcsb.org/download/{pdb}.pdb", timeout=60).read().decode()
    model = next(iter(PDBParser(QUIET=True).get_structure(pdb, io.StringIO(text))))
    seq, xyz = [], []
    for res in model[chain]:
        name = res.get_resname()
        atom = "CA" if name == "GLY" else "CB"
        if res.id[0] != " " or name not in THREE or atom not in res:
            continue
        seq.append(THREE[name])
        xyz.append(res[atom].coord)
    xyz = np.array(xyz)
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    return "".join(seq), (dist < 8.0).astype(int)


def structure_cif(pdb: str, label_asym: str, first: int, last: int) -> tuple[str, np.ndarray]:
    """The same for the expanded set: label_seq_id first..last of one chain (label_asym_id),
    checked gap-free by scripts/data/08_contact_map_set.py."""
    import gemmi

    path = Path(f"data/contact_map/cif/{pdb}.cif.gz")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(f"https://files.rcsb.org/download/{pdb}.cif.gz", timeout=60).read())
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_alternative_conformations()
    res = [r for ch in st[0] for r in ch.get_polymer() if r.subchain == label_asym and first <= r.label_seq <= last]
    assert [r.label_seq for r in res] == list(range(first, last + 1)), pdb
    xyz = np.array([r.find_atom("CA" if r.name == "GLY" else "CB", "*").pos.tolist() for r in res])
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    return "".join(THREE[r.name] for r in res), (dist < 8.0).astype(int)


def prompt(seq: str) -> str:
    lines = "\n".join(seq[i : i + 60] for i in range(0, len(seq), 60))
    n = len(seq)
    return (f"Below is a protein sequence of {n} amino acids in one-letter codes. Predict its residue-residue contact map: "
            f"residues i and j are in contact if their C-beta atoms (C-alpha for glycine) are within 8 Å. Give your best guess "
            f"even if you are uncertain. Do not explain or refuse. Reply with the {n}x{n} contact map only: {n} lines, each a "
            f"string of {n} characters, 0 or 1, with no spaces, where character j of line i is 1 if residues i and j are in "
            f"contact. Nothing else.\n\n>sequence\n{lines}")


def prompt_list(seq: str) -> str:
    """The same question, answered as a list of contacting pairs: no grid to keep aligned."""
    lines = "\n".join(seq[i : i + 60] for i in range(0, len(seq), 60))
    n = len(seq)
    return (f"Below is a protein sequence of {n} amino acids in one-letter codes. Predict its residue-residue contacts: "
            f"residues i and j are in contact if their C-beta atoms (C-alpha for glycine) are within 8 Å. Give your best guess "
            f"even if you are uncertain. Do not explain or refuse. Reply with the contacts only, one pair per line, written as "
            f"\"i j\": the two residue numbers (1 to {n}, numbered from the first residue) separated by a space, with i < j and "
            f"j - i >= 6. Nothing else.\n\n>sequence\n{lines}")


PROMPTS = {"grid": prompt, "list": prompt_list}


def load_truth() -> dict:
    """The pilot's twelve proteins (PDB format, author chain) and the expanded set (mmCIF, label chain)."""
    import pandas as pd

    truth = {}
    for name, (pdb, chain, group) in PROTEINS.items():
        seq, cmap = structure(pdb, chain)
        truth[name] = {"pdb": pdb, "chain": chain, "set": group, "sequence": seq, "map": ["".join(map(str, r)) for r in cmap]}
    for r in pd.read_csv("data/contact_map_proteins.tsv", sep="\t").itertuples():
        if r.pdb in {v["pdb"] for v in truth.values()}:
            continue  # a pilot protein, re-checked there but scored on its pilot map
        seq, cmap = structure_cif(r.pdb, r.label_asym, r.seq_from, r.seq_to)
        assert seq == r.sequence, r.pdb
        truth[r.pdb] = {"pdb": r.pdb, "chain": r.chain, "set": "expanded", "sequence": seq, "map": ["".join(map(str, x)) for x in cmap]}
    return truth


if __name__ == "__main__":
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--prompt", choices=PROMPTS, default="grid")
    ap.add_argument("--proteins", choices=["all", "clean"], default="all", help="clean: the cleanest group only")
    ap.add_argument("--tag", default="", help="a repeat run's name, e.g. run2")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    out = Path("results/contact_map")
    out.mkdir(parents=True, exist_ok=True)
    truth = load_truth()
    (out / "truth.json").write_text(json.dumps(truth))
    asked = dict(truth)
    if args.proteins == "clean":
        clean = set(pd.read_csv("data/contact_map_proteins.tsv", sep="\t").query("group == 'clean'").pdb)
        asked = {k: v for k, v in truth.items() if v["pdb"] in clean}
    print(f"{len(truth)} proteins with true maps; asking about {len(asked)}")
    for model in args.models.split(","):
        suffix = f"__{args.prompt}_{args.tag}" if args.tag else "" if args.prompt == "grid" else f"__{args.prompt}"
        path = out / f"{model}{suffix}.jsonl"
        kept = {r["protein"]: r for r in map(json.loads, path.open()) if "error" not in r} if path.exists() else {}
        todo = [n for n in asked if n not in kept]
        cost = sum(len(asked[n]["sequence"]) ** 2 for n in todo) / 2.5 * 40 / 1e6  # ~2.5 characters a token, $40/M output at most
        print(f"{model} {args.prompt}{' ' + args.tag if args.tag else ''}: kept {len(kept)}, to ask {len(todo)} (at most ~${cost:.0f})")
        if not args.run:
            continue

        def one(name, model=model):
            t = truth[name]
            rec = {"model": model, "prompt": args.prompt, "protein": name, "pdb": t["pdb"], "length": len(t["sequence"])}
            try:
                r = call(model, PROMPTS[args.prompt](t["sequence"]), max_tokens=32000)
                rec |= {"text": r.text, "stop": r.stop, "usage": r.usage}
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            return rec

        with ThreadPoolExecutor(8) as pool:
            new = dict(zip(todo, pool.map(one, todo)))
        with path.open("w") as f:  # replies about proteins no longer in the set are kept, not scored
            for name in [*truth, *(k for k in kept if k not in truth)]:
                if name in kept or name in new:
                    f.write(json.dumps(kept.get(name) or new[name]) + "\n")
        print(model, f"kept {len(kept)}, asked {len(new)}, errors {sum('error' in r for r in new.values())}")
