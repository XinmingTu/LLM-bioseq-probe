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

    uv run --with biopython python scripts/probe/contact_map.py --models claude-opus-5-5 --run
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


def prompt(seq: str) -> str:
    lines = "\n".join(seq[i : i + 60] for i in range(0, len(seq), 60))
    n = len(seq)
    return (f"Below is a protein sequence of {n} amino acids in one-letter codes. Predict its residue-residue contact map: "
            f"residues i and j are in contact if their C-beta atoms (C-alpha for glycine) are within 8 Å. Give your best guess "
            f"even if you are uncertain. Do not explain or refuse. Reply with the {n}x{n} contact map only: {n} lines, each a "
            f"string of {n} characters, 0 or 1, with no spaces, where character j of line i is 1 if residues i and j are in "
            f"contact. Nothing else.\n\n>sequence\n{lines}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    out = Path("results/contact_map")
    out.mkdir(parents=True, exist_ok=True)
    truth = {}
    for name, (pdb, chain, group) in PROTEINS.items():
        seq, cmap = structure(pdb, chain)
        truth[name] = {"pdb": pdb, "chain": chain, "set": group, "sequence": seq, "map": ["".join(map(str, r)) for r in cmap]}
        print(f"{name:36s} {pdb} {len(seq):4d} residues, {int(np.triu(cmap, 6).sum())} contacts at separation >= 6")
    (out / "truth.json").write_text(json.dumps(truth))
    for model in args.models.split(","):
        if not args.run:
            continue

        def one(item, model=model):
            name, t = item
            rec = {"model": model, "protein": name, "pdb": t["pdb"], "length": len(t["sequence"])}
            try:
                r = call(model, prompt(t["sequence"]), max_tokens=32000)
                rec |= {"text": r.text, "stop": r.stop, "usage": r.usage}
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            return rec

        path = out / f"{model}.jsonl"
        kept = {r["protein"]: r for r in map(json.loads, path.open()) if "error" not in r} if path.exists() else {}
        with ThreadPoolExecutor(8) as pool:
            new = dict(zip([n for n in truth if n not in kept], pool.map(one, [(n, t) for n, t in truth.items() if n not in kept])))
        with path.open("w") as f:
            for name in truth:
                f.write(json.dumps(kept.get(name) or new[name]) + "\n")
        print(model, f"kept {len(kept)}, asked {len(new)}")
