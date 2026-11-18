"""Check the clean contact-map proteins against every UniParc sequence from the same organism ->
updates data/contact_map_proteins.tsv. Free.

scripts/data/08_contact_map_set.py dates a sequence through UniProtKB (the entry the PDB entry names,
and Peptide Search), which leaves out much of what is public elsewhere (GenBank/ENA, RefSeq,
patents) and misses a construct with point mutations. So for each protein in the clean group,
every UniParc record whose taxonomy falls under the protein's source organism (taxonomy_id
query, which covers the subtree) is downloaded with its "first seen" date, records sharing at
least three 5-mers with the modelled sequence are aligned to it locally, and:
- org_exact_first_public: the earliest first-seen date of a record identical over at least 85% of
  the modelled residues; org_near_first_public: the same at 95% identity or more;
- org_older_identity, org_older_upi: the closest record first seen before 2026-07-01 (identical
  residues over the modelled length).
The group is recomputed with these dates; the UniProtKB-only group is kept as "group_uniprot".
(NCBI BLAST against nr would also cover other organisms; its URL API kept every job queued for
over an hour on 2026-11-18, so it is not used.)

    uv run --with biopython --with gemmi --with pydssp python scripts/data/09_contact_map_organism.py
"""

import gzip
import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
CUTOFF = "2026-07-01"
CACHE = Path("data/contact_map/cache")


def uniparc_org(taxid: str) -> pd.DataFrame:
    path = CACHE / f"uniparc_tax{taxid}.tsv.gz"
    if not path.exists():
        url = "https://rest.uniprot.org/uniparc/stream?" + urllib.parse.urlencode(
            {"query": f"taxonomy_id:{taxid}", "fields": "upi,oldestCrossRefCreated,sequence", "format": "tsv", "compressed": "true"})
        path.write_bytes(urllib.request.urlopen(url, timeout=1800).read())
    return pd.read_csv(io.BytesIO(gzip.decompress(path.read_bytes())), sep="\t")


def check(seq: str, taxid: str) -> dict:
    from importlib import import_module

    align = import_module("08_contact_map_set").align
    db = uniparc_org(taxid)
    kmers = {seq[i : i + 5] for i in range(len(seq) - 4)}
    hits = []
    for r in db.itertuples():
        s = r.Sequence
        if sum(s[i : i + 5] in kmers for i in range(0, len(s) - 4)) < 3:
            continue
        ident, cover, *_ = align(seq, s)
        hits.append((ident * cover, ident, cover, r[2], r.Entry))
    exact = [h[3] for h in hits if h[1] == 1.0 and h[2] >= 0.85]
    near = [h[3] for h in hits if h[1] >= 0.95 and h[2] >= 0.85]
    older = max((h for h in hits if h[3] < CUTOFF), default=None)
    return {"org_records": len(db), "org_aligned": len(hits), "org_exact_first_public": min(exact, default=None),
            "org_near_first_public": min(near, default=None), "org_older_identity": round(older[0], 3) if older else 0.0,
            "org_older_upi": older[4] if older else ""}


if __name__ == "__main__":
    prot = pd.read_csv("data/contact_map_proteins.tsv", sep="\t", dtype={"taxid": str})
    if "group_uniprot" not in prot:
        prot["group_uniprot"] = prot.group
    todo = prot[prot.group_uniprot == "clean"]
    res = {r.pdb: check(r.sequence, r.taxid.split(",")[0]) for r in todo.itertuples()}
    for col in ("org_records", "org_aligned", "org_exact_first_public", "org_near_first_public", "org_older_identity", "org_older_upi"):
        prot[col] = [res.get(p, {}).get(col) for p in prot.pdb]

    def group(r):
        dates = [d for d in (r.first_public, r.near_identical_first_public, r.org_exact_first_public, r.org_near_first_public)
                 if isinstance(d, str) and d]
        new_seq = not dates or min(dates) >= CUTOFF
        if r.pdb_relatives == 0:
            return "clean" if new_seq else "old sequence"
        return "new sequence, PDB relative" if new_seq else "old sequence, PDB relative"

    prot["group"] = [group(r) for r in prot.itertuples()]
    prot.to_csv("data/contact_map_proteins.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(prot.loc[prot.pdb.isin(todo.pdb), ["pdb", "organism", "group_uniprot", "group", "org_records", "org_aligned", "org_exact_first_public",
                                             "org_near_first_public", "org_older_identity", "org_older_upi"]].to_string(index=False))
    print(prot.groupby(["role", "group"]).size().to_string())
