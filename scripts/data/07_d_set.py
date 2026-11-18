"""The D set: 56 proteins first public after the assumed training cutoff, with no earlier relative at
50% identity or more -> data/d_set_sequences.tsv, in the format of data/confirm_sequences.tsv. Free.

The proteins and their selection are Leo Tianlai Chen's (data/d_set/D56_proteins.tsv; method in
"D set: how the 56 proteins were selected", built 17 November 2026 from UniParc and UniRef50
2026_04). Each was first made public in RefSeq between 2 and 27 August 2026; its closest relative
public before August is 34% to 49% identical.

Cuts follow that document: five distinct positions drawn uniformly from
[max(domain_start - 1 + 10, 30), domain_end - 10), where the domain is the 1-based envelope of the
family's best domain, seeded by set, family and protein. Family sizes come from the main set,
which holds all 31 families.

    uv run python scripts/data/07_d_set.py
"""

import random
import zlib

import pandas as pd

from bioseq_probe.probe import STANDARD

d = pd.read_csv("data/d_set/D56_proteins.tsv", sep="\t")
main = pd.read_csv("data/confirm_sequences.tsv", sep="\t").groupby("pfam")[["family_proteins", "size_bin"]].first()
rows = []
for r in d.itertuples():
    assert len(r.sequence) == r.length and set(r.sequence) <= set(STANDARD), r.upi
    start, end = map(int, r.domain.split("-"))
    lo, hi = max(start - 1 + 10, 30), end - 10
    rng = random.Random(zlib.crc32(f"D|{r.pfam}|{r.upi}".encode()))
    cuts = sorted(rng.sample(range(lo, hi), 5))
    rows.append({"pfam": r.pfam, "family_proteins": main.family_proteins[r.pfam], "size_bin": main.size_bin[r.pfam],
                 "accession": r.refseq_accession, "upi": r.upi, "organism": r.organism, "phylum": r.phylum,
                 "refseq_created": r.refseq_created, "best_identity": r.best_identity_any_coverage50, "length": r.length,
                 "domain": r.domain, "cuts": ",".join(map(str, cuts)), "sequence": r.sequence})
out = pd.DataFrame(rows)
out.to_csv("data/d_set_sequences.tsv", sep="\t", index=False)
print(len(out), "proteins,", out.pfam.nunique(), "families,", 5 * len(out), "cuts")
print(out.size_bin.value_counts().to_string())
