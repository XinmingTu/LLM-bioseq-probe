"""The contact-map set: recent X-ray structures, screened for how new each protein really is
-> data/contact_map_candidates.tsv (every protein that passes the structure filters) and
data/contact_map_proteins.tsv (the chosen set, plus the six recent pilot proteins re-checked). Free.

Structure filters (RCSB search and data APIs, mmCIF files):
- initial release on or after 2026-08-01, X-ray, resolution 2.5 A or better;
- one polymer entity, a protein, with no non-standard monomer (no selenomethionine);
- a modelled chain that is gap-free (consecutive label_seq_id, every residue with C-alpha, and
  C-beta unless glycine; consecutive C-N distance < 2 A) and 60 to 150 standard residues long;
- not a de novo design or other synthetic construct (source organism given, not "synthetic"; no
  "de novo" in the title or keywords: 35SB, a designed homodimer, lists E. coli as its source).

Novelty, per protein, on the modelled sequence (what the models are shown):
(a) earlier PDB relative: RCSB sequence search (MMseqs2), e-value < 1e-3, restricted to entries
    released before the protein's own release; the best identity among the top 100 hits, and the
    earliest such entry.
(b) the date its sequence first became public: the UniProtKB entry the PDB entry names is aligned
    to the modelled sequence (tag residues at the ends fall outside the alignment); if the
    natural stretch is identical (or at least 95% identical, recorded apart) its UniParc "first
    seen" date is taken (the oldest cross-reference of that UniParc record, often older than the
    UniProtKB entry). Unless that entry is already older than the cutoff, UniProt Peptide Search
    also finds every other UniProtKB entry containing the natural stretch exactly (without a
    named entry to align to, the stretch searched is the core, 8 residues in from each end, as
    construct tags such as a TEV or PreScission scar would otherwise block an exact match). UniParc is
    also searched by CRC64 checksum for the modelled, natural and full deposited sequences. The earliest date found is the first-public date. If nothing is found
    the protein has no public sequence before its PDB entry. (scripts/data/09_contact_map_organism.py
    then checks the proteins that look new against every UniParc record of their organism,
    which holds sequences UniProtKB leaves out.)
(c) AlphaFold DB: whether any UniProtKB accession found in (b) has a model (AFDB API).

API responses are cached in data/contact_map/cache/ (not committed), so a rerun is quick.

    uv run --with gemmi --with pydssp python scripts/data/08_contact_map_set.py
"""

import gzip
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

CUTOFF = "2026-07-01"  # sequences first public on or after this cannot have had a public predicted structure in training
RELEASED_FROM = "2026-08-01"
CACHE = Path("data/contact_map/cache")
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
         "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
PILOT_RECENT = {"9WYY": "A", "9YMS": "A", "23AA": "A", "39RI": "A", "24BT": "A", "37TF": "A"}  # scripts/probe/contact_map.py


def fetch(url: str, data: dict | None = None, tries: int = 6, raw: bool = False):
    """GET (or POST JSON) with a disk cache keyed by the request; None on 404/204."""
    key = hashlib.sha1((url + json.dumps(data, sort_keys=True)).encode()).hexdigest()
    path = CACHE / f"{key}.gz"
    if path.exists():
        body = gzip.decompress(path.read_bytes())
    else:
        for k in range(tries):
            try:
                req = urllib.request.Request(url, json.dumps(data).encode() if data else None,
                                             {"Content-Type": "application/json"} if data else {})
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = r.read() if r.status != 204 else b""
                break
            except urllib.error.HTTPError as e:
                if e.code in (400, 404):
                    body = b""
                    break
                if k == tries - 1:
                    raise
            except Exception:
                if k == tries - 1:
                    raise
            time.sleep(2 * (k + 1))
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(body))
    if raw:
        return body
    return json.loads(body) if body else None


def search_entries() -> list[str]:
    nodes = [("rcsb_accession_info.initial_release_date", "greater_or_equal", RELEASED_FROM),
             ("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
             ("rcsb_entry_info.resolution_combined", "less_or_equal", 2.5),
             ("rcsb_entry_info.polymer_entity_count", "equals", 1),
             ("rcsb_entry_info.polymer_entity_count_protein", "equals", 1),
             ("entity_poly.rcsb_sample_sequence_length", "range", {"from": 60, "to": 400, "include_lower": True, "include_upper": True})]
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {"attribute": a, "operator": o, "value": v}} for a, o, v in nodes]},
        "return_type": "entry", "request_options": {"return_all_hits": True, "results_content_type": ["experimental"]}}
    return sorted(r["identifier"] for r in fetch("https://search.rcsb.org/rcsbsearch/v2/query", q)["result_set"])


GQL = """query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id rcsb_accession_info{initial_release_date}
 rcsb_entry_info{resolution_combined} struct{title} struct_keywords{pdbx_keywords}
 polymer_entities{rcsb_polymer_entity{pdbx_description} entity_poly{pdbx_seq_one_letter_code_can rcsb_non_std_monomers type}
 rcsb_entity_source_organism{scientific_name ncbi_taxonomy_id}
 rcsb_polymer_entity_container_identifiers{reference_sequence_identifiers{database_accession database_name}}}}}"""


def metadata(ids: list[str]) -> dict[str, dict]:
    out = {}
    for k in range(0, len(ids), 100):
        for e in fetch("https://data.rcsb.org/graphql", {"query": GQL, "variables": {"ids": ids[k : k + 100]}})["data"]["entries"]:
            ent = e["polymer_entities"][0]
            src = ent["rcsb_entity_source_organism"] or []
            refs = (ent["rcsb_polymer_entity_container_identifiers"] or {}).get("reference_sequence_identifiers") or []
            out[e["rcsb_id"]] = {
                "pdb": e["rcsb_id"], "released": e["rcsb_accession_info"]["initial_release_date"][:10],
                "resolution": min(e["rcsb_entry_info"]["resolution_combined"] or [9.9]), "title": e["struct"]["title"],
                "keywords": (e["struct_keywords"] or {}).get("pdbx_keywords"), "description": ent["rcsb_polymer_entity"]["pdbx_description"],
                "entity_sequence": ent["entity_poly"]["pdbx_seq_one_letter_code_can"].replace("\n", ""),
                "nonstandard": ",".join(ent["entity_poly"]["rcsb_non_std_monomers"] or []), "poly_type": ent["entity_poly"]["type"],
                "organism": "; ".join(s["scientific_name"] or "" for s in src), "taxid": ",".join(str(s["ncbi_taxonomy_id"]) for s in src),
                "pdb_uniprot": ",".join(sorted({r["database_accession"] for r in refs if r["database_name"] == "UniProt"}))}
    return out


def chain_of(pdb: str) -> dict | None:
    """The first chain whose modelled residues form one gap-free stretch of 60-150 standard
    residues with C-alpha and C-beta (C-alpha for glycine) -> sequence, label_seq range, contact
    map (C-beta within 8 A), secondary structure (DSSP via pydssp)."""
    import pydssp

    path = CACHE.parent / "cif" / f"{pdb}.cif.gz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(f"https://files.rcsb.org/download/{pdb}.cif.gz", timeout=120).read())
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    for chain in st[0]:
        res = [r for r in chain.get_polymer()]
        if not 60 <= len(res) <= 150:
            continue
        seqids = [r.label_seq for r in res]
        if any(s is None for s in seqids) or seqids != list(range(seqids[0], seqids[0] + len(res))):
            continue
        if any(r.name not in THREE for r in res):
            continue
        atoms = []
        ok = True
        for r in res:
            need = ["N", "CA", "C", "O", "CA" if r.name == "GLY" else "CB"]
            found = {a: r.find_atom(a, "*") for a in need}
            if any(v is None for v in found.values()):
                ok = False
                break
            atoms.append([found[a].pos.tolist() for a in need])
        if not ok:
            continue
        xyz = np.array(atoms)  # L x 5 x 3: N CA C O CB
        if (np.linalg.norm(xyz[1:, 0] - xyz[:-1, 2], axis=1) > 2.0).any():
            continue
        cb = xyz[:, 4]
        cmap = (np.linalg.norm(cb[:, None] - cb[None], axis=-1) < 8.0).astype(int)
        ss = "".join(pydssp.assign(xyz[:, :4], out_type="c3"))
        return {"chain": chain.name, "label_asym": res[0].subchain, "seq_from": seqids[0], "seq_to": seqids[-1],
                "sequence": "".join(THREE[r.name] for r in res), "map": cmap, "ss": ss}
    return None


def fold_class(ss: str) -> str:
    h, e = ss.count("H") / len(ss), ss.count("E") / len(ss)
    if h >= 0.4 and e < 0.1:
        return "all-alpha"
    if e >= 0.3 and h < 0.1:
        return "all-beta"
    if h >= 0.15 and e >= 0.15:
        return "alpha+beta"
    return "other"


def pdb_relatives(seq: str, released: str) -> dict:
    """Earlier PDB entries with a sequence e-value < 1e-3 (top 100 by score)."""
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "sequence", "parameters": {"evalue_cutoff": 1e-3, "identity_cutoff": 0, "sequence_type": "protein", "value": seq}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_accession_info.initial_release_date", "operator": "less", "value": released}}]},
        "return_type": "polymer_entity", "request_options": {"paginate": {"start": 0, "rows": 100}, "results_verbosity": "verbose"}}
    r = fetch("https://search.rcsb.org/rcsbsearch/v2/query", q)
    if not r:
        return {"pdb_relatives": 0, "pdb_best_identity": 0.0, "pdb_best_hit": "", "pdb_best_evalue": np.nan}
    best = (0.0, "", np.nan)
    for hit in r["result_set"]:
        for node in (n for sv in hit["services"] for n in sv["nodes"]):
            for m in node.get("match_context", []):
                if m["sequence_identity"] > best[0]:
                    best = (m["sequence_identity"], hit["identifier"], m["evalue"])
    return {"pdb_relatives": r["total_count"], "pdb_best_identity": round(best[0], 3), "pdb_best_hit": best[1], "pdb_best_evalue": best[2]}


def peptide_search(seq: str) -> list[str] | None:
    """UniProtKB accessions whose sequence contains seq exactly (UniProt Peptide Search), or None
    if the service did not answer (it often drops connections)."""
    key = CACHE / f"pep_{hashlib.sha1(seq.encode()).hexdigest()}.txt"
    if key.exists():
        return key.read_text().split()
    for k in range(4):
        try:
            req = urllib.request.Request("https://peptidesearch.uniprot.org/asyncrest",
                                         urllib.parse.urlencode({"peps": seq, "lEQi": "off", "spOnly": "off"}).encode())
            with urllib.request.urlopen(req, timeout=60) as r:
                job = r.headers["Location"].rsplit("/", 1)[1]
            for _ in range(60):
                time.sleep(3)
                try:
                    with urllib.request.urlopen(f"https://peptidesearch.uniprot.org/asyncrest/jobs/{job}", timeout=60) as r:
                        if r.status == 200:
                            accs = r.read().decode().split()
                            CACHE.mkdir(parents=True, exist_ok=True)
                            key.write_text(" ".join(accs))
                            return accs
                except urllib.error.HTTPError as e:
                    if e.code not in (303, 202, 404):
                        raise
        except Exception:
            time.sleep(5 * (k + 1))
    return None


def crc64(seq: str) -> str:
    """SWISS-PROT CRC64, as UniParc uses for checksums."""
    poly = 0xD800000000000000
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ poly if c & 1 else c >> 1
        table.append(c)
    crc = 0
    for ch in seq.encode():
        crc = table[(crc ^ ch) & 0xFF] ^ (crc >> 8)
    return f"{crc:016X}"


def uniparc_dates(query: str) -> list[dict]:
    url = "https://rest.uniprot.org/uniparc/search?" + urllib.parse.urlencode(
        {"query": query, "fields": "upi,oldestCrossRefCreated,length", "format": "json", "size": 500})
    r = fetch(url) or {"results": []}
    return [{"upi": x["uniParcId"], "first_seen": x.get("oldestCrossRefCreated")} for x in r["results"]]


def uniprot_entry(acc: str) -> dict:
    """A UniProtKB entry's creation date, UniParc id and sequence. A deleted entry (for example one
    dropped as "not part of a reference proteome") keeps its UniParc id; the sequence comes from there."""
    d = fetch(f"https://rest.uniprot.org/uniprotkb/{acc}.json") or {}
    upi = (d.get("extraAttributes") or {}).get("uniParcId")
    seq = (d.get("sequence") or {}).get("value", "")
    if upi and not seq:
        seq = ((fetch(f"https://rest.uniprot.org/uniparc/{upi}.json") or {}).get("sequence") or {}).get("value", "")
    return {"acc": acc, "created": (d.get("entryAudit") or {}).get("firstPublicDate"), "upi": upi, "sequence": seq}


def align(query: str, target: str) -> tuple[float, float, int, int]:
    """Best local alignment of query to target: (identity over the aligned query residues,
    fraction of the query aligned, aligned query start, end). Tag residues at the ends of a
    construct fall outside the alignment."""
    from Bio import Align

    al = Align.PairwiseAligner(mode="local", match_score=2, mismatch_score=-1, open_gap_score=-4, extend_gap_score=-1)
    a = al.align(query, target)[0]
    qs, qe = a.aligned[0][0][0], a.aligned[0][-1][1]
    pairs = [(q, t) for q, t in zip(a[0], a[1]) if q != "-"]
    return sum(q == t for q, t in pairs) / len(pairs), (qe - qs) / len(query), int(qs), int(qe)


def afdb(acc: str) -> str | None:
    """Earliest AlphaFold DB model date for this accession, or None."""
    r = fetch(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}")
    if not r:
        return None
    dates = [m.get("modelCreatedDate") for m in r if isinstance(m, dict)]
    return min((d for d in dates if d), default="unknown")[:10]


def novelty(seq: str, entity_seq: str, pdb_uniprot: str) -> dict:
    """(b) and (c) for one modelled sequence; see the module docstring. A construct's tag residues
    are not part of the natural sequence, so "exact" means: the stretch that aligns to the UniProt
    sequence the PDB entry names covers at least 85% of the modelled residues and is identical."""
    exact, near, best = set(), set(), (np.nan, np.nan)
    native = seq
    entries = {}
    for acc in filter(None, pdb_uniprot.split(",")):
        e = entries[acc] = uniprot_entry(acc)
        if not e["sequence"]:
            continue
        ident, cover, qs, qe = align(seq, e["sequence"])
        if np.isnan(best[0]) or (ident * cover > best[0] * best[1]):
            best = (ident, cover)
        if cover >= 0.85 and ident == 1.0:
            exact.add(acc)
            native = seq[qs:qe]
        elif cover >= 0.85 and ident >= 0.95:
            near.add(acc)
    # Other UniProtKB entries holding the same stretch matter only if the named one is not already old.
    named_old = any((uniprot_entry(a)["created"] or "9999") < CUTOFF for a in exact | near)
    if not exact:  # no named entry to trim tags against: search the core, 8 residues in from each end
        native = seq[8:-8]
    pep = [] if named_old else peptide_search(native)
    exact |= set(pep or [])
    found, found_near = [], []  # (date, source)
    for acc in sorted(exact | near):
        e = entries.get(acc) or uniprot_entry(acc)
        dates = found if acc in exact else found_near
        if e["created"]:
            dates.append((e["created"], f"UniProtKB {acc}"))
        for x in uniparc_dates(f"upi:{e['upi']}") if e["upi"] else []:
            if x["first_seen"]:
                dates.append((x["first_seen"], f"UniParc {x['upi']} ({acc})"))
    for what, s in {"modelled": seq, "native": native, "deposited": entity_seq}.items():
        for x in uniparc_dates(f"checksum:{crc64(s)}"):
            if x["first_seen"]:
                found.append((x["first_seen"], f"UniParc {x['upi']} ({what} sequence)"))
    accs = sorted(exact | near)
    with ThreadPoolExecutor(8) as pool:
        models = {a: d for a, d in zip(accs[:100], pool.map(afdb, accs[:100])) if d}
    first = min(found) if found else (None, "")
    return {"native": native, "peptide_search": "not needed" if named_old else "failed" if pep is None else len(pep), "uniprot_exact": len(exact),
            "uniprot_example": ",".join(sorted(exact)[:4]), "pdb_uniprot_identity": best[0], "pdb_uniprot_coverage": best[1],
            "first_public": first[0], "first_public_source": first[1],
            "near_identical_first_public": min(found_near)[0] if found_near else None,
            "afdb_models": len(models), "afdb_first": min(models.values()) if models else None, "afdb_example": next(iter(models), "")}


def screen(pdb: str, meta: dict, chain_hint: str | None = None) -> dict | None:
    c = chain_of(pdb)
    if c is None:
        return None
    row = meta | {k: c[k] for k in ("chain", "label_asym", "seq_from", "seq_to", "sequence", "ss")}
    row |= {"length": len(c["sequence"]), "helix": round(c["ss"].count("H") / len(c["ss"]), 2),
            "strand": round(c["ss"].count("E") / len(c["ss"]), 2), "fold": fold_class(c["ss"])}
    row |= pdb_relatives(c["sequence"], meta["released"])
    return row


def clusters(cand: pd.DataFrame) -> list[int]:
    """Greedy clusters: a protein joins the first earlier cluster whose representative it matches
    at 40% identity or more over half its length, or that names the same UniProt accession."""
    reps, out = [], []
    for r in cand.itertuples():
        for k, (seq, acc) in enumerate(reps):
            if (isinstance(r.pdb_uniprot, str) and r.pdb_uniprot and r.pdb_uniprot == acc) or (lambda x: x[0] >= 0.4 and x[1] >= 0.5)(align(r.sequence, seq)):
                out.append(k)
                break
        else:
            reps.append((r.sequence, r.pdb_uniprot))
            out.append(len(reps) - 1)
    return out


def group(r) -> str:
    """The four novelty groups. A sequence counts as public before the cutoff if it, or a version
    at least 95% identical over 85% of it, was in UniProtKB or UniParc before 2026-07-01."""
    dates = [d for d in (r.first_public, r.near_identical_first_public) if isinstance(d, str)]
    new_seq = not dates or min(dates) >= CUTOFF
    return ("clean" if new_seq else "old sequence") if r.pdb_relatives == 0 else ("new sequence, PDB relative" if new_seq else "old sequence, PDB relative")


if __name__ == "__main__":
    ids = search_entries()
    meta = metadata(ids)
    print(len(ids), "entries from the search")
    keep = {k: m for k, m in meta.items() if not m["nonstandard"] and m["poly_type"] == "polypeptide(L)"
            and m["organism"] and "synthetic" not in m["organism"].lower() and "artificial" not in m["organism"].lower()
            and "DE NOVO" not in f"{m['keywords']} {m['title']}".upper()}
    print(len(keep), "with no non-standard monomer, a natural source organism, and not a de novo design")
    with ThreadPoolExecutor(8) as pool:
        rows = [r for r in pool.map(lambda k: screen(k, keep[k]), sorted(keep)) if r]
    cand = pd.DataFrame(rows)
    print(len(cand), "with a gap-free modelled chain of 60-150 standard residues")
    with ThreadPoolExecutor(6) as pool:
        nov = list(pool.map(lambda r: novelty(r.sequence, r.entity_sequence, r.pdb_uniprot if isinstance(r.pdb_uniprot, str) else ""),
                            cand.itertuples()))
    cand = pd.concat([cand, pd.DataFrame(nov)], axis=1)
    cand["group"] = [group(r) for r in cand.itertuples()]
    cand["cluster"] = clusters(cand)
    cand.to_csv("data/contact_map_candidates.tsv", sep="\t", index=False)
    print(cand.group.value_counts().to_string())
    print("clusters:", cand.cluster.nunique())

    # The chosen set: the pilot's six recent proteins (re-checked), then one protein per cluster,
    # best resolution first: every cluster with no earlier PDB relative or with a new sequence, then the rest drawn at
    # random in turn from each fold class, up to 40 new proteins. Clusters whose best-resolved
    # member has a new sequence or no earlier relative come first.
    pilot = cand[cand.pdb.isin(PILOT_RECENT)].assign(role="pilot")
    pool_ = cand[~cand.cluster.isin(pilot.cluster)].sort_values("resolution").drop_duplicates("cluster")
    first = pool_[(pool_.pdb_relatives == 0) | (pool_.group == "new sequence, PDB relative")]
    rest = pool_[~pool_.pdb.isin(first.pdb)].sample(frac=1, random_state=2026)
    picks, by_fold = list(first.pdb), {f: list(d.pdb) for f, d in rest.groupby("fold")}
    while len(picks) < 40 and any(by_fold.values()):
        for f in sorted(by_fold):
            if by_fold[f] and len(picks) < 40:
                picks.append(by_fold[f].pop(0))
    chosen = pd.concat([pilot, cand[cand.pdb.isin(picks)].assign(role="expanded")], ignore_index=True)
    chosen.to_csv("data/contact_map_proteins.tsv", sep="\t", index=False)
    print(len(chosen), "chosen")
    print(chosen.groupby(["role", "group"]).size().to_string())
    print(chosen.groupby(["role", "fold"]).size().to_string())
