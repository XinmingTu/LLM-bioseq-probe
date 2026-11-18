"""Copy every model reply into results/outputs/<run>/<model>.jsonl.gz for publication. Free.
Name runs on the command line to export only those (e.g. contact_map).

Redaction: a reply that contains an e-mail address keeps only its text before the first
blank line (the sequence the model wrote), followed by "[non-sequence text redacted]".
A few replies drifted from the sequence into text resembling a real person's e-mail
signature; nothing else in the records is personal.
"""

import gzip
import json
import re
import sys
from pathlib import Path

RUNS = {  # local working directory -> what it holds
    "probe": "protein pilot, chat prompt",
    "probe_guess": "protein pilot, best-guess prompt",
    "confirm": "protein confirmation set, chat prompt",
    "raw": "protein confirmation set, raw continuation",
    "echo": "protein confirmation subset, copy-then-continue prompt",
    "recall_length": "recall length on well-known proteins",
    "dna": "DNA pilot",
    "dna_guess": "DNA pilot, best-guess prompt",
    "dna_confirm": "DNA confirmation set",
    "plm_confirm": "protein language models on the confirmation set",
    "plm_pilot": "ProtGPT3-10B on the pilot set",
    "no_instruction": "protein subset, the sequence alone with no instruction",
    "fable_repeat": "Claude Fable 5.2, 400 main-set questions asked again",
    "raw_logprobs": "open-weight models, top 5 next-token log-probabilities on the main set",
    "d_set": "D set: proteins first public after the assumed training cutoff, chat prompt",
    "plm_d": "protein language models on the D set",
    "small_families": "protein small-family set (20 to 300 proteins per family), chat prompt",
    "plm_small": "ProtGPT3-10B on the small-family set",
    "contact_map": "contact maps from sequence, grid and list prompts (ESM-2 probabilities left out: rerun them from the public weights)",
}
SKIP = ("esm2_",)  # large and regenerable
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
OUT = Path("results/outputs")

redacted = 0
for run, what in RUNS.items():  # all runs, or only those named on the command line
    if sys.argv[1:] and run not in sys.argv[1:]:
        continue
    src = Path("results") / run
    if not src.exists():
        continue
    (OUT / run).mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in src.glob("*.jsonl") if not p.name.startswith(SKIP)):
        rows = []
        for line in path.open():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = d.get("text")
            if isinstance(text, str) and EMAIL.search(text):
                d["text"] = text.split("\n\n")[0] + "\n\n[non-sequence text redacted]"
                redacted += 1
            rows.append(json.dumps(d, ensure_ascii=False))
        with gzip.open(OUT / run / f"{path.stem}.jsonl.gz", "wt") as f:
            f.write("\n".join(rows) + "\n")
    print(f"{run:15s} {len([p for p in src.glob('*.jsonl') if not p.name.startswith(SKIP)]):3d} files  ({what})")
print(f"{redacted} replies redacted")
