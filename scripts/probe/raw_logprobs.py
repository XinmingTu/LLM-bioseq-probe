"""The raw-continuation probe read as probabilities -> results/raw_logprobs/<model>.jsonl.
Dry by default; --run makes paid calls; resumable.

Same text as scripts/probe/protein_raw.py (">sequence\\n" + the FASTA-wrapped prefix, Fireworks
/completions, no chat template), but one output token with the top 5 log-probabilities, the most
Fireworks returns. At temperature 0 the most likely token is what the raw probe already scored;
the top 5 add a more sensitive reading: whether the true residue is among them, and its probability.
Analysis: scripts/analysis/raw_logprobs.py.

    uv run python scripts/probe/raw_logprobs.py --models gpt-oss-120b --run
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

from bioseq_probe.probe import fasta, shuffled

load_dotenv()
MODELS = ("deepseek-v4p1-flash", "qwen3p8-2p4t-a95b", "kimi-k3p1-pro", "glm-5p3", "minimax-m3", "gpt-oss-120b", "nemotron-3-ultra-nvfp4")
TOP = 5  # Fireworks' limit


def text_of(prefix: str) -> str:  # as in protein_raw.py
    t = ">sequence\n" + fasta(prefix)
    return t + "\n" if len(prefix) % 60 == 0 else t


def top_tokens(model: str, prefix: str) -> tuple[list, dict]:
    """[(token, logprob), ...] for the next token, most likely first; and usage."""
    for attempt in range(8):  # back off on rate limits (429)
        r = httpx.post("https://api.fireworks.ai/inference/v1/completions", timeout=120,
                       headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"},
                       json={"model": f"accounts/fireworks/models/{model}", "prompt": text_of(prefix),
                             "max_tokens": 1, "temperature": 0, "logprobs": TOP})
        if r.status_code != 429:
            break
        time.sleep(min(2**attempt, 60))
    r.raise_for_status()
    j = r.json()
    top = j["choices"][0]["logprobs"]["top_logprobs"][0]
    return sorted(top.items(), key=lambda kv: -kv[1]), j.get("usage", {})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--sequences", default="data/confirm_sequences.tsv")
    ap.add_argument("--out", default="results/raw_logprobs")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    seqs = pd.read_csv(args.sequences, sep="\t")
    todo = [{"accession": r.accession, "pfam": r.pfam, "cut": int(c), "condition": cond, "prefix": s[: int(c)], "target": s[int(c) : int(c) + 10]}
            for r in seqs.itertuples() for c in r.cuts.split(",")
            for cond, s in (("real", r.sequence), ("shuffled", shuffled(r.sequence, r.accession)))]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for model in args.models.split(","):
        path = out / f"{model}.jsonl"
        done = {(d["accession"], d["cut"], d["condition"]) for d in map(json.loads, path.open())} if path.exists() else set()
        left = [p for p in todo if (p["accession"], p["cut"], p["condition"]) not in done]
        print(model, len(left), "of", len(todo), "questions left")
        if not args.run:
            continue

        def one(p, model=model):
            rec = {"model": model, **{k: p[k] for k in ("accession", "pfam", "cut", "condition", "target")}}
            try:
                rec["top"], rec["usage"] = top_tokens(model, p["prefix"])
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
            return rec

        with path.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
            for rec in pool.map(one, left):
                f.write(json.dumps(rec) + "\n")
                f.flush()
