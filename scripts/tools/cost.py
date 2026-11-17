"""Spend so far, and projected for the full run, from the recorded token usage. Free.

Prices are USD per 1M tokens (input, output). Anthropic's are from its pricing
table, OpenAI's from developers.openai.com/api/docs/pricing (2026-11-16); those
marked ASSUMED are guesses set high, so the projection errs on overspending.
"""

import json
from pathlib import Path

PRICE = {
    "claude-opus-5-5": (4, 20), "claude-fable-5-2": (8, 40), "claude-fable-5-1": (10, 50), "claude-fable-5": (10, 50), "claude-opus-5": (5, 25), "claude-opus-4-6": (5, 25), "claude-opus-4-8": (5, 25), "claude-opus-4-7": (5, 25), "claude-opus-4-5-20251101": (5, 25), "claude-sonnet-4-5-20250929": (3, 15),
    "gpt-4o-2024-11-20": (2.5, 10), "gpt-4.1-2025-04-14": (2, 8),
    "gpt-5.5": (5, 30), "gpt-6.1-sol": (2, 10), "gpt-6-astra": (10, 50),
    "gpt-6.2-sol": (2, 10), "gpt-6.2-astra": (10, 50),  # list prices not checked; assumed equal to 6.1 Sol and 6 Astra
    "gemini-2.5-pro": (1.25, 10), "gemini-3.5-flash": (1, 6), "gemini-3.8-flash": (1, 6), "gemini-4-carbon": (5, 30),  # flash/4 ASSUMED
}
PRICE.update({  # Fireworks serverless, fireworks.ai/pricing (2026-11-16); Qwen 3.8 2.4T priced as Qwen 3.8 Max
    "kimi-k3p1-pro": (2, 10), "glm-5p3": (1.4, 4.4), "qwen3p8-2p4t-a95b": (2, 6), "deepseek-v4p1-flash": (0.3, 1.2),
    "minimax-m3": (0.3, 1.2), "gpt-oss-120b": (0.15, 0.6), "nemotron-3-ultra-nvfp4": (0.6, 2.4),
})
FULL = 2220  # probes per model in the pilot run


def tokens(u: dict) -> tuple[int, int]:
    t_in = u.get("input_tokens") or u.get("prompt_tokens") or u.get("prompt_token_count") or 0
    t_out = u.get("output_tokens") or u.get("completion_tokens") or ((u.get("candidates_token_count") or 0) + (u.get("thoughts_token_count") or 0))
    return t_in, t_out


total_now = total_full = 0.0
for p in sorted(p for d in ("results/probe", "results/probe_guess", "results/confirm", "results/dna", "results/dna_guess", "results/dna_confirm", "results/echo", "results/raw", "results/small_families", "results/no_instruction", "results/fable_repeat", "results/recall_length") for p in Path(d).glob("*.jsonl")):
    recs = [json.loads(line) for line in p.open()]
    t = [tokens(r.get("usage") or {}) for r in recs]
    pi, po = PRICE[p.stem]
    spent = sum(a * pi + b * po for a, b in t) / 1e6
    per = spent / len(recs)
    total_now += spent
    total_full += per * FULL
    print(f"{p.parent.name + '/' + p.stem:40s} n={len(recs):5d} spent ${spent:7.3f}  per probe ${per:.5f}  full ${per * FULL:7.2f}  mean out tokens {sum(b for _, b in t) / len(t):.0f}")
print(f"spent ${total_now:.2f}; projected full run ${total_full:.2f}")
