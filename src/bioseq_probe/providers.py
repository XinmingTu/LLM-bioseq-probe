"""One call shape across Anthropic, OpenAI and Gemini: prompt in, text (+ logprobs) out.

Each model's settings come from MODELS. Sampling is as close to greedy as each
API allows: temperature 0 where accepted; reasoning at its lowest setting where
it cannot be turned off.
"""

import os
from dataclasses import dataclass, field

import anthropic
import openai
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# model id -> (provider, request settings): thinking off, or at its lowest where it cannot be turned off.
MODELS: dict[str, tuple[str, dict]] = {
    "gpt-4o-2024-11-20": ("openai_chat", {}),
    "gpt-4.1-2025-04-14": ("openai_chat", {}),
    "gpt-5.5": ("openai_responses", {"reasoning": {"effort": "none"}}),
    "gpt-6.1-sol": ("openai_responses", {"reasoning": {"effort": "low"}}),  # lowest it accepts
    "gpt-6-astra": ("openai_responses", {"reasoning": {"effort": "low"}}),  # lowest it accepts
    "gpt-6.2-sol": ("openai_responses", {"reasoning": {"effort": "low"}}),  # lowest it accepts
    "gpt-6.2-astra": ("openai_responses", {"reasoning": {"effort": "low"}}),  # lowest it accepts
    "claude-sonnet-4-5-20250929": ("anthropic", {}),
    "claude-opus-4-5-20251101": ("anthropic", {}),  # no thinking unless asked
    "claude-opus-4-6": ("anthropic", {"output_config": {"effort": "low"}}),
    "claude-opus-4-7": ("anthropic", {"output_config": {"effort": "low"}}),
    "claude-opus-4-8": ("anthropic", {"output_config": {"effort": "low"}}),  # no thinking unless asked
    "claude-opus-5": ("anthropic", {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}),
    "claude-opus-5-5": ("anthropic", {"output_config": {"effort": "low"}}),
    "gemini-2.5-pro": ("gemini", {"thinking_config": {"thinking_budget": 128}}),
    "gemini-3.5-flash": ("gemini", {"thinking_config": {"thinking_level": "minimal"}}),
    "gemini-3.8-flash": ("gemini", {"thinking_config": {"thinking_budget": 0}}),  # "minimal" unsupported; still thinks ~550 tokens
    "gemini-4-carbon": ("gemini", {"thinking_config": {"thinking_level": "low"}}),
    # Open-weight models served by Fireworks (OpenAI-compatible); settings from smoke tests.
    "deepseek-v4p1-flash": ("fireworks", {"reasoning_effort": "none"}),
    "qwen3p8-2p4t-a95b": ("fireworks", {"reasoning_effort": "none"}),
    "kimi-k3p1-pro": ("fireworks", {"reasoning_effort": "none"}),
    "glm-5p3": ("fireworks", {"reasoning_effort": "low"}),  # thinking-only; lowest
    "minimax-m3": ("fireworks", {"reasoning_effort": "none"}),
    "gpt-oss-120b": ("fireworks", {"reasoning_effort": "low"}),  # thinking-only; lowest
    "nemotron-3-ultra-nvfp4": ("fireworks", {"reasoning_effort": "none"}),
    # no direct route: the Anthropic API serves Fable only to workspaces with data retention enabled
    "claude-fable-5-2": ("openrouter", {"id": "anthropic/claude-fable-5.2", "reasoning": {"effort": "low"}}),
    "claude-fable-5-1": ("openrouter", {"id": "anthropic/claude-fable-5.1", "reasoning": {"effort": "low"}}),
    "claude-fable-5": ("openrouter", {"id": "anthropic/claude-fable-5", "reasoning": {"effort": "low"}}),
}


@dataclass
class Reply:
    text: str
    stop: str | None = None  # provider stop reason; "refusal" is kept apart from a miss
    top_logprobs: list[dict[str, float]] = field(default_factory=list)  # per output token
    usage: dict = field(default_factory=dict)


def call(model: str, prompt: str, max_tokens: int = 4000) -> Reply:
    provider, cfg = MODELS[model]
    return _CALLS[provider](model, prompt, max_tokens, cfg)


def _anthropic(model, prompt, max_tokens, cfg):
    # Streamed: the SDK refuses a non-streaming request whose max_tokens could take over 10 minutes.
    with anthropic.Anthropic().messages.stream(
        model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}], **cfg
    ) as s:
        r = s.get_final_message()
    text = "".join(b.text for b in r.content if b.type == "text")
    return Reply(text, r.stop_reason, usage=r.usage.model_dump())


def _openai_chat(model, prompt, max_tokens, cfg):
    r = openai.OpenAI().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=64,
        logprobs=True,
        top_logprobs=20,
        **cfg,
    )
    c = r.choices[0]
    tops = [{t.token: t.logprob for t in tok.top_logprobs} for tok in (c.logprobs.content or [])]
    return Reply(c.message.content or "", c.finish_reason, tops, r.usage.model_dump())


def responses_stop(r) -> str | None:
    """The Responses API's stop reason: its status, or why it is incomplete ("max_output_tokens"
    when the reply hit the token limit; see bioseq_probe.probe.cut_off)."""
    details = getattr(r, "incomplete_details", None)
    return getattr(details, "reason", None) or r.status if r.status == "incomplete" else r.status


def _openai_responses(model, prompt, max_tokens, cfg):
    r = openai.OpenAI().responses.create(model=model, input=prompt, max_output_tokens=max_tokens, **cfg)
    return Reply(r.output_text or "", responses_stop(r), usage=r.usage.model_dump() if r.usage else {})


def _fireworks(model, prompt, max_tokens, cfg):
    client = openai.OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"])
    r = client.chat.completions.create(
        model=f"accounts/fireworks/models/{model}",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        **cfg,
    )
    c = r.choices[0]
    return Reply(c.message.content or "", c.finish_reason, usage=r.usage.model_dump() if r.usage else {})


def _openrouter(model, prompt, max_tokens, cfg):
    cfg = dict(cfg)
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    r = client.chat.completions.create(
        model=cfg.pop("id"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        extra_body=cfg,
    )
    c = r.choices[0]
    stop = "refusal" if c.finish_reason == "content_filter" else c.finish_reason  # Anthropic's usage-policy block
    return Reply(c.message.content or "", stop, usage=r.usage.model_dump() if r.usage else {})


def _gemini(model, prompt, max_tokens, cfg):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(api_version="v1beta"))
    conf = types.GenerateContentConfig(
        temperature=0, max_output_tokens=max_tokens, thinking_config=types.ThinkingConfig(**cfg["thinking_config"])
    )
    r = client.models.generate_content(model=model, contents=prompt, config=conf)
    cand = r.candidates[0] if r.candidates else None
    stop = str(cand.finish_reason) if cand else "no_candidate"
    return Reply(r.text or "", stop, usage=r.usage_metadata.model_dump() if r.usage_metadata else {})


_CALLS = {
    "anthropic": _anthropic,
    "openai_chat": _openai_chat,
    "openai_responses": _openai_responses,
    "gemini": _gemini,
    "fireworks": _fireworks,
    "openrouter": _openrouter,
}
