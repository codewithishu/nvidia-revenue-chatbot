"""Thin provider abstraction. Keys come from env vars only."""
from __future__ import annotations

import os

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o"}


class LLM:
    def __init__(self, provider: str, model: str = "", max_tokens: int = 1200):
        self.provider = provider
        self.model = model or DEFAULT_MODELS.get(provider, "")
        self.max_tokens = max_tokens
        if provider == "anthropic":
            import anthropic

            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example).")
            self.client = anthropic.Anthropic()
        elif provider == "openai":
            from openai import OpenAI

            if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL"):
                raise RuntimeError("OPENAI_API_KEY is not set (see .env.example).")
            self.client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (use 'anthropic' or 'openai').")

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        mt = max_tokens or self.max_tokens
        if self.provider == "anthropic":
            r = self.client.messages.create(
                model=self.model,
                max_tokens=mt,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
        r = self.client.chat.completions.create(
            model=self.model,
            max_tokens=mt,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (r.choices[0].message.content or "").strip()
