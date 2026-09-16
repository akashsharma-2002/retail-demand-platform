"""Language-model access with usage and cost accounting. Optional: the assistant works without it."""

import time
from dataclasses import dataclass, field

from retail_platform.observability import tracing
from retail_platform.observability.metrics import LLM_SECONDS, LLM_TOKENS

# USD per 1M tokens for gpt-4o-mini (input, cached input, output)
PRICES = {"gpt-4o-mini": (0.15, 0.075, 0.60)}


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    events: list[dict] = field(default_factory=list)


class OpenAIChat:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_output_tokens: int = 800, timeout_s: float = 20.0):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=timeout_s, max_retries=1)
        self.model = model
        self.max_output_tokens = max_output_tokens

    def complete(self, system: str, user: str, usage: Usage) -> str:
        with tracing.observe("llm.explain", as_type="generation", model=self.model, input=user) as gen:
            text = self._complete(system, user, usage)
            gen.update(output=text, usage_details={"input": usage.input_tokens, "output": usage.output_tokens})
        return text

    def _complete(self, system: str, user: str, usage: Usage) -> str:
        started = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            max_tokens=self.max_output_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        u = resp.usage
        if u is None:
            raise RuntimeError("Model response did not include token usage")
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        price_in, price_cached, price_out = PRICES.get(self.model, PRICES["gpt-4o-mini"])
        cost = ((u.prompt_tokens - cached) * price_in + cached * price_cached + u.completion_tokens * price_out) / 1e6
        usage.calls += 1
        usage.input_tokens += u.prompt_tokens
        usage.cached_tokens += cached
        usage.output_tokens += u.completion_tokens
        usage.cost_usd += cost
        usage.seconds += time.perf_counter() - started
        LLM_SECONDS.labels(self.model).observe(time.perf_counter() - started)
        LLM_TOKENS.labels(self.model, "input").inc(u.prompt_tokens)
        LLM_TOKENS.labels(self.model, "output").inc(u.completion_tokens)
        return resp.choices[0].message.content or ""


class OllamaChat:
    """Ollama chat API. Works with a local daemon (cloud models after `ollama signin`) or directly with
    https://ollama.com using OLLAMA_API_KEY. Ollama cloud is billed by plan, not per token, so cost is reported
    as 0 and tokens are still counted."""

    def __init__(
        self,
        model: str = "gemma4:31b-cloud",
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
        max_output_tokens: int = 800,
        timeout_s: float = 60.0,
    ):
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s)
        self.model = model
        self.max_output_tokens = max_output_tokens

    def complete(self, system: str, user: str, usage: Usage) -> str:
        before_in, before_out = usage.input_tokens, usage.output_tokens
        with tracing.observe("llm.explain", as_type="generation", model=self.model, input=user) as gen:
            text = self._complete(system, user, usage)
            gen.update(
                output=text,
                usage_details={"input": usage.input_tokens - before_in, "output": usage.output_tokens - before_out},
            )
        return text

    def _complete(self, system: str, user: str, usage: Usage) -> str:
        started = time.perf_counter()
        resp = self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": 0.1, "num_predict": self.max_output_tokens},
            },
        )
        resp.raise_for_status()
        body = resp.json()
        usage.calls += 1
        usage.input_tokens += int(body.get("prompt_eval_count", 0))
        usage.output_tokens += int(body.get("eval_count", 0))
        usage.seconds += time.perf_counter() - started
        LLM_SECONDS.labels(self.model).observe(time.perf_counter() - started)
        LLM_TOKENS.labels(self.model, "input").inc(int(body.get("prompt_eval_count", 0)))
        LLM_TOKENS.labels(self.model, "output").inc(int(body.get("eval_count", 0)))
        return str(body.get("message", {}).get("content", ""))
