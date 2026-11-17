"""OpenRouter access through the OpenAI SDK. Agents only call `structured_completion`."""
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Generic, TypeVar

import httpx
import openai
from openai import AsyncOpenAI
from openai.lib._parsing._completions import type_to_response_format_param
from pydantic import BaseModel, ValidationError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-5-mini")  # the interior designer's model
# The other agents run on a small model unless the user picks another one per agent.
HELPER_MODEL = os.environ.get("OPENROUTER_HELPER_MODEL", "openai/gpt-5-mini")
# Agents with their own model choice (graph node name -> label). The interior designer uses the main model.
AGENT_MODELS = {
    "interior_designer": "Interior Designer",
    "interior_critic": "Interior Critic",
    "furniture_research": "Furniture Research",
    "orchestrator": "Request Router",
}
MAX_ATTEMPTS = 2  # transient failures (timeouts, 429, 5xx) are retried once

# Suggestions for the frontend dropdown; all support structured outputs on OpenRouter.
AVAILABLE_MODELS = [
    {"id": "openai/gpt-5-mini", "label": "GPT-5 mini"},
    {"id": "openai/gpt-5.4-mini", "label": "GPT-5.4 mini"},
    {"id": "openai/gpt-5.5", "label": "GPT-5.5"},
    {"id": "google/gemini-3.5-flash", "label": "Gemini 3.5 Flash"},
    {"id": "anthropic/claude-sonnet-5.5", "label": "Claude Sonnet 5.5"},
]

T = TypeVar("T", bound=BaseModel)


@dataclass
class CallInfo:
    """What happened during one structured completion; safe to show to users (no credentials)."""
    model: str
    started_at: float = field(default_factory=time.time)
    duration_ms: float = 0
    attempts: list[dict] = field(default_factory=list)
    resolved_model: str | None = None
    provider: str | None = None
    generation_id: str | None = None
    usage: dict | None = None
    cost: float | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "resolvedModel": self.resolved_model,
            "provider": self.provider,
            "generationId": self.generation_id,
            "durationMs": round(self.duration_ms),
            "attempts": self.attempts,
            "retries": max(len(self.attempts) - 1, 0),
            "usage": self.usage,
            "cost": self.cost,
            "error": self.error,
        }


@dataclass
class StructuredResult(Generic[T]):
    parsed: T
    info: CallInfo


class LLMError(Exception):
    def __init__(self, message: str, status_code: int = 502, info: CallInfo | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.info = info


_input_modalities: dict[str, list[str]] = {}
_modalities_fetched_at = 0.0


async def supports_image_input(model: str) -> bool | None:
    """Whether OpenRouter lists image input for `model`; None if that cannot be determined."""
    global _modalities_fetched_at
    if time.time() - _modalities_fetched_at > 3600:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{OPENROUTER_BASE_URL}/models")
                response.raise_for_status()
                _input_modalities.clear()
                _input_modalities.update({m["id"]: m.get("architecture", {}).get("input_modalities", [])
                                          for m in response.json()["data"]})
                _modalities_fetched_at = time.time()
        except (httpx.HTTPError, KeyError, ValueError):
            return None
    if model not in _input_modalities:
        return None
    return "image" in _input_modalities[model]


@lru_cache
def get_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY is not set on the server.", status_code=500)
    return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=150, max_retries=0)


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    return isinstance(exc, openai.APIStatusError) and (exc.status_code == 429 or exc.status_code >= 500)


def _to_llm_error(exc: Exception, model: str) -> LLMError:
    if isinstance(exc, openai.LengthFinishReasonError):
        return LLMError("The model's answer was cut off before the layout was complete.")
    if isinstance(exc, openai.APITimeoutError):
        return LLMError("The model took too long to answer.", status_code=504)
    if isinstance(exc, openai.APIConnectionError):
        return LLMError("OpenRouter is not reachable.")
    if isinstance(exc, openai.APIStatusError):
        status = 400 if exc.status_code in (400, 404) else 502
        return LLMError(f"OpenRouter error ({exc.status_code}) for model '{model}': {exc.message}", status)
    if isinstance(exc, ValidationError):
        return LLMError("The model returned data that does not match the expected layout format.")
    raise exc


def _usage(completion) -> tuple[dict | None, float | None]:
    if not completion.usage:
        return None, None
    usage = completion.usage
    details = usage.completion_tokens_details
    extra = usage.model_extra or {}
    return {
        "promptTokens": usage.prompt_tokens,
        "completionTokens": usage.completion_tokens,
        "reasoningTokens": getattr(details, "reasoning_tokens", None),
        "totalTokens": usage.total_tokens,
    }, extra.get("cost")


# Only route to providers that support every parameter used (JSON schema, tools); report cost in `usage`.
OPENROUTER_EXTRA = {"provider": {"require_parameters": True}, "usage": {"include": True}}


async def _complete(info: CallInfo, request) -> object:
    """Runs `request()` (one OpenRouter call) with retries for transient errors, recording each attempt."""
    started = time.perf_counter()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempt_started = time.perf_counter()
        try:
            completion = await request()
            info.attempts.append({"attempt": len(info.attempts) + 1, "durationMs": round((time.perf_counter() - attempt_started) * 1000)})
            break
        except Exception as exc:
            info.attempts.append({
                "attempt": len(info.attempts) + 1,
                "durationMs": round((time.perf_counter() - attempt_started) * 1000),
                "error": f"{type(exc).__name__}: {exc}"[:300],
            })
            if attempt < MAX_ATTEMPTS and _is_transient(exc):
                await asyncio.sleep(1.5)
                continue
            error = _to_llm_error(exc, info.model)
            info.duration_ms = (time.perf_counter() - started) * 1000
            info.error = str(error)
            error.info = info
            raise error

    info.duration_ms = (time.perf_counter() - started) * 1000
    info.resolved_model = completion.model
    info.provider = (completion.model_extra or {}).get("provider")
    info.generation_id = completion.id
    info.usage, info.cost = _usage(completion)
    return completion


async def tool_completion(model: str, messages: list[dict], tools: list[dict],
                          tool_choice: str | dict = "auto") -> tuple[dict, CallInfo]:
    """One chat completion with function tools. Returns the assistant message as a plain dict
    (ready to append to `messages`) and the call info."""
    info = CallInfo(model=model)
    completion = await _complete(info, lambda: get_client().chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice=tool_choice, extra_body=OPENROUTER_EXTRA))
    message = completion.choices[0].message
    result = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        result["tool_calls"] = [
            {"id": call.id, "type": "function",
             "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"}}
            for call in message.tool_calls
        ]
    return result, info


def parse_structured(content: str, schema: type[T]) -> T:
    """The answer as `schema`. Some models append text or repeat the object after a complete JSON
    answer ("trailing characters"); the first complete JSON value is used then."""
    try:
        return schema.model_validate_json(content)
    except ValidationError as exc:
        try:
            value, _ = json.JSONDecoder().raw_decode(content.strip())
        except ValueError:
            raise exc from None
        return schema.model_validate(value)  # still raises if the first value does not match


async def structured_completion(model: str, messages: list[dict], schema: type[T]) -> StructuredResult[T]:
    """Runs one chat completion whose answer must match `schema` (strict JSON schema). An answer that
    cannot be read as `schema` is asked for once more before failing."""
    info = CallInfo(model=model)
    response_format = type_to_response_format_param(schema)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        completion = await _complete(info, lambda: get_client().chat.completions.create(
            model=model, messages=messages, response_format=response_format, extra_body=OPENROUTER_EXTRA))
        choice = completion.choices[0]
        message = choice.message
        if message.refusal:
            info.error = f"Refusal: {message.refusal}"
            raise LLMError(f"The model refused the request: {message.refusal}", status_code=422, info=info)
        if choice.finish_reason == "length":
            info.error = "Answer cut off (length)"
            raise LLMError("The model's answer was cut off before the layout was complete.", info=info)
        if not message.content:
            info.error = "No structured answer"
            raise LLMError("The model returned no structured answer.", info=info)
        try:
            return StructuredResult(parsed=parse_structured(message.content, schema), info=info)
        except ValidationError as exc:
            info.attempts[-1]["error"] = f"{type(exc).__name__}: {exc}"[:300]
            if attempt == MAX_ATTEMPTS:
                info.error = str(_to_llm_error(exc, model))
                raise LLMError(info.error, info=info) from exc
    raise AssertionError("unreachable")
