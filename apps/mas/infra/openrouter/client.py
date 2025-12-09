from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from ..env import load_env, get_openrouter_api_key, get_openai_base_url, get_openrouter_headers


Message = Dict[str, str]


class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class ChatResult(BaseModel):
    text: str
    usage: ChatUsage
    raw: Dict[str, Any]


@dataclass
class OpenRouterConfig:
    model: str
    temperature: float = 0.2
    top_p: float = 0.95
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    max_output_tokens: int = 512
    request_timeout_s: int = 60
    max_retries: int = 3


class OpenRouterClient:
    def __init__(self, config: OpenRouterConfig):
        load_env()
        api_key = get_openrouter_api_key()
        base_url = get_openai_base_url()
        headers = get_openrouter_headers()
        if not api_key:
            raise RuntimeError("OpenRouter API key missing. Set OPENROUTER_API_KEY.")
        self.client = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)
        self.config = config

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((APIError, RateLimitError, APITimeoutError)),
    )
    def complete_chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        cfg = self.config
        model_name = model or cfg.model
        start = time.perf_counter()
        if stream:
            chunks: List[str] = []
            usage = ChatUsage()
            with self.client.chat.completions.with_streaming_response.create(
                model=model_name,
                messages=messages,
                temperature=temperature if temperature is not None else cfg.temperature,
                top_p=top_p if top_p is not None else cfg.top_p,
                presence_penalty=presence_penalty if presence_penalty is not None else cfg.presence_penalty,
                frequency_penalty=frequency_penalty if frequency_penalty is not None else cfg.frequency_penalty,
                max_tokens=max_tokens if max_tokens is not None else cfg.max_output_tokens,
                timeout=cfg.request_timeout_s,
                **(extra or {}),
            ) as resp:
                for event in resp:
                    if event.type == "chunk":
                        delta = event.response.choices[0].delta.content or ""
                        chunks.append(delta)
                    elif event.type == "error":
                        raise APIError(message=str(event.error), request=None, response=None)
                text = "".join(chunks)
                raw = resp.get_final_response().model_dump()
                usage = ChatUsage(
                    prompt_tokens=raw.get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens=raw.get("usage", {}).get("completion_tokens", 0),
                    total_tokens=raw.get("usage", {}).get("total_tokens", 0),
                )
        else:
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature if temperature is not None else cfg.temperature,
                top_p=top_p if top_p is not None else cfg.top_p,
                presence_penalty=presence_penalty if presence_penalty is not None else cfg.presence_penalty,
                frequency_penalty=frequency_penalty if frequency_penalty is not None else cfg.frequency_penalty,
                max_tokens=max_tokens if max_tokens is not None else cfg.max_output_tokens,
                timeout=cfg.request_timeout_s,
                **(extra or {}),
            )
            raw = completion.model_dump()
            # Robust extraction of text across providers (incl. reasoning models)
            msg = completion.choices[0].message
            text = (getattr(msg, "content", None) or "") if msg is not None else ""
            if not text:
                try:
                    raw_msg = raw.get("choices", [{}])[0].get("message", {})
                    content = raw_msg.get("content")
                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            t = part.get("text") or part.get("output_text") or ""
                            if isinstance(t, str):
                                parts.append(t)
                        text = "".join(parts).strip()
                    elif isinstance(content, str):
                        text = content
                    if not text:
                        # Some providers return 'output_text' at the message level
                        text = (raw_msg.get("output_text") or "").strip()
                except Exception:
                    pass
            # If still empty, leave as-is; agent layer will trigger model fallback.
            usage = ChatUsage(
                prompt_tokens=raw.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=raw.get("usage", {}).get("completion_tokens", 0),
                total_tokens=raw.get("usage", {}).get("total_tokens", 0),
            )

        end = time.perf_counter()
        usage.latency_ms = (end - start) * 1000.0
        return ChatResult(text=text, usage=usage, raw=raw)

