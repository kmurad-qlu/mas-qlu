from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from ..env import load_env, get_openrouter_api_key, get_openai_base_url, get_openrouter_headers


Message = Dict[str, str]


# ----- Response Cache for deterministic queries -----

@dataclass
class CacheEntry:
    """An entry in the response cache."""
    result: "ChatResult"
    timestamp: float
    hit_count: int = 0


class ResponseCache:
    """
    LRU cache for LLM responses to avoid duplicate API calls.
    Only caches deterministic calls (temperature=0.0).
    
    Thread-safe for concurrent access.
    """
    
    def __init__(
        self,
        max_size: int = 256,
        ttl_seconds: float = 3600.0,
        enabled: bool = True,
    ):
        """
        Initialize the response cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds
            enabled: Whether caching is enabled
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._enabled = enabled
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _make_key(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
    ) -> str:
        """Generate a cache key from the request parameters."""
        # Create a deterministic hash of the messages and parameters
        key_data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    def get(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
    ) -> Optional["ChatResult"]:
        """
        Get a cached result if available and not expired.
        
        Only caches deterministic calls (temperature=0.0).
        """
        if not self._enabled or temperature != 0.0:
            return None
        
        key = self._make_key(messages, model, temperature)
        
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            
            # Check TTL
            if time.time() - entry.timestamp > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Update hit count
            entry.hit_count += 1
            self._hits += 1
            return entry.result
    
    def set(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        result: "ChatResult",
    ) -> None:
        """
        Cache a result.
        
        Only caches deterministic calls (temperature=0.0).
        """
        if not self._enabled or temperature != 0.0:
            return
        
        key = self._make_key(messages, model, temperature)
        
        with self._lock:
            # Evict oldest entry if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                # Find oldest entry
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].timestamp
                )
                del self._cache[oldest_key]
            
            self._cache[key] = CacheEntry(
                result=result,
                timestamp=time.time(),
                hit_count=0,
            )
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "enabled": self._enabled,
            }


# Global response cache instance
_response_cache: Optional[ResponseCache] = None
_cache_lock = threading.Lock()


def get_response_cache(
    max_size: int = 256,
    ttl_seconds: float = 3600.0,
    enabled: bool = True,
) -> ResponseCache:
    """
    Get or create the global response cache.
    
    Thread-safe singleton pattern.
    """
    global _response_cache
    
    with _cache_lock:
        if _response_cache is None:
            _response_cache = ResponseCache(
                max_size=max_size,
                ttl_seconds=ttl_seconds,
                enabled=enabled,
            )
        return _response_cache


def set_response_cache_enabled(enabled: bool) -> None:
    """Enable or disable the global response cache."""
    cache = get_response_cache()
    cache._enabled = enabled


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
    def __init__(self, config: OpenRouterConfig, cache_enabled: bool = True):
        load_env()
        api_key = get_openrouter_api_key()
        base_url = get_openai_base_url()
        headers = get_openrouter_headers()
        if not api_key:
            raise RuntimeError("OpenRouter API key missing. Set OPENROUTER_API_KEY.")
        self.client = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)
        self.config = config
        self.cache_enabled = cache_enabled
        self._cache = get_response_cache(enabled=cache_enabled)

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
        use_cache: bool = True,
    ) -> ChatResult:
        cfg = self.config
        model_name = model or cfg.model
        temp = temperature if temperature is not None else cfg.temperature
        
        # Check cache for deterministic calls (temperature=0.0, non-streaming)
        if use_cache and self.cache_enabled and temp == 0.0 and not stream:
            cached = self._cache.get(messages, model_name, temp)
            if cached is not None:
                # Return cached result with updated latency (cache hit is fast)
                cached_result = ChatResult(
                    text=cached.text,
                    usage=ChatUsage(
                        prompt_tokens=cached.usage.prompt_tokens,
                        completion_tokens=cached.usage.completion_tokens,
                        total_tokens=cached.usage.total_tokens,
                        latency_ms=0.1,  # Cache hit latency
                    ),
                    raw={"cached": True, **cached.raw},
                )
                return cached_result
        
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
        result = ChatResult(text=text, usage=usage, raw=raw)
        
        # Cache deterministic results (temperature=0.0, non-streaming)
        if use_cache and self.cache_enabled and temp == 0.0 and not stream:
            self._cache.set(messages, model_name, temp, result)
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the response cache."""
        return self._cache.stats()

