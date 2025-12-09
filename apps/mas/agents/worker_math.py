from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..infra.openrouter.client import OpenRouterClient


SYSTEM_MATH = (
    "You are MathWorker. Solve arithmetic, algebra, and mathematical problems precisely.\n"
    "- Think step by step internally.\n"
    "- For numeric answers, output EXACTLY one line in the format: '#### <final_numeric_answer>'\n"
    "- For non-numeric math answers (coordinates, expressions), provide the answer clearly.\n"
    "- ALWAYS provide an answer. Never return empty."
)


class MathWorker:
    def __init__(
        self,
        client: OpenRouterClient,
        model_name: str | None = None,
        fallback_model: str | None = None,
        secondary_fallback_model: str | None = None,
    ):
        self.client = client
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.secondary_fallback_model = secondary_fallback_model
        self._thinking_callback: Optional[Callable[[str, str], None]] = None

    def set_thinking_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set callback for intermediate thinking: callback(stage, content)"""
        self._thinking_callback = callback

    def _emit_thinking(self, stage: str, content: str) -> None:
        if self._thinking_callback:
            self._thinking_callback(stage, content)

    def run(self, instruction: str, context: str = "") -> str:
        self._emit_thinking("math_start", f"Computing: {instruction[:200]}...")
        
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_MATH},
            {
                "role": "user",
                "content": f"Task: {instruction}\nContext (optional): {context}\nProvide the final answer (use '#### <answer>' format for numeric results).",
            },
        ]
        
        self._emit_thinking("math_model", f"Querying primary model: {self.model_name or 'default'}")
        result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.model_name)
        text = result.text.strip()
        
        if not text and self.fallback_model:
            self._emit_thinking("math_fallback", f"Primary empty, trying fallback: {self.fallback_model}")
            result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.fallback_model)
            text = result.text.strip()
            
        if not text and self.secondary_fallback_model:
            self._emit_thinking("math_secondary", f"Fallback empty, trying secondary: {self.secondary_fallback_model}")
            result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.secondary_fallback_model)
            text = result.text.strip()
        
        # Final fallback: never return empty
        if not text:
            self._emit_thinking("math_emergency", "All models returned empty, using emergency fallback")
            text = f"Unable to compute: {instruction[:100]}. Please verify the mathematical expression."
        
        self._emit_thinking("math_complete", f"Result: {text[:200]}...")
        return text

