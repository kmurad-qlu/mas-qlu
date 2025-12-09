from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..infra.openrouter.client import OpenRouterClient


SYSTEM_LOGIC = (
    "You are LogicWorker. Perform logical deduction and multi-step reasoning.\n"
    "- For simple logical problems, provide a brief conclusion with key reasoning.\n"
    "- For complex reasoning problems (puzzles, proofs, multi-step analysis), "
    "show your step-by-step logical process.\n"
    "- For boolean/true-false questions, state the answer clearly with justification.\n"
    "- ALWAYS provide a substantive answer. Never return empty or refuse to reason.\n"
    "- If the problem is ambiguous, state your assumptions and proceed."
)


class LogicWorker:
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
        self._emit_thinking("logic_start", f"Analyzing: {instruction[:200]}...")
        
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_LOGIC},
            {
                "role": "user",
                "content": f"Task: {instruction}\nContext (optional): {context}\nReasoning and Conclusion:",
            },
        ]
        
        self._emit_thinking("logic_model", f"Querying primary model: {self.model_name or 'default'}")
        result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.model_name)
        text = result.text.strip()
        
        if not text and self.fallback_model:
            self._emit_thinking("logic_fallback", f"Primary empty, trying fallback: {self.fallback_model}")
            result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.fallback_model)
            text = result.text.strip()
            
        if not text and self.secondary_fallback_model:
            self._emit_thinking("logic_secondary", f"Fallback empty, trying secondary: {self.secondary_fallback_model}")
            result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.secondary_fallback_model)
            text = result.text.strip()
        
        # Final fallback: never return empty
        if not text:
            self._emit_thinking("logic_emergency", "All models returned empty, using emergency fallback")
            text = f"Unable to derive a logical conclusion for: {instruction[:100]}. The problem may require additional context or constraints."
        
        self._emit_thinking("logic_complete", f"Conclusion: {text[:200]}...")
        return text

