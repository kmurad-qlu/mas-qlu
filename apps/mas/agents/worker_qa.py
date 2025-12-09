from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..infra.openrouter.client import OpenRouterClient


SYSTEM_QA = (
    "You are QAWorker. Answer factual and knowledge questions thoroughly.\n"
    "- For simple factual questions, provide a concise answer (1-2 sentences).\n"
    "- For complex questions (humanities, history, literature, philosophy, science), "
    "provide a well-reasoned explanation with key details.\n"
    "- If yes/no, answer 'yes' or 'no' followed by a brief justification.\n"
    "- ALWAYS provide a substantive answer. Never return empty or refuse to answer.\n"
    "- If uncertain, provide your best assessment with caveats."
)


class QAWorker:
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
        self._emit_thinking("qa_start", f"Processing: {instruction[:200]}...")
        
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_QA},
            {
                "role": "user",
                "content": f"Question: {instruction}\nContext (optional): {context}\nAnswer:",
            },
        ]
        
        self._emit_thinking("qa_model", f"Querying primary model: {self.model_name or 'default'}")
        result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.model_name)
        text = result.text.strip()
        
        if not text and self.fallback_model:
            self._emit_thinking("qa_fallback", f"Primary empty, trying fallback: {self.fallback_model}")
            result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.fallback_model)
            text = result.text.strip()
            
        if not text and self.secondary_fallback_model:
            self._emit_thinking("qa_secondary", f"Fallback empty, trying secondary: {self.secondary_fallback_model}")
            result = self.client.complete_chat(messages=messages, temperature=0.0, model=self.secondary_fallback_model)
            text = result.text.strip()
        
        # Final fallback: never return empty
        if not text:
            self._emit_thinking("qa_emergency", "All models returned empty, using emergency fallback")
            text = f"Unable to provide a definitive answer for: {instruction[:100]}. Please rephrase or provide more context."
        
        self._emit_thinking("qa_complete", f"Answer: {text[:200]}...")
        return text

