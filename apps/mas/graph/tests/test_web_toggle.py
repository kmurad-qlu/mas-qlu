from __future__ import annotations

import os
from types import SimpleNamespace


def test_web_disabled_skips_web_evidence(monkeypatch):
    # Ensure client can be constructed without real credentials.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    # Patch OpenRouterClient.complete_chat to avoid any network calls.
    from apps.mas.infra.openrouter import client as or_client

    def fake_complete_chat(self, messages, model=None, temperature=None, top_p=None, presence_penalty=None, frequency_penalty=None, max_tokens=None, stream=False, extra=None):
        sys = (messages[0].get("content") or "").lower()
        user = (messages[-1].get("content") or "").lower()

        # Supervisor decomposition expects JSON list.
        if "you are the supervisor. decompose" in sys or "decompose the following problem" in user:
            return SimpleNamespace(text='[{"role":"qa","instruction":"Answer the question directly."}]')

        # Critique expects OK.
        if "you are the supervisor critic" in sys:
            return SimpleNamespace(text="OK")

        # Numeric verification can just echo.
        if "verify_numeric" in user or "verifier" in sys:
            return SimpleNamespace(text="OK")

        # Synthesis: return a stable factual answer.
        if "you are the supervisor synthesizing" in sys:
            return SimpleNamespace(text="Between 1995-07-10 and 1995-07-15.")

        # Swarm worker responses.
        if "you are qaworker" in sys or "you are logicworker" in sys:
            return SimpleNamespace(text="Between 1995-07-10 and 1995-07-15.")

        return SimpleNamespace(text="OK")

    monkeypatch.setattr(or_client.OpenRouterClient, "complete_chat", fake_complete_chat, raising=True)

    from apps.mas.graph.plan_graph import solve_with_budget

    cfg = os.path.join("apps", "mas", "configs", "openrouter.yaml")
    thinking = []

    def cb(stage: str, content: str) -> None:
        thinking.append((stage, content))

    _ = solve_with_budget(
        problem="When was Darryl Prue sent home?",
        config_path=cfg,
        timeout_s=10.0,
        thinking_callback=cb,
        web_enabled=False,
    )

    stages = [s for s, _ in thinking]
    assert "web_disabled" in stages
    assert not any(s.startswith("web_evidence_") for s in stages)
    assert not any("web_fetch" in s for s in stages)


