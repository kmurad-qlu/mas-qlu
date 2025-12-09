from __future__ import annotations

from typing import Optional

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from ..infra.hf_runner import HFRunner, HFRunnerConfig
from ..latent.wa_train import load_alignment
from ..latent.latent_io import project_hidden_to_inputs_embeds, nearest_token_ids


class LatentState(BaseModel):
    question: str
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    alignment_path: str
    final_answer: Optional[str] = None


def build_latent_pipeline() -> StateGraph:
    def node_run(state: LatentState) -> LatentState:
        runner = HFRunner(HFRunnerConfig(model_name_or_path=state.model_name))
        W = load_alignment(state.alignment_path)
        prompt = f"You are a helpful math solver. Solve concisely.\n\nQuestion:\n{state.question}\n\nAnswer:"
        enc = runner.encode(prompt)
        out = runner.forward_with_hidden(enc["input_ids"], enc.get("attention_mask"))
        last_hidden = out["last_hidden"]
        pkv = out["past_key_values"]
        embeds = project_hidden_to_inputs_embeds(last_hidden[:, -1:, :], W)
        seed_ids = nearest_token_ids(runner, embeds)[0].tolist()
        text, _ = runner.continue_with_past(token_ids=seed_ids, past_key_values=pkv, max_new_tokens=64)
        return LatentState(
            question=state.question,
            model_name=state.model_name,
            alignment_path=state.alignment_path,
            final_answer=text.strip(),
        )

    graph = StateGraph(LatentState)
    graph.add_node("latent_run", node_run)
    graph.set_entry_point("latent_run")
    graph.add_edge("latent_run", END)
    return graph

