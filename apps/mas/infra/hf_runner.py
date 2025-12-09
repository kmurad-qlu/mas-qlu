from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFRunnerConfig:
    model_name_or_path: str
    load_in_4bit: bool = False
    device_map: str = "auto"
    torch_dtype: str = "auto"  # "auto" | "float16" | "bfloat16" | "float32"


class HFRunner:
    def __init__(self, cfg: HFRunnerConfig):
        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(cfg.torch_dtype, "auto")

        quant_kwargs: Dict[str, Any] = {}
        if cfg.load_in_4bit:
            quant_kwargs = {"load_in_4bit": True}

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            device_map=cfg.device_map,
            torch_dtype=torch_dtype,
            **quant_kwargs,
        )
        self.model.eval()

    @torch.inference_mode()
    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    @torch.inference_mode()
    def get_input_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.model.get_input_embeddings()
        return emb(input_ids)

    @torch.inference_mode()
    def forward_with_hidden(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]
        return {
            "last_hidden": last_hidden,
            "past_key_values": outputs.past_key_values,
            "logits": outputs.logits,
        }

    @torch.inference_mode()
    def step_inputs_embeds(self, inputs_embeds: torch.Tensor, past_key_values: Any) -> Dict[str, Any]:
        """
        One decoding step that consumes a 1-token inputs_embeds with an existing KV cache.
        inputs_embeds: [1, 1, embed_dim]
        """
        # Ensure dtype/device match model parameters to avoid matmul dtype errors.
        target_dtype = self.model.get_input_embeddings().weight.dtype
        inputs_embeds = inputs_embeds.to(dtype=target_dtype, device=self.model.device)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return {"past_key_values": outputs.past_key_values, "logits": outputs.logits}

    @staticmethod
    def _sample_top_p(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 0.7) -> int:
        logits = logits / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        # keep first token above threshold
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
        sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-8)
        next_idx = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, next_idx)
        return int(next_token.item())

    @torch.inference_mode()
    def continue_with_past(
        self,
        token_ids: List[int],
        past_key_values: Any,
        max_new_tokens: int = 32,
    ) -> Tuple[str, Any]:
        device = self.model.device
        generated: List[int] = []
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)

        pkv = past_key_values
        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn,
                past_key_values=pkv,
                use_cache=True,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            generated.append(int(next_token_id.item()))
            input_ids = next_token_id.unsqueeze(0)
            attn = torch.ones_like(input_ids)
            pkv = outputs.past_key_values
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text, pkv

    @torch.inference_mode()
    def generate_sampled_from_pkv(
        self,
        past_key_values: Any,
        seed_token_id: int,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        greedy_after: int = 8,
    ) -> str:
        """
        Continue generation from an existing KV cache using sampling for the first
        few tokens, then greedy.
        """
        device = self.model.device
        pkv = past_key_values
        generated: List[int] = []
        input_ids = torch.tensor([[seed_token_id]], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        for step in range(max_new_tokens):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn,
                past_key_values=pkv,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            if step < greedy_after:
                next_id = self._sample_top_p(logits.squeeze(0), top_p=top_p, temperature=temperature)
                next_token_id = torch.tensor([[next_id]], dtype=torch.long, device=device)
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            token_int = int(next_token_id.item())
            generated.append(token_int)
            if eos_token_id is not None and token_int == eos_token_id:
                break
            input_ids = next_token_id
            attn = torch.ones_like(input_ids)
            pkv = outputs.past_key_values
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text

