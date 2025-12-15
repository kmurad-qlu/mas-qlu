"""
Latent Communication Module for Inter-Agent Hidden State Sharing.

This module enables agents to share rich latent representations beyond
natural language, including:
- Dense embeddings of outputs
- Confidence scores
- Attention maps showing focus areas
- Reasoning traces

This enables richer consensus computation, disagreement detection,
and information propagation between agents.
"""

from .latent_state import (
    LatentState,
    AttentionMap,
    ReasoningStep,
)
from .latent_encoder import LatentEncoder
from .latent_bus import LatentCommunicationBus, Message

__all__ = [
    "LatentState",
    "AttentionMap", 
    "ReasoningStep",
    "LatentEncoder",
    "LatentCommunicationBus",
    "Message",
]

