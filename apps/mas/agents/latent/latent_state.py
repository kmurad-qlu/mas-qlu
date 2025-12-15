"""
Latent State data structures for inter-agent hidden state sharing.

These structures enable richer communication between agents beyond
simple text strings, including embeddings, confidence scores, and
attention maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import hashlib
import json

# Use list for embeddings to avoid numpy dependency issues
# Can be converted to numpy arrays when needed
EmbeddingType = List[float]


@dataclass
class AttentionMap:
    """
    Represents which parts of input/context the agent focused on.
    
    Enables other agents to understand what information was deemed
    important without re-analyzing the full context.
    """
    # Entity → importance weight (0-1)
    entity_weights: Dict[str, float] = field(default_factory=dict)
    
    # RAG chunk_id → relevance score (0-1)
    chunk_weights: Dict[str, float] = field(default_factory=dict)
    
    # Key phrases that were focused on
    key_phrases: List[str] = field(default_factory=list)
    
    def merge(self, other: "AttentionMap") -> "AttentionMap":
        """Merge two attention maps, taking maximum weights."""
        merged = AttentionMap()
        
        # Merge entity weights
        all_entities = set(self.entity_weights.keys()) | set(other.entity_weights.keys())
        for entity in all_entities:
            merged.entity_weights[entity] = max(
                self.entity_weights.get(entity, 0.0),
                other.entity_weights.get(entity, 0.0)
            )
        
        # Merge chunk weights
        all_chunks = set(self.chunk_weights.keys()) | set(other.chunk_weights.keys())
        for chunk_id in all_chunks:
            merged.chunk_weights[chunk_id] = max(
                self.chunk_weights.get(chunk_id, 0.0),
                other.chunk_weights.get(chunk_id, 0.0)
            )
        
        # Combine key phrases (deduplicated)
        merged.key_phrases = list(set(self.key_phrases + other.key_phrases))
        
        return merged
    
    def top_entities(self, k: int = 5) -> List[tuple]:
        """Return top-k entities by weight."""
        sorted_entities = sorted(
            self.entity_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_entities[:k]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entity_weights": self.entity_weights,
            "chunk_weights": self.chunk_weights,
            "key_phrases": self.key_phrases,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttentionMap":
        """Create from dictionary."""
        return cls(
            entity_weights=data.get("entity_weights", {}),
            chunk_weights=data.get("chunk_weights", {}),
            key_phrases=data.get("key_phrases", []),
        )


@dataclass
class ReasoningStep:
    """
    A compressed representation of a reasoning step.
    
    Enables other agents to understand the reasoning chain
    without parsing natural language explanations.
    """
    step_id: str
    step_type: str  # "definition", "calculation", "inference", "verification", "retrieval"
    input_summary: str  # Brief description of input
    conclusion: str  # Brief conclusion
    confidence: float  # 0-1 confidence in this step
    dependencies: List[str] = field(default_factory=list)  # Step IDs this depends on
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "input_summary": self.input_summary,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "dependencies": self.dependencies,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningStep":
        """Create from dictionary."""
        return cls(
            step_id=data["step_id"],
            step_type=data["step_type"],
            input_summary=data["input_summary"],
            conclusion=data["conclusion"],
            confidence=data.get("confidence", 0.5),
            dependencies=data.get("dependencies", []),
        )


@dataclass
class LatentState:
    """
    Hidden state shared between agents.
    
    Enables richer communication than plain text by including:
    - Dense embeddings for semantic similarity computation
    - Confidence scores for uncertainty propagation
    - Attention maps showing what information was important
    - Reasoning traces for understanding derivation
    
    Backward compatible: can be converted to plain text string.
    """
    # Identity
    agent_id: str
    task_id: str
    
    # Natural language output (backward compatible)
    text_output: str
    
    # Latent representations
    embedding: Optional[EmbeddingType] = None  # Dense vector
    confidence: float = 0.5  # Overall confidence in output (0-1)
    uncertainty_regions: List[str] = field(default_factory=list)  # Uncertain parts
    
    # Attention/focus
    attention: Optional[AttentionMap] = None
    
    # Reasoning trace
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    
    # Metadata
    model_name: str = ""
    latency_ms: float = 0.0
    token_count: int = 0
    timestamp: float = 0.0
    
    def to_context_string(
        self,
        include_confidence: bool = True,
        include_uncertainty: bool = True,
        include_key_entities: bool = False,
    ) -> str:
        """
        Convert to string for models that don't support latent input.
        This enables backward compatibility with existing text-based flow.
        """
        parts = [self.text_output]
        
        if include_confidence and self.confidence < 0.7:
            parts.append(f"\n[Latent: confidence={self.confidence:.2f}]")
        
        if include_uncertainty and self.uncertainty_regions:
            uncertain = ", ".join(self.uncertainty_regions[:3])
            parts.append(f"\n[Latent: uncertain about: {uncertain}]")
        
        if include_key_entities and self.attention:
            top_entities = self.attention.top_entities(3)
            if top_entities:
                entities_str = ", ".join(e[0] for e in top_entities)
                parts.append(f"\n[Latent: key entities: {entities_str}]")
        
        return "\n".join(parts)
    
    def get_embedding_hash(self) -> str:
        """Get a hash of the embedding for quick comparison."""
        if self.embedding is None:
            return ""
        # Hash first 10 values for speed
        sample = str(self.embedding[:10])
        return hashlib.md5(sample.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "text_output": self.text_output,
            "embedding": self.embedding,
            "confidence": self.confidence,
            "uncertainty_regions": self.uncertainty_regions,
            "attention": self.attention.to_dict() if self.attention else None,
            "reasoning_steps": [s.to_dict() for s in self.reasoning_steps],
            "model_name": self.model_name,
            "latency_ms": self.latency_ms,
            "token_count": self.token_count,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LatentState":
        """Create from dictionary."""
        attention = None
        if data.get("attention"):
            attention = AttentionMap.from_dict(data["attention"])
        
        reasoning_steps = [
            ReasoningStep.from_dict(s) 
            for s in data.get("reasoning_steps", [])
        ]
        
        return cls(
            agent_id=data["agent_id"],
            task_id=data["task_id"],
            text_output=data["text_output"],
            embedding=data.get("embedding"),
            confidence=data.get("confidence", 0.5),
            uncertainty_regions=data.get("uncertainty_regions", []),
            attention=attention,
            reasoning_steps=reasoning_steps,
            model_name=data.get("model_name", ""),
            latency_ms=data.get("latency_ms", 0.0),
            token_count=data.get("token_count", 0),
            timestamp=data.get("timestamp", 0.0),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "LatentState":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


def compute_embedding_similarity(
    emb1: Optional[EmbeddingType],
    emb2: Optional[EmbeddingType],
) -> float:
    """
    Compute cosine similarity between two embeddings.
    Returns 0.0 if either embedding is None.
    """
    if emb1 is None or emb2 is None:
        return 0.0
    
    if len(emb1) != len(emb2):
        return 0.0
    
    # Compute dot product and magnitudes
    dot_product = sum(a * b for a, b in zip(emb1, emb2))
    mag1 = sum(a * a for a in emb1) ** 0.5
    mag2 = sum(b * b for b in emb2) ** 0.5
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)

