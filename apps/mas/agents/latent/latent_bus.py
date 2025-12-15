"""
Latent Communication Bus for inter-agent hidden state sharing.

Provides a central pub/sub mechanism for agents to share their
latent states and subscribe to updates from other agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
import threading
import time

from .latent_state import LatentState, AttentionMap, compute_embedding_similarity


@dataclass
class Message:
    """A message on the latent bus."""
    sender: str
    recipient: str  # "*" for broadcast, specific agent_id for direct
    state: LatentState
    timestamp: float = field(default_factory=time.time)
    
    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message."""
        return self.recipient == "*"


@dataclass
class Disagreement:
    """Represents a disagreement between two agents."""
    agent1: str
    agent2: str
    similarity: float
    state1: LatentState
    state2: LatentState
    
    def describe(self) -> str:
        """Generate human-readable description."""
        return (
            f"Disagreement between {self.agent1} and {self.agent2} "
            f"(similarity={self.similarity:.3f})"
        )


class LatentCommunicationBus:
    """
    Central bus for inter-agent latent state sharing.
    
    Enables agents to:
    - Publish their latent states
    - Subscribe to other agents' updates
    - Query the collective state
    - Detect disagreements
    - Compute consensus embeddings
    
    Thread-safe for concurrent agent execution.
    """
    
    def __init__(
        self,
        max_history: int = 100,
        disagreement_threshold: float = 0.7,
    ):
        """
        Initialize the latent communication bus.
        
        Args:
            max_history: Maximum messages to keep in history
            disagreement_threshold: Similarity below which agents "disagree"
        """
        self.max_history = max_history
        self.disagreement_threshold = disagreement_threshold
        
        # Current states by agent
        self._states: Dict[str, LatentState] = {}
        
        # Message history
        self._history: List[Message] = []
        
        # Subscribers: topic → list of callbacks
        self._subscribers: Dict[str, List[Callable[[LatentState], None]]] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "messages_published": 0,
            "disagreements_detected": 0,
            "consensus_computed": 0,
        }
    
    def publish(
        self,
        state: LatentState,
        broadcast: bool = True,
        recipient: Optional[str] = None,
    ) -> None:
        """
        Publish an agent's latent state to the bus.
        
        Args:
            state: The latent state to publish
            broadcast: Whether to broadcast to all subscribers
            recipient: Specific recipient (if not broadcasting)
        """
        with self._lock:
            # Update current state
            self._states[state.agent_id] = state
            
            # Create message
            msg = Message(
                sender=state.agent_id,
                recipient="*" if broadcast else (recipient or ""),
                state=state,
                timestamp=time.time(),
            )
            
            # Add to history
            self._history.append(msg)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
            
            # Update stats
            self._stats["messages_published"] += 1
            
            # Notify subscribers
            subscribers = self._subscribers.get(state.agent_id, [])
            if broadcast:
                # Also notify "*" subscribers
                subscribers = subscribers + self._subscribers.get("*", [])
        
        # Call subscribers outside lock to avoid deadlock
        for callback in subscribers:
            try:
                callback(state)
            except Exception:
                pass  # Don't let subscriber errors crash the bus
    
    def get_state(self, agent_id: str) -> Optional[LatentState]:
        """Get an agent's latest latent state."""
        with self._lock:
            return self._states.get(agent_id)
    
    def get_all_states(self) -> Dict[str, LatentState]:
        """Get all agents' current latent states."""
        with self._lock:
            return dict(self._states)
    
    def get_states_by_task(self, task_id: str) -> List[LatentState]:
        """Get all states for a specific task."""
        with self._lock:
            return [
                state for state in self._states.values()
                if state.task_id == task_id
            ]
    
    def subscribe(
        self,
        agent_id: str,
        callback: Callable[[LatentState], None],
    ) -> None:
        """
        Subscribe to an agent's state updates.
        
        Use agent_id="*" to subscribe to all updates.
        """
        with self._lock:
            if agent_id not in self._subscribers:
                self._subscribers[agent_id] = []
            self._subscribers[agent_id].append(callback)
    
    def unsubscribe(
        self,
        agent_id: str,
        callback: Callable[[LatentState], None],
    ) -> None:
        """Unsubscribe from an agent's updates."""
        with self._lock:
            if agent_id in self._subscribers:
                try:
                    self._subscribers[agent_id].remove(callback)
                except ValueError:
                    pass
    
    def compute_consensus_embedding(
        self,
        task_id: Optional[str] = None,
    ) -> Optional[List[float]]:
        """
        Compute a consensus embedding from all agents.
        
        Weighted by confidence scores to give more weight to
        confident outputs.
        
        Args:
            task_id: If provided, only consider states for this task
        """
        with self._lock:
            states = list(self._states.values())
            if task_id:
                states = [s for s in states if s.task_id == task_id]
        
        embeddings = []
        weights = []
        
        for state in states:
            if state.embedding is not None:
                embeddings.append(state.embedding)
                weights.append(state.confidence)
        
        if not embeddings:
            return None
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total_weight for w in weights]
        
        # Compute weighted average
        dim = len(embeddings[0])
        consensus = [0.0] * dim
        
        for emb, weight in zip(embeddings, weights):
            for i in range(dim):
                consensus[i] += emb[i] * weight
        
        # Normalize
        magnitude = sum(x * x for x in consensus) ** 0.5
        if magnitude > 0:
            consensus = [x / magnitude for x in consensus]
        
        self._stats["consensus_computed"] += 1
        
        return consensus
    
    def find_disagreements(
        self,
        task_id: Optional[str] = None,
    ) -> List[Disagreement]:
        """
        Identify where agents disagree based on embedding distance.
        
        Returns pairs of agents with similarity below threshold.
        """
        with self._lock:
            states = list(self._states.values())
            if task_id:
                states = [s for s in states if s.task_id == task_id]
        
        disagreements = []
        
        for i, s1 in enumerate(states):
            for s2 in states[i + 1:]:
                if s1.embedding is None or s2.embedding is None:
                    continue
                
                similarity = compute_embedding_similarity(s1.embedding, s2.embedding)
                
                if similarity < self.disagreement_threshold:
                    disagreements.append(Disagreement(
                        agent1=s1.agent_id,
                        agent2=s2.agent_id,
                        similarity=similarity,
                        state1=s1,
                        state2=s2,
                    ))
        
        if disagreements:
            self._stats["disagreements_detected"] += len(disagreements)
        
        return disagreements
    
    def get_merged_attention(
        self,
        task_id: Optional[str] = None,
    ) -> AttentionMap:
        """
        Merge attention maps from all agents.
        
        Takes maximum weight for each entity across all agents.
        """
        with self._lock:
            states = list(self._states.values())
            if task_id:
                states = [s for s in states if s.task_id == task_id]
        
        merged = AttentionMap()
        
        for state in states:
            if state.attention:
                merged = merged.merge(state.attention)
        
        return merged
    
    def get_low_confidence_agents(
        self,
        threshold: float = 0.6,
        task_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get agents with confidence below threshold.
        
        Returns list of (agent_id, confidence) tuples.
        """
        with self._lock:
            states = list(self._states.values())
            if task_id:
                states = [s for s in states if s.task_id == task_id]
        
        return [
            (state.agent_id, state.confidence)
            for state in states
            if state.confidence < threshold
        ]
    
    def get_average_confidence(
        self,
        task_id: Optional[str] = None,
    ) -> float:
        """Get average confidence across all agents."""
        with self._lock:
            states = list(self._states.values())
            if task_id:
                states = [s for s in states if s.task_id == task_id]
        
        if not states:
            return 0.5
        
        return sum(s.confidence for s in states) / len(states)
    
    def get_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Message]:
        """
        Get message history.
        
        Args:
            agent_id: Filter by sender (None for all)
            limit: Maximum messages to return
        """
        with self._lock:
            history = self._history
            if agent_id:
                history = [m for m in history if m.sender == agent_id]
            return history[-limit:]
    
    def clear(self, task_id: Optional[str] = None) -> None:
        """
        Clear the bus state.
        
        Args:
            task_id: If provided, only clear states for this task
        """
        with self._lock:
            if task_id:
                self._states = {
                    k: v for k, v in self._states.items()
                    if v.task_id != task_id
                }
                self._history = [
                    m for m in self._history
                    if m.state.task_id != task_id
                ]
            else:
                self._states.clear()
                self._history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bus statistics."""
        with self._lock:
            return {
                **self._stats,
                "current_agents": len(self._states),
                "history_size": len(self._history),
                "subscriber_count": sum(len(v) for v in self._subscribers.values()),
            }
    
    def to_context_summary(
        self,
        task_id: Optional[str] = None,
        max_length: int = 1000,
    ) -> str:
        """
        Generate a summary for injection into prompts.
        
        Useful for providing latent context to models that
        don't directly support latent inputs.
        """
        parts = []
        
        # Average confidence
        avg_conf = self.get_average_confidence(task_id)
        parts.append(f"[Latent Bus: avg_confidence={avg_conf:.2f}]")
        
        # Low confidence warnings
        low_conf = self.get_low_confidence_agents(0.6, task_id)
        if low_conf:
            agents = ", ".join(f"{a}({c:.2f})" for a, c in low_conf[:3])
            parts.append(f"[Low confidence: {agents}]")
        
        # Disagreements
        disagreements = self.find_disagreements(task_id)
        if disagreements:
            desc = disagreements[0].describe()
            parts.append(f"[Warning: {desc}]")
        
        # Key entities from merged attention
        merged = self.get_merged_attention(task_id)
        top_entities = merged.top_entities(5)
        if top_entities:
            entities = ", ".join(e[0] for e in top_entities)
            parts.append(f"[Key entities: {entities}]")
        
        summary = "\n".join(parts)
        return summary[:max_length]


# Singleton instance for global access
_global_bus: Optional[LatentCommunicationBus] = None


def get_global_bus() -> LatentCommunicationBus:
    """Get the global latent communication bus singleton."""
    global _global_bus
    if _global_bus is None:
        _global_bus = LatentCommunicationBus()
    return _global_bus


def reset_global_bus() -> None:
    """Reset the global bus (useful for testing)."""
    global _global_bus
    _global_bus = None

