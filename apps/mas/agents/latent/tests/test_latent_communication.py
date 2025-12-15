"""Tests for latent communication module."""

import pytest
import time

from ..latent_state import (
    LatentState,
    AttentionMap,
    ReasoningStep,
    compute_embedding_similarity,
)
from ..latent_encoder import LatentEncoder
from ..latent_bus import LatentCommunicationBus, Message, reset_global_bus


class TestAttentionMap:
    """Tests for AttentionMap."""
    
    def test_attention_map_creation(self):
        """Test basic AttentionMap creation."""
        attention = AttentionMap(
            entity_weights={"France": 0.9, "Paris": 0.7},
            key_phrases=["capital city"],
        )
        assert attention.entity_weights["France"] == 0.9
        assert "capital city" in attention.key_phrases
    
    def test_attention_map_merge(self):
        """Test merging two attention maps."""
        a1 = AttentionMap(entity_weights={"A": 0.5, "B": 0.3})
        a2 = AttentionMap(entity_weights={"A": 0.7, "C": 0.4})
        
        merged = a1.merge(a2)
        
        assert merged.entity_weights["A"] == 0.7  # Max of 0.5 and 0.7
        assert merged.entity_weights["B"] == 0.3
        assert merged.entity_weights["C"] == 0.4
    
    def test_top_entities(self):
        """Test getting top entities by weight."""
        attention = AttentionMap(
            entity_weights={"A": 0.9, "B": 0.5, "C": 0.3, "D": 0.1}
        )
        
        top = attention.top_entities(2)
        
        assert len(top) == 2
        assert top[0][0] == "A"
        assert top[1][0] == "B"
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        original = AttentionMap(
            entity_weights={"Test": 0.5},
            chunk_weights={"chunk_0": 0.8},
            key_phrases=["test phrase"],
        )
        
        data = original.to_dict()
        restored = AttentionMap.from_dict(data)
        
        assert restored.entity_weights == original.entity_weights
        assert restored.chunk_weights == original.chunk_weights
        assert restored.key_phrases == original.key_phrases


class TestLatentState:
    """Tests for LatentState."""
    
    def test_latent_state_creation(self):
        """Test basic LatentState creation."""
        state = LatentState(
            agent_id="test_agent",
            task_id="test_task",
            text_output="The answer is 42.",
            confidence=0.9,
        )
        
        assert state.agent_id == "test_agent"
        assert state.text_output == "The answer is 42."
        assert state.confidence == 0.9
    
    def test_to_context_string_basic(self):
        """Test basic context string generation."""
        state = LatentState(
            agent_id="agent1",
            task_id="task1",
            text_output="Simple answer",
            confidence=0.8,
        )
        
        ctx = state.to_context_string()
        assert "Simple answer" in ctx
    
    def test_to_context_string_low_confidence(self):
        """Test context string with low confidence warning."""
        state = LatentState(
            agent_id="agent1",
            task_id="task1",
            text_output="Uncertain answer",
            confidence=0.4,
        )
        
        ctx = state.to_context_string(include_confidence=True)
        assert "Uncertain answer" in ctx
        assert "confidence=0.4" in ctx
    
    def test_json_serialization(self):
        """Test JSON serialization round-trip."""
        original = LatentState(
            agent_id="agent1",
            task_id="task1",
            text_output="Test output",
            embedding=[0.1, 0.2, 0.3],
            confidence=0.75,
            attention=AttentionMap(entity_weights={"Test": 0.9}),
        )
        
        json_str = original.to_json()
        restored = LatentState.from_json(json_str)
        
        assert restored.agent_id == original.agent_id
        assert restored.text_output == original.text_output
        assert restored.embedding == original.embedding
        assert restored.confidence == original.confidence


class TestEmbeddingSimilarity:
    """Tests for embedding similarity computation."""
    
    def test_identical_embeddings(self):
        """Test similarity of identical embeddings."""
        emb = [0.5, 0.5, 0.5]
        sim = compute_embedding_similarity(emb, emb)
        assert abs(sim - 1.0) < 0.001
    
    def test_orthogonal_embeddings(self):
        """Test similarity of orthogonal embeddings."""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        sim = compute_embedding_similarity(emb1, emb2)
        assert abs(sim) < 0.001
    
    def test_none_embeddings(self):
        """Test handling of None embeddings."""
        emb = [0.1, 0.2]
        assert compute_embedding_similarity(None, emb) == 0.0
        assert compute_embedding_similarity(emb, None) == 0.0
        assert compute_embedding_similarity(None, None) == 0.0


class TestLatentEncoder:
    """Tests for LatentEncoder."""
    
    def test_encode_text(self):
        """Test basic text encoding."""
        encoder = LatentEncoder(embedding_dim=128)
        emb = encoder.encode_text("Hello world")
        
        assert len(emb) == 128
        assert any(v != 0 for v in emb)  # Non-zero embedding
    
    def test_encode_empty_text(self):
        """Test encoding empty text."""
        encoder = LatentEncoder(embedding_dim=128)
        emb = encoder.encode_text("")
        
        assert len(emb) == 128
        assert all(v == 0 for v in emb)  # All zeros for empty
    
    def test_encoding_consistency(self):
        """Test that same text produces same embedding."""
        encoder = LatentEncoder(embedding_dim=128)
        emb1 = encoder.encode_text("test text")
        emb2 = encoder.encode_text("test text")
        
        assert emb1 == emb2
    
    def test_estimate_confidence_high(self):
        """Test confidence estimation for confident text."""
        encoder = LatentEncoder()
        conf = encoder.estimate_confidence(
            "The answer is definitely 42. This is clearly the case."
        )
        assert conf > 0.7
    
    def test_estimate_confidence_low(self):
        """Test confidence estimation for uncertain text."""
        encoder = LatentEncoder()
        conf = encoder.estimate_confidence(
            "I'm not sure, but possibly the answer might be around 42."
        )
        assert conf < 0.7
    
    def test_extract_attention(self):
        """Test attention extraction."""
        encoder = LatentEncoder()
        attention = encoder.extract_attention(
            "The answer involves France and Germany.",
            "Context about European countries..."
        )
        
        assert "France" in attention.entity_weights
        assert "Germany" in attention.entity_weights
    
    def test_create_latent_state(self):
        """Test full latent state creation."""
        encoder = LatentEncoder(embedding_dim=128)
        state = encoder.create_latent_state(
            agent_id="test_agent",
            task_id="test_task",
            text_output="The answer is definitely Paris.",
            include_embedding=True,
            include_attention=True,
        )
        
        assert state.agent_id == "test_agent"
        assert state.embedding is not None
        assert len(state.embedding) == 128
        assert state.attention is not None
        assert "Paris" in state.attention.entity_weights


class TestLatentCommunicationBus:
    """Tests for LatentCommunicationBus."""
    
    def setup_method(self):
        """Reset global bus before each test."""
        reset_global_bus()
    
    def test_publish_and_get_state(self):
        """Test basic publish and retrieval."""
        bus = LatentCommunicationBus()
        state = LatentState(
            agent_id="agent1",
            task_id="task1",
            text_output="Test output",
        )
        
        bus.publish(state)
        retrieved = bus.get_state("agent1")
        
        assert retrieved is not None
        assert retrieved.agent_id == "agent1"
        assert retrieved.text_output == "Test output"
    
    def test_get_all_states(self):
        """Test getting all states."""
        bus = LatentCommunicationBus()
        
        bus.publish(LatentState(agent_id="a1", task_id="t1", text_output="o1"))
        bus.publish(LatentState(agent_id="a2", task_id="t1", text_output="o2"))
        
        all_states = bus.get_all_states()
        
        assert len(all_states) == 2
        assert "a1" in all_states
        assert "a2" in all_states
    
    def test_find_disagreements(self):
        """Test disagreement detection."""
        bus = LatentCommunicationBus(disagreement_threshold=0.9)
        
        # Create states with different embeddings
        s1 = LatentState(
            agent_id="a1", task_id="t1", text_output="o1",
            embedding=[1.0, 0.0, 0.0, 0.0]
        )
        s2 = LatentState(
            agent_id="a2", task_id="t1", text_output="o2",
            embedding=[0.0, 1.0, 0.0, 0.0]  # Orthogonal = similarity 0
        )
        
        bus.publish(s1)
        bus.publish(s2)
        
        disagreements = bus.find_disagreements()
        
        assert len(disagreements) == 1
        assert disagreements[0].similarity < 0.9
    
    def test_compute_consensus_embedding(self):
        """Test consensus embedding computation."""
        bus = LatentCommunicationBus()
        
        bus.publish(LatentState(
            agent_id="a1", task_id="t1", text_output="o1",
            embedding=[1.0, 0.0], confidence=0.5
        ))
        bus.publish(LatentState(
            agent_id="a2", task_id="t1", text_output="o2",
            embedding=[0.0, 1.0], confidence=0.5
        ))
        
        consensus = bus.compute_consensus_embedding()
        
        assert consensus is not None
        assert len(consensus) == 2
    
    def test_get_average_confidence(self):
        """Test average confidence calculation."""
        bus = LatentCommunicationBus()
        
        bus.publish(LatentState(agent_id="a1", task_id="t1", text_output="o1", confidence=0.8))
        bus.publish(LatentState(agent_id="a2", task_id="t1", text_output="o2", confidence=0.6))
        
        avg = bus.get_average_confidence()
        
        assert abs(avg - 0.7) < 0.001
    
    def test_get_low_confidence_agents(self):
        """Test finding low-confidence agents."""
        bus = LatentCommunicationBus()
        
        bus.publish(LatentState(agent_id="high", task_id="t1", text_output="o1", confidence=0.9))
        bus.publish(LatentState(agent_id="low", task_id="t1", text_output="o2", confidence=0.4))
        
        low_conf = bus.get_low_confidence_agents(0.6)
        
        assert len(low_conf) == 1
        assert low_conf[0][0] == "low"
        assert low_conf[0][1] == 0.4
    
    def test_merged_attention(self):
        """Test merged attention from all agents."""
        bus = LatentCommunicationBus()
        
        bus.publish(LatentState(
            agent_id="a1", task_id="t1", text_output="o1",
            attention=AttentionMap(entity_weights={"A": 0.5})
        ))
        bus.publish(LatentState(
            agent_id="a2", task_id="t1", text_output="o2",
            attention=AttentionMap(entity_weights={"A": 0.8, "B": 0.3})
        ))
        
        merged = bus.get_merged_attention()
        
        assert merged.entity_weights["A"] == 0.8  # Max
        assert merged.entity_weights["B"] == 0.3
    
    def test_clear_by_task(self):
        """Test clearing states by task ID."""
        bus = LatentCommunicationBus()
        
        bus.publish(LatentState(agent_id="a1", task_id="keep", text_output="o1"))
        bus.publish(LatentState(agent_id="a2", task_id="remove", text_output="o2"))
        
        bus.clear(task_id="remove")
        
        assert bus.get_state("a1") is not None
        assert bus.get_state("a2") is None
    
    def test_to_context_summary(self):
        """Test context summary generation."""
        bus = LatentCommunicationBus()
        
        bus.publish(LatentState(
            agent_id="a1", task_id="t1", text_output="o1",
            confidence=0.9,
            attention=AttentionMap(entity_weights={"France": 0.8})
        ))
        
        summary = bus.to_context_summary()
        
        assert "Latent Bus" in summary
        assert "France" in summary

