"""
Smoke tests for Codestral embeddings.

Run: pytest apps/mas/rag/tests/test_embeddings.py -v
"""

from __future__ import annotations

import os
import pytest
import numpy as np


@pytest.fixture
def embedder():
    """Create embedder instance, skipping if API key not available."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")
    
    from apps.mas.rag.embeddings import CodestralEmbedder
    return CodestralEmbedder(api_key=api_key)


class TestCodestralEmbedder:
    """Tests for the Codestral embedding client."""
    
    def test_embed_single_query(self, embedder):
        """Verify single query embedding returns correct dimension."""
        vec = embedder.embed_query("What is a Cayley graph?")
        
        assert vec is not None
        assert len(vec) == 1536  # Codestral dimension
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
    
    def test_embed_empty_query(self, embedder):
        """Verify empty query returns zero vector."""
        vec = embedder.embed_query("")
        
        assert vec is not None
        assert len(vec) == 1536
        assert np.allclose(vec, 0)
    
    def test_embed_batch_documents(self, embedder):
        """Verify batch embedding works correctly."""
        docs = [
            "Eigenvalues of matrices",
            "Knot theory and quandles",
            "Group representation theory"
        ]
        vecs = embedder.embed_documents(docs)
        
        assert vecs.shape == (3, 1536)
        assert vecs.dtype == np.float32
        # Each doc should have non-zero embedding
        for i in range(3):
            assert not np.allclose(vecs[i], 0)
    
    def test_embed_batch_with_empty(self, embedder):
        """Verify batch handles empty strings correctly."""
        docs = ["Valid text", "", "Another valid text"]
        vecs = embedder.embed_documents(docs)
        
        assert vecs.shape == (3, 1536)
        # Empty string should have zero vector
        assert np.allclose(vecs[1], 0)
        # Others should have non-zero
        assert not np.allclose(vecs[0], 0)
        assert not np.allclose(vecs[2], 0)
    
    def test_embedding_similarity(self, embedder):
        """Verify semantically similar texts have higher cosine similarity."""
        v1 = embedder.embed_query("mathematics algebra groups")
        v2 = embedder.embed_query("algebraic group theory")
        v3 = embedder.embed_query("cooking recipes pasta")
        
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_related = cosine_sim(v1, v2)
        sim_unrelated = cosine_sim(v1, v3)
        
        assert sim_related > sim_unrelated, \
            f"Related texts should have higher similarity: {sim_related:.4f} vs {sim_unrelated:.4f}"
    
    def test_embedding_deterministic(self, embedder):
        """Verify same text produces same embedding (deterministic)."""
        text = "Test determinism"
        v1 = embedder.embed_query(text)
        v2 = embedder.embed_query(text)
        
        # Should be identical (or very close due to floating point)
        assert np.allclose(v1, v2, rtol=1e-5)
    
    def test_embedder_dimension_property(self, embedder):
        """Verify dimension property is correct."""
        assert embedder.dimension == 1024
        assert embedder.DIMENSION == 1024

