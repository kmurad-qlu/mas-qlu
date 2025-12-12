"""
Smoke tests for Wikipedia indexer.

Run: pytest apps/mas/rag/tests/test_indexer.py -v
"""

from __future__ import annotations

import tempfile
import pytest
import numpy as np


class MockEmbedder:
    """Mock embedder for testing without API calls."""
    
    dimension = 1536
    
    def embed_documents(self, texts):
        """Return random vectors for testing."""
        return np.random.rand(len(texts), 1536).astype('float32')
    
    def embed_query(self, query):
        """Return random vector for testing."""
        return np.random.rand(1536).astype('float32')


@pytest.fixture
def temp_db():
    """Create temporary database directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestChunker:
    """Tests for text chunking utilities."""
    
    def test_chunk_text_basic(self):
        """Verify basic chunking works."""
        from apps.mas.rag.chunker import chunk_text
        
        # Create text with multiple sentences
        original = "This is sentence one. This is sentence two. This is sentence three. " * 20
        chunks = chunk_text(original, chunk_size=200, overlap=20)
        
        assert len(chunks) > 1, "Should produce multiple chunks"
        for chunk in chunks:
            assert len(chunk) <= 250  # Allow some flexibility
    
    def test_chunk_text_preserves_content(self):
        """Verify chunking doesn't lose significant content."""
        from apps.mas.rag.chunker import chunk_text
        
        original = "Word " * 1000  # ~5000 chars
        chunks = chunk_text(original, chunk_size=512, overlap=50)
        
        assert len(chunks) > 1, "Should produce multiple chunks"
        # Verify no significant content loss (allowing for overlap)
        reconstructed_len = sum(len(c) for c in chunks)
        assert reconstructed_len >= len(original) * 0.9
    
    def test_chunk_text_short_text(self):
        """Verify short text is not chunked."""
        from apps.mas.rag.chunker import chunk_text
        
        short_text = "This is a short text."
        chunks = chunk_text(short_text, chunk_size=512)
        
        assert len(chunks) == 1
        assert chunks[0] == short_text
    
    def test_chunk_text_empty(self):
        """Verify empty text returns empty list."""
        from apps.mas.rag.chunker import chunk_text
        
        chunks = chunk_text("", chunk_size=512)
        assert chunks == []
    
    def test_chunk_document(self):
        """Verify document chunking with metadata."""
        from apps.mas.rag.chunker import chunk_document
        
        chunks = chunk_document(
            doc_id="test_123",
            title="Test Document",
            text="Content " * 200,
            url="http://example.com",
            chunk_size=256,
        )
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "id" in chunk
            assert "doc_id" in chunk
            assert chunk["doc_id"] == "test_123"
            assert "title" in chunk
            assert chunk["title"] == "Test Document"
            assert "text" in chunk
            assert "url" in chunk
            assert "chunk_idx" in chunk


class TestWikipediaIndexer:
    """Tests for the Wikipedia indexer."""
    
    def test_index_sample_documents(self, temp_db):
        """Verify indexing creates valid LanceDB table."""
        from apps.mas.rag.indexer import WikipediaIndexer
        
        indexer = WikipediaIndexer(temp_db, MockEmbedder())
        
        # Index sample documents
        sample_docs = [
            {
                "id": "1",
                "title": "Test Article",
                "text": "Sample content about mathematics and algebra. " * 20,
                "url": "http://test.com"
            },
            {
                "id": "2",
                "title": "Another Article",
                "text": "More content about eigenvalues and matrices. " * 20,
                "url": "http://test2.com"
            },
        ]
        chunks_indexed = indexer.index_documents(sample_docs, chunk_size=256, show_progress=False)
        
        assert chunks_indexed > 0
        
        # Verify table exists and has data
        assert indexer.count_chunks() > 0
    
    def test_indexer_clear(self, temp_db):
        """Verify clearing index works."""
        from apps.mas.rag.indexer import WikipediaIndexer
        
        indexer = WikipediaIndexer(temp_db, MockEmbedder())
        
        # Index some documents
        docs = [{"id": "1", "title": "Test", "text": "Content " * 50, "url": "http://test.com"}]
        indexer.index_documents(docs, show_progress=False)
        
        assert indexer.count_chunks() > 0
        
        # Clear
        indexer.clear()
        
        assert indexer.count_chunks() == 0
    
    def test_tokenize_for_bm25(self):
        """Verify BM25 tokenization."""
        from apps.mas.rag.indexer import tokenize_for_bm25
        
        text = "Hello, World! This is a TEST."
        tokens = tokenize_for_bm25(text)
        
        assert tokens == "hello world this is a test"
        assert "," not in tokens
        assert "!" not in tokens

