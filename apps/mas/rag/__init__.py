"""
RAG (Retrieval-Augmented Generation) module for RA-TGR.

This module provides hybrid fusion retrieval combining:
- Semantic search via Codestral embeddings
- Lexical search via BM25
- Reciprocal Rank Fusion (RRF) for combining results
"""

from .embeddings import CodestralEmbedder
from .chunker import chunk_text, chunk_document
from .indexer import WikipediaIndexer, WikiChunk
from .retriever import HybridRetriever

__all__ = [
    "CodestralEmbedder",
    "chunk_text",
    "chunk_document",
    "WikipediaIndexer",
    "WikiChunk",
    "HybridRetriever",
]

