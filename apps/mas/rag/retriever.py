"""
Hybrid Retriever with Reciprocal Rank Fusion.

Combines semantic (dense vector) and lexical (BM25) search
using Reciprocal Rank Fusion (RRF) for improved retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import lancedb
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

from .embeddings import CodestralEmbedder
from .indexer import WikiChunk, tokenize_for_bm25


@dataclass
class RetrievedChunk:
    """A retrieved chunk with relevance score."""
    id: str
    doc_id: str
    title: str
    text: str
    url: str
    chunk_idx: int
    score: float
    retrieval_method: str  # "semantic", "lexical", or "fusion"
    
    def __repr__(self) -> str:
        return f"RetrievedChunk(title='{self.title[:30]}...', score={self.score:.4f}, method={self.retrieval_method})"


class HybridRetriever:
    """
    Fusion RAG: Combines semantic (dense) and lexical (sparse) retrieval
    using Reciprocal Rank Fusion (RRF).
    
    The retriever provides three search modes:
    1. semantic_search: Dense vector similarity using Codestral embeddings
    2. lexical_search: BM25-based full-text search
    3. fusion_search: RRF combination of both methods
    """
    
    TABLE_NAME = "wiki_chunks"
    
    def __init__(
        self,
        db_path: str,
        embedder: CodestralEmbedder,
        rrf_k: int = 60,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.5,
        thinking_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            db_path: Path to LanceDB database
            embedder: CodestralEmbedder for query embedding
            rrf_k: RRF parameter k (higher = more weight to top ranks)
            semantic_weight: Weight for semantic search in fusion
            lexical_weight: Weight for lexical search in fusion
            thinking_callback: Optional callback for logging
        """
        if not LANCEDB_AVAILABLE:
            raise ImportError("lancedb is required. Install with: pip install lancedb")
        
        self.db_path = db_path
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self._emit = thinking_callback or (lambda s, c: None)
        
        # Connect to database
        self.db = lancedb.connect(db_path)
        
        # Open table if it exists
        self._table = None
        if self.TABLE_NAME in self.db.table_names():
            self._table = self.db.open_table(self.TABLE_NAME)
    
    @property
    def table(self):
        """Get the LanceDB table, raising error if not available."""
        if self._table is None:
            if self.TABLE_NAME in self.db.table_names():
                self._table = self.db.open_table(self.TABLE_NAME)
            else:
                raise ValueError(
                    f"Table {self.TABLE_NAME} not found. "
                    "Run indexing first with WikipediaIndexer."
                )
        return self._table
    
    def _row_to_chunk(
        self,
        row: Dict[str, Any],
        score: float,
        method: str,
    ) -> RetrievedChunk:
        """Convert a LanceDB row to RetrievedChunk."""
        return RetrievedChunk(
            id=row.get("id", ""),
            doc_id=row.get("doc_id", ""),
            title=row.get("title", ""),
            text=row.get("text", ""),
            url=row.get("url", ""),
            chunk_idx=row.get("chunk_idx", 0),
            score=score,
            retrieval_method=method,
        )
    
    def semantic_search(
        self,
        query: str,
        k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Dense vector similarity search.
        
        Args:
            query: Search query text
            k: Number of results to return
        
        Returns:
            List of (chunk_id, distance) tuples, sorted by distance (lower is better)
        """
        if not query or not query.strip():
            return []
        
        # Embed query
        query_vec = self.embedder.embed_query(query)
        
        # Vector search
        results = (
            self.table
            .search(query_vec.tolist())
            .limit(k)
            .to_list()
        )
        
        return [(r["id"], r.get("_distance", 0.0)) for r in results]
    
    def semantic_search_full(
        self,
        query: str,
        k: int = 20,
    ) -> List[RetrievedChunk]:
        """
        Dense vector similarity search returning full chunks.
        
        Args:
            query: Search query text
            k: Number of results to return
        
        Returns:
            List of RetrievedChunk objects
        """
        if not query or not query.strip():
            return []
        
        query_vec = self.embedder.embed_query(query)
        
        results = (
            self.table
            .search(query_vec.tolist())
            .limit(k)
            .to_list()
        )
        
        chunks = []
        for r in results:
            # Convert distance to similarity score (1 / (1 + distance))
            distance = r.get("_distance", 0.0)
            score = 1.0 / (1.0 + distance)
            chunks.append(self._row_to_chunk(r, score, "semantic"))
        
        return chunks
    
    def lexical_search(
        self,
        query: str,
        k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        BM25/full-text search using LanceDB FTS.
        
        Args:
            query: Search query text
            k: Number of results to return
        
        Returns:
            List of (chunk_id, score) tuples, sorted by score (higher is better)
        """
        if not query or not query.strip():
            return []
        
        # Tokenize query for BM25
        tokenized_query = tokenize_for_bm25(query)
        
        if not tokenized_query:
            return []
        
        try:
            results = (
                self.table
                .search(tokenized_query, query_type="fts")
                .limit(k)
                .to_list()
            )
            return [(r["id"], r.get("_score", 0.0)) for r in results]
        except Exception as e:
            # FTS might not be indexed, fall back to manual filtering
            self._emit("retrieval_warning", f"FTS search failed: {e}, using fallback")
            return self._fallback_lexical_search(tokenized_query, k)
    
    def _fallback_lexical_search(
        self,
        tokenized_query: str,
        k: int,
    ) -> List[Tuple[str, float]]:
        """
        Fallback lexical search using simple token matching.
        
        Used when FTS index is not available.
        """
        query_tokens = set(tokenized_query.lower().split())
        if not query_tokens:
            return []
        
        # Scan table and score by token overlap
        all_rows = self.table.to_pandas()
        scores = []
        
        for _, row in all_rows.iterrows():
            doc_tokens = set(str(row.get("bm25_tokens", "")).split())
            if not doc_tokens:
                continue
            
            # Jaccard-like score
            intersection = len(query_tokens & doc_tokens)
            if intersection > 0:
                score = intersection / len(query_tokens)
                scores.append((row["id"], score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def lexical_search_full(
        self,
        query: str,
        k: int = 20,
    ) -> List[RetrievedChunk]:
        """
        Full-text search returning full chunks.
        
        Args:
            query: Search query text
            k: Number of results to return
        
        Returns:
            List of RetrievedChunk objects
        """
        if not query or not query.strip():
            return []
        
        tokenized_query = tokenize_for_bm25(query)
        if not tokenized_query:
            return []
        
        try:
            results = (
                self.table
                .search(tokenized_query, query_type="fts")
                .limit(k)
                .to_list()
            )
            
            chunks = []
            for r in results:
                score = r.get("_score", 0.0)
                chunks.append(self._row_to_chunk(r, score, "lexical"))
            
            return chunks
        except Exception:
            # Fallback
            id_scores = self._fallback_lexical_search(tokenized_query, k)
            return self._fetch_chunks_by_ids(
                [id_ for id_, _ in id_scores],
                {id_: score for id_, score in id_scores},
                "lexical"
            )
    
    def _fetch_chunks_by_ids(
        self,
        chunk_ids: List[str],
        scores: Dict[str, float],
        method: str,
    ) -> List[RetrievedChunk]:
        """Fetch full chunk data by IDs."""
        if not chunk_ids:
            return []
        
        # Build filter for IDs
        chunks = []
        df = self.table.to_pandas()
        
        for chunk_id in chunk_ids:
            row = df[df["id"] == chunk_id]
            if len(row) > 0:
                r = row.iloc[0].to_dict()
                chunks.append(self._row_to_chunk(r, scores.get(chunk_id, 0.0), method))
        
        return chunks
    
    def fusion_search(
        self,
        query: str,
        k: int = 10,
        semantic_k: Optional[int] = None,
        lexical_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Reciprocal Rank Fusion: RRF(d) = Σ 1/(k + rank(d))
        
        Combines semantic and lexical rankings using RRF.
        
        Args:
            query: Search query text
            k: Number of final results to return
            semantic_k: Number of semantic results to consider (default: k*2)
            lexical_k: Number of lexical results to consider (default: k*2)
        
        Returns:
            List of RetrievedChunk objects with fusion scores
        """
        if not query or not query.strip():
            return []
        
        semantic_k = semantic_k or k * 2
        lexical_k = lexical_k or k * 2
        
        self._emit("retrieval_start", f"Fusion search for: {query[:100]}...")
        
        # Get ranked results from both methods
        sem_results = self.semantic_search(query, k=semantic_k)
        lex_results = self.lexical_search(query, k=lexical_k)
        
        self._emit("retrieval_counts", f"Semantic: {len(sem_results)}, Lexical: {len(lex_results)}")
        
        # Compute RRF scores
        rrf_scores: Dict[str, float] = {}
        chunk_sources: Dict[str, List[str]] = {}  # Track which methods found each chunk
        
        # Semantic contribution (distance-based, so rank 0 = best)
        for rank, (doc_id, _) in enumerate(sem_results):
            rrf_contribution = self.semantic_weight / (self.rrf_k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_contribution
            chunk_sources.setdefault(doc_id, []).append("semantic")
        
        # Lexical contribution (score-based, already sorted by score desc)
        for rank, (doc_id, _) in enumerate(lex_results):
            rrf_contribution = self.lexical_weight / (self.rrf_k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_contribution
            chunk_sources.setdefault(doc_id, []).append("lexical")
        
        # Sort by fused RRF score
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )[:k]
        
        # Fetch full chunk data
        chunks = self._fetch_chunks_by_ids(
            sorted_ids,
            rrf_scores,
            "fusion"
        )
        
        # Update retrieval method to show sources
        for chunk in chunks:
            sources = chunk_sources.get(chunk.id, [])
            if len(sources) > 1:
                chunk.retrieval_method = "fusion(sem+lex)"
            elif sources:
                chunk.retrieval_method = f"fusion({sources[0]})"
        
        self._emit("retrieval_complete", f"Returned {len(chunks)} chunks via RRF fusion")
        
        return chunks
    
    def search(
        self,
        query: str,
        k: int = 10,
        method: str = "fusion",
    ) -> List[RetrievedChunk]:
        """
        Unified search interface.
        
        Args:
            query: Search query text
            k: Number of results
            method: "semantic", "lexical", or "fusion"
        
        Returns:
            List of RetrievedChunk objects
        """
        if method == "semantic":
            return self.semantic_search_full(query, k)
        elif method == "lexical":
            return self.lexical_search_full(query, k)
        else:
            return self.fusion_search(query, k)
    
    def format_context(
        self,
        chunks: List[RetrievedChunk],
        max_length: int = 4000,
        include_titles: bool = True,
    ) -> str:
        """
        Format retrieved chunks as context string.
        
        Args:
            chunks: List of retrieved chunks
            max_length: Maximum total context length
            include_titles: Whether to include chunk titles
        
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        formatted = []
        total_len = 0
        
        for i, chunk in enumerate(chunks, 1):
            if include_titles and chunk.title:
                header = f"[{i}] {chunk.title}"
            else:
                header = f"[{i}]"
            
            # Truncate chunk text if needed
            remaining = max_length - total_len - len(header) - 10
            if remaining <= 0:
                break
            
            text = chunk.text[:remaining] if len(chunk.text) > remaining else chunk.text
            entry = f"{header}\n{text}"
            
            formatted.append(entry)
            total_len += len(entry)
        
        return "\n\n".join(formatted)

