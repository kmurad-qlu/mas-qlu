"""
Wikipedia Indexer for LanceDB.

Provides ingestion pipeline to load Wikipedia Arrow dataset,
chunk documents, compute embeddings, and store in LanceDB
for hybrid retrieval.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

try:
    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False
    LanceModel = object
    Vector = None

from .chunker import chunk_document
from .embeddings import CodestralEmbedder


# BM25 tokenization utilities
def tokenize_for_bm25(text: str) -> str:
    """
    Tokenize text for BM25 lexical search.
    
    Converts text to lowercase, removes punctuation, and returns
    space-separated tokens for full-text search indexing.
    """
    if not text:
        return ""
    # Lowercase and remove non-alphanumeric (keep spaces)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@dataclass
class WikiChunk:
    """
    A chunk of Wikipedia article content.
    
    This dataclass represents a single indexed chunk with its
    embedding vector and metadata for retrieval.
    """
    id: str
    doc_id: str
    title: str
    text: str
    url: str
    chunk_idx: int
    vector: List[float]
    bm25_tokens: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LanceDB insertion."""
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "chunk_idx": self.chunk_idx,
            "vector": self.vector,
            "bm25_tokens": self.bm25_tokens,
        }


class WikipediaIndexer:
    """
    Indexer for Wikipedia Arrow dataset into LanceDB.
    
    Handles the full ingestion pipeline:
    1. Load Arrow dataset
    2. Chunk documents
    3. Compute embeddings via Codestral
    4. Store in LanceDB with FTS index
    """
    
    TABLE_NAME = "wiki_chunks"
    VECTOR_DIMENSION = 1536  # Codestral embedding dimension
    
    def __init__(
        self,
        db_path: str,
        embedder: CodestralEmbedder,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the Wikipedia indexer.
        
        Args:
            db_path: Path to LanceDB database directory
            embedder: CodestralEmbedder instance for computing embeddings
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        if not LANCEDB_AVAILABLE:
            raise ImportError("lancedb is required. Install with: pip install lancedb")
        
        self.db_path = db_path
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Ensure database directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Connect to LanceDB
        self.db = lancedb.connect(db_path)
    
    def _create_table_schema(self) -> pa.Schema:
        """Create PyArrow schema for the chunks table."""
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("text", pa.string()),
            pa.field("url", pa.string()),
            pa.field("chunk_idx", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), self.VECTOR_DIMENSION)),
            pa.field("bm25_tokens", pa.string()),
        ])
    
    def _load_arrow_dataset(self, arrow_path: str) -> Iterator[Dict[str, Any]]:
        """
        Load Wikipedia Arrow dataset and yield documents.
        
        Supports both HuggingFace datasets (Arrow IPC format) and standard Arrow files.
        
        Args:
            arrow_path: Path to Arrow dataset directory
        
        Yields:
            Document dictionaries with id, title, text, url, lang
        """
        path = Path(arrow_path)
        
        # Method 1: Try loading as HuggingFace dataset (most common for HF Arrow files)
        try:
            from datasets import load_from_disk, Dataset
            
            # Check if this is a HF dataset directory
            if (path / "dataset_info.json").exists() or (path / "state.json").exists():
                print(f"Loading as HuggingFace dataset from {arrow_path}...")
                dataset = load_from_disk(str(path))
                
                for item in dataset:
                    yield dict(item)
                return
        except Exception as e:
            print(f"HF dataset load failed, trying Arrow IPC: {e}")
        
        # Method 2: Try loading Arrow IPC files directly
        arrow_files = list(path.glob("*.arrow"))
        if not arrow_files:
            arrow_files = list(path.glob("**/data-*.arrow"))
        
        if not arrow_files:
            raise FileNotFoundError(f"No Arrow files found in {arrow_path}")
        
        for arrow_file in arrow_files:
            print(f"Loading Arrow IPC file: {arrow_file}...")
            try:
                # Try Arrow IPC format (used by HuggingFace)
                with pa.ipc.open_file(str(arrow_file)) as reader:
                    table = reader.read_all()
                    for i in range(len(table)):
                        row = {col: table.column(col)[i].as_py() for col in table.column_names}
                        yield row
            except pa.ArrowInvalid:
                # Fallback to streaming format
                try:
                    with pa.ipc.open_stream(str(arrow_file)) as reader:
                        for batch in reader:
                            table = pa.Table.from_batches([batch])
                            for i in range(len(table)):
                                row = {col: table.column(col)[i].as_py() for col in table.column_names}
                                yield row
                except Exception as e2:
                    # Final fallback: try as dataset
                    try:
                        dataset = ds.dataset(arrow_file, format="ipc")
                        for batch in dataset.to_batches():
                            table = pa.Table.from_batches([batch])
                            for i in range(len(table)):
                                row = {col: table.column(col)[i].as_py() for col in table.column_names}
                                yield row
                    except Exception as e3:
                        raise RuntimeError(
                            f"Could not load {arrow_file}. Tried IPC file, IPC stream, and dataset formats. "
                            f"Errors: {e}, {e2}, {e3}"
                        )
    
    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> int:
        """
        Index a list of documents.
        
        Args:
            documents: List of document dicts with id, title, text, url
            chunk_size: Override default chunk size
            show_progress: Whether to print progress
        
        Returns:
            Number of chunks indexed
        """
        chunk_size = chunk_size or self.chunk_size
        
        # Chunk all documents
        all_chunks = []
        for doc in documents:
            chunks = chunk_document(
                doc_id=doc.get("id", ""),
                title=doc.get("title", ""),
                text=doc.get("text", ""),
                url=doc.get("url", ""),
                chunk_size=chunk_size,
                overlap=self.chunk_overlap,
            )
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return 0
        
        if show_progress:
            print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Compute embeddings
        texts = [chunk["text"] for chunk in all_chunks]
        if show_progress:
            print("Computing embeddings...")
        embeddings = self.embedder.embed_documents(texts, show_progress=show_progress)
        
        # Prepare data for LanceDB
        records = []
        for i, chunk in enumerate(all_chunks):
            record = {
                "id": chunk["id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "text": chunk["text"],
                "url": chunk["url"],
                "chunk_idx": chunk["chunk_idx"],
                "vector": embeddings[i].tolist(),
                "bm25_tokens": tokenize_for_bm25(chunk["text"]),
            }
            records.append(record)
        
        # Create or append to table
        if self.TABLE_NAME in self.db.table_names():
            table = self.db.open_table(self.TABLE_NAME)
            table.add(records)
        else:
            table = self.db.create_table(self.TABLE_NAME, records)
            # Create FTS index for lexical search
            try:
                table.create_fts_index("bm25_tokens")
                if show_progress:
                    print("Created full-text search index on bm25_tokens")
            except Exception as e:
                print(f"Warning: Could not create FTS index: {e}")
        
        if show_progress:
            print(f"Indexed {len(records)} chunks to {self.TABLE_NAME}")
        
        return len(records)
    
    def ingest_arrow_dataset(
        self,
        arrow_path: str,
        chunk_size: Optional[int] = None,
        batch_size: int = 100,
        max_documents: Optional[int] = None,
        show_progress: bool = True,
    ) -> int:
        """
        Ingest Wikipedia Arrow dataset into LanceDB.
        
        Args:
            arrow_path: Path to Arrow dataset directory
            chunk_size: Override default chunk size
            batch_size: Number of documents to process at a time
            max_documents: Maximum documents to index (None for all)
            show_progress: Whether to print progress
        
        Returns:
            Total number of chunks indexed
        """
        chunk_size = chunk_size or self.chunk_size
        total_chunks = 0
        doc_count = 0
        batch = []
        
        if show_progress:
            print(f"Ingesting Arrow dataset from {arrow_path}")
            print(f"Chunk size: {chunk_size}, overlap: {self.chunk_overlap}")
        
        for doc in self._load_arrow_dataset(arrow_path):
            batch.append(doc)
            doc_count += 1
            
            if len(batch) >= batch_size:
                chunks_added = self.index_documents(batch, chunk_size, show_progress)
                total_chunks += chunks_added
                batch = []
                
                if show_progress:
                    print(f"Progress: {doc_count} documents, {total_chunks} chunks")
            
            if max_documents and doc_count >= max_documents:
                break
        
        # Process remaining batch
        if batch:
            chunks_added = self.index_documents(batch, chunk_size, show_progress)
            total_chunks += chunks_added
        
        if show_progress:
            print(f"\nCompleted: {doc_count} documents, {total_chunks} chunks indexed")
        
        return total_chunks
    
    def get_table(self):
        """Get the LanceDB table for direct queries."""
        if self.TABLE_NAME not in self.db.table_names():
            raise ValueError(f"Table {self.TABLE_NAME} does not exist. Run indexing first.")
        return self.db.open_table(self.TABLE_NAME)
    
    def count_chunks(self) -> int:
        """Get the total number of indexed chunks."""
        if self.TABLE_NAME not in self.db.table_names():
            return 0
        table = self.db.open_table(self.TABLE_NAME)
        return table.count_rows()
    
    def clear(self) -> None:
        """Clear all indexed data."""
        if self.TABLE_NAME in self.db.table_names():
            self.db.drop_table(self.TABLE_NAME)
            print(f"Dropped table {self.TABLE_NAME}")

