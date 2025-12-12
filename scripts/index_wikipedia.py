#!/usr/bin/env python3
"""
Script to ingest Wikipedia Arrow dataset into LanceDB.

This script loads the Wikipedia subset dataset (Arrow format),
chunks the documents, computes embeddings using Codestral,
and stores everything in a LanceDB database for hybrid retrieval.

Usage:
    python scripts/index_wikipedia.py --arrow-path wikipedia-subset-hf-dataset/wikipedia-subset/

Options:
    --arrow-path    Path to the Arrow dataset directory
    --db-path       Path to LanceDB database (default: apps/mas/data/wiki_lance)
    --chunk-size    Target chunk size in characters (default: 512)
    --chunk-overlap Overlap between chunks (default: 50)
    --batch-size    Documents per batch (default: 100)
    --max-docs      Maximum documents to index (default: all)
    --clear         Clear existing index before ingesting
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Wikipedia Arrow dataset into LanceDB for RAG"
    )
    parser.add_argument(
        "--arrow-path",
        type=str,
        required=True,
        help="Path to Arrow dataset directory",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="apps/mas/data/wiki_lance",
        help="Path to LanceDB database",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Documents per batch",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to index (default: all)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before ingesting",
    )
    
    args = parser.parse_args()
    
    # Check Arrow path exists
    if not os.path.exists(args.arrow_path):
        print(f"Error: Arrow path does not exist: {args.arrow_path}")
        sys.exit(1)
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)
    
    print("=" * 60)
    print("Wikipedia RAG Indexer")
    print("=" * 60)
    print(f"Arrow path:    {args.arrow_path}")
    print(f"Database path: {args.db_path}")
    print(f"Chunk size:    {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Max documents: {args.max_docs or 'all'}")
    print("=" * 60)
    
    # Import after path setup
    from apps.mas.rag.embeddings import CodestralEmbedder
    from apps.mas.rag.indexer import WikipediaIndexer
    
    # Initialize embedder
    print("\n[1/3] Initializing Codestral embedder...")
    start_time = time.time()
    embedder = CodestralEmbedder(api_key=api_key)
    print(f"  Embedder ready (model: {embedder.MODEL}, dim: {embedder.dimension})")
    
    # Initialize indexer
    print("\n[2/3] Initializing LanceDB indexer...")
    indexer = WikipediaIndexer(
        db_path=args.db_path,
        embedder=embedder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    
    if args.clear:
        print("  Clearing existing index...")
        indexer.clear()
    
    existing_count = indexer.count_chunks()
    print(f"  Database ready (existing chunks: {existing_count})")
    
    # Ingest dataset
    print("\n[3/3] Ingesting Arrow dataset...")
    total_chunks = indexer.ingest_arrow_dataset(
        arrow_path=args.arrow_path,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        max_documents=args.max_docs,
        show_progress=True,
    )
    
    elapsed = time.time() - start_time
    final_count = indexer.count_chunks()
    
    print("\n" + "=" * 60)
    print("Indexing Complete!")
    print("=" * 60)
    print(f"Total chunks indexed: {total_chunks}")
    print(f"Total chunks in DB:   {final_count}")
    print(f"Time elapsed:         {elapsed:.1f}s")
    print(f"Database location:    {args.db_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

