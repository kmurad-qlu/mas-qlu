#!/usr/bin/env python3
"""
Quick smoke test script for RA-TGR pipeline.

This script validates the complete RAG integration:
1. Environment setup (API key)
2. Codestral embeddings
3. LanceDB connection
4. Hybrid retrieval
5. RAG template selection
6. GoT retrieval nodes

Run: python scripts/smoke_test_rag.py

Prerequisites:
- OPENROUTER_API_KEY environment variable set
- RAG database indexed (run scripts/index_wikipedia.py first)
"""

from __future__ import annotations

import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_result(success: bool, message: str) -> None:
    """Print a test result."""
    status = "✓ OK" if success else "✗ FAIL"
    print(f"  {status}: {message}")


def test_environment() -> bool:
    """Test 1: Check environment setup."""
    print("\n[1/6] Checking environment...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print_result(False, "OPENROUTER_API_KEY not set")
        print("       Set it with: export OPENROUTER_API_KEY='your-key'")
        return False
    
    print_result(True, f"API key found (length: {len(api_key)})")
    return True


def test_embeddings() -> bool:
    """Test 2: Test Codestral embeddings."""
    print("\n[2/6] Testing Codestral embeddings...")
    
    try:
        from apps.mas.rag.embeddings import CodestralEmbedder
        
        start = time.time()
        embedder = CodestralEmbedder()
        vec = embedder.embed_query("What is a Cayley graph?")
        elapsed = time.time() - start
        
        if len(vec) != 1536:
            print_result(False, f"Wrong dimension: expected 1536, got {len(vec)}")
            return False
        
        print_result(True, f"Embedding dimension = {len(vec)} ({elapsed:.2f}s)")
        return True
        
    except Exception as e:
        print_result(False, f"Embedding failed: {str(e)[:80]}")
        return False


def test_lancedb_connection() -> bool:
    """Test 3: Test LanceDB connection."""
    print("\n[3/6] Testing LanceDB connection...")
    
    db_path = "apps/mas/data/wiki_lance"
    
    try:
        import lancedb
        
        if not os.path.exists(db_path):
            print_result(False, f"Database not found at {db_path}")
            print("       Run: python scripts/index_wikipedia.py --arrow-path wikipedia-subset-hf-dataset/wikipedia-subset/")
            return False
        
        db = lancedb.connect(db_path)
        tables = db.table_names()
        
        if "wiki_chunks" not in tables:
            print_result(False, "Table 'wiki_chunks' not found")
            return False
        
        table = db.open_table("wiki_chunks")
        row_count = table.count_rows()
        
        print_result(True, f"Connected - {row_count} chunks in database")
        return True
        
    except ImportError:
        print_result(False, "lancedb not installed - run: pip install lancedb")
        return False
    except Exception as e:
        print_result(False, f"Connection failed: {str(e)[:80]}")
        return False


def test_hybrid_retrieval() -> bool:
    """Test 4: Test hybrid retrieval."""
    print("\n[4/6] Testing hybrid retrieval...")
    
    try:
        from apps.mas.rag.embeddings import CodestralEmbedder
        from apps.mas.rag.retriever import HybridRetriever
        
        embedder = CodestralEmbedder()
        retriever = HybridRetriever(
            db_path="apps/mas/data/wiki_lance",
            embedder=embedder,
        )
        
        start = time.time()
        results = retriever.fusion_search("mathematics eigenvalue", k=3)
        elapsed = time.time() - start
        
        if not results:
            print_result(False, "No results returned")
            return False
        
        print_result(True, f"Retrieved {len(results)} documents ({elapsed:.2f}s)")
        
        # Show first result
        if results:
            first = results[0]
            title = first.title[:40] if hasattr(first, 'title') else "N/A"
            print(f"       Top result: '{title}...'")
        
        return True
        
    except Exception as e:
        print_result(False, f"Retrieval failed: {str(e)[:80]}")
        return False


def test_rag_template_selection() -> bool:
    """Test 5: Test RAG-enhanced template selection."""
    print("\n[5/6] Testing RAG template selection...")
    
    try:
        from apps.mas.rag.embeddings import CodestralEmbedder
        from apps.mas.rag.retriever import HybridRetriever
        from apps.mas.graph.template_distiller import RAGTemplateDistiller
        
        embedder = CodestralEmbedder()
        retriever = HybridRetriever(
            db_path="apps/mas/data/wiki_lance",
            embedder=embedder,
        )
        distiller = RAGTemplateDistiller(
            templates_dir="apps/mas/configs/templates",
            retriever=retriever,
        )
        
        start = time.time()
        template, score, context = distiller.select_with_rag("eigenvalue of Cayley graph")
        elapsed = time.time() - start
        
        if template:
            print_result(True, f"Selected '{template.template_id}' (score={score}) ({elapsed:.2f}s)")
            print(f"       Retrieved {len(context)} context snippets")
        else:
            print_result(False, "No template selected")
            return False
        
        return True
        
    except Exception as e:
        print_result(False, f"Template selection failed: {str(e)[:80]}")
        return False


def test_got_retrieval_node() -> bool:
    """Test 6: Test GoT retrieval node execution."""
    print("\n[6/6] Testing GoT retrieval node...")
    
    try:
        from apps.mas.rag.embeddings import CodestralEmbedder
        from apps.mas.rag.retriever import HybridRetriever
        from apps.mas.graph.got_controller import GoTController
        
        embedder = CodestralEmbedder()
        retriever = HybridRetriever(
            db_path="apps/mas/data/wiki_lance",
            embedder=embedder,
        )
        
        # Create minimal controller for testing
        controller = GoTController(
            problem="What are the eigenvalues of a Cayley graph?",
            template={
                "template_id": "test",
                "graph_blueprint": {"nodes": [], "edges": []}
            },
            swarm=None,
            researcher=None,
            verifier=None,
            retriever=retriever,
            knowledge_seeds=["Eigenvalues are computed using character sums"],
        )
        
        # Test retrieval
        start = time.time()
        result = controller._run_retrieval("Cayley graph eigenvalue", "")
        elapsed = time.time() - start
        
        if "[1]" not in result and "No retriever" in result:
            print_result(False, "Retrieval returned no documents")
            return False
        
        print_result(True, f"Retrieved documents via GoT node ({elapsed:.2f}s)")
        
        # Test seed augmentation
        augmented = controller._augment_seeds_with_rag()
        if len(augmented) > len(controller.knowledge_seeds):
            print(f"       Seeds augmented: {len(controller.knowledge_seeds)} → {len(augmented)}")
        
        return True
        
    except Exception as e:
        print_result(False, f"GoT retrieval failed: {str(e)[:80]}")
        return False


def main():
    """Run all smoke tests."""
    print_header("RA-TGR Smoke Test Suite")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {project_root}")
    
    start_time = time.time()
    
    # Run tests
    results = []
    
    # Test 1: Environment
    results.append(("Environment", test_environment()))
    
    if results[-1][1]:  # Only continue if env is OK
        # Test 2: Embeddings
        results.append(("Embeddings", test_embeddings()))
        
        # Test 3: LanceDB
        results.append(("LanceDB", test_lancedb_connection()))
        
        if results[-1][1]:  # Only continue if DB exists
            # Test 4: Hybrid Retrieval
            results.append(("Hybrid Retrieval", test_hybrid_retrieval()))
            
            # Test 5: RAG Template Selection
            results.append(("RAG Template Selection", test_rag_template_selection()))
            
            # Test 6: GoT Retrieval Node
            results.append(("GoT Retrieval Node", test_got_retrieval_node()))
    
    # Summary
    elapsed = time.time() - start_time
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print_header("Summary")
    print(f"  Tests run: {total}")
    print(f"  Passed:    {passed}")
    print(f"  Failed:    {total - passed}")
    print(f"  Time:      {elapsed:.2f}s")
    print()
    
    if passed == total:
        print("  🎉 ALL SMOKE TESTS PASSED!")
        print()
        print("  RA-TGR is ready to use. Next steps:")
        print("  - Run: python -m apps.mas.web.chat_ui --config apps/mas/configs/openrouter.yaml")
        print("  - Or use solve_with_budget() directly in Python")
        return 0
    else:
        print("  ⚠️  SOME TESTS FAILED")
        print()
        print("  Check the errors above and:")
        print("  1. Ensure OPENROUTER_API_KEY is set")
        print("  2. Run: python scripts/index_wikipedia.py --arrow-path wikipedia-subset-hf-dataset/wikipedia-subset/")
        print("  3. Install dependencies: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

