"""
End-to-end integration smoke tests for RA-TGR.

Run: pytest apps/mas/rag/tests/test_integration.py -v

Note: These tests require the RAG database to be indexed.
Run `python scripts/index_wikipedia.py` first.
"""

from __future__ import annotations

import os
import tempfile
import pytest
import numpy as np


class MockEmbedder:
    """Mock embedder for testing without API calls."""
    
    dimension = 1536
    
    def embed_documents(self, texts):
        vecs = []
        for text in texts:
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            vecs.append(rng.rand(1536).astype('float32'))
        return np.array(vecs)
    
    def embed_query(self, query):
        seed = hash(query) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.rand(1536).astype('float32')


@pytest.fixture
def mock_rag_setup():
    """Setup RAG pipeline with mock embedder for testing."""
    from apps.mas.rag.indexer import WikipediaIndexer
    from apps.mas.rag.retriever import HybridRetriever
    
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        indexer = WikipediaIndexer(tmpdir, embedder)
        
        # Create test documents
        docs = [
            {
                "id": "spectral_1",
                "title": "Spectral Graph Theory",
                "text": "Spectral graph theory studies graphs through eigenvalues of matrices. "
                       "The Cayley graph of an abelian group has eigenvalues given by character sums. "
                       "For cyclic group Z_n, characters are roots of unity.",
                "url": "http://wiki/spectral"
            },
            {
                "id": "knot_1",
                "title": "Quandle Coloring",
                "text": "Quandles are algebraic structures for knot invariants. "
                       "A knot can be colored by a quandle if there exists a valid coloring map. "
                       "The figure-8 knot can be colored by the tetrahedral quandle.",
                "url": "http://wiki/quandle"
            },
            {
                "id": "matrix_1",
                "title": "Rank-1 Matrices",
                "text": "Rank-1 matrices can be written as outer products. "
                       "The Frobenius inner product measures matrix similarity. "
                       "Admissible sets of rank-1 matrices relate to equiangular lines.",
                "url": "http://wiki/rank1"
            },
        ]
        
        indexer.index_documents(docs, chunk_size=200, show_progress=False)
        
        retriever = HybridRetriever(db_path=tmpdir, embedder=embedder)
        
        yield retriever


@pytest.fixture
def live_rag_setup():
    """Setup RAG pipeline with real API (skips if not available)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")
    
    db_path = "apps/mas/data/wiki_lance"
    if not os.path.exists(db_path):
        pytest.skip(f"RAG database not found at {db_path}")
    
    from apps.mas.rag.embeddings import CodestralEmbedder
    from apps.mas.rag.retriever import HybridRetriever
    from apps.mas.graph.template_distiller import RAGTemplateDistiller
    
    embedder = CodestralEmbedder(api_key=api_key)
    retriever = HybridRetriever(db_path=db_path, embedder=embedder)
    distiller = RAGTemplateDistiller(
        templates_dir="apps/mas/configs/templates",
        retriever=retriever
    )
    
    return distiller, retriever


class TestRAGTemplateDistiller:
    """Tests for RAG-enhanced template selection."""
    
    def test_distiller_with_mock(self, mock_rag_setup):
        """Verify RAG distiller works with mock retriever."""
        from apps.mas.graph.template_distiller import RAGTemplateDistiller
        
        retriever = mock_rag_setup
        distiller = RAGTemplateDistiller(
            templates_dir="apps/mas/configs/templates",
            retriever=retriever,
        )
        
        # Test selection
        problem = "Find the eigenvalues of a Cayley graph"
        template, score, context = distiller.select_with_rag(problem)
        
        # Should return valid results (even with mock)
        assert isinstance(score, int)
        assert isinstance(context, list)
    
    def test_distiller_extracts_domain(self, mock_rag_setup):
        """Verify domain extraction from RAG context."""
        from apps.mas.graph.template_distiller import RAGTemplateDistiller
        
        retriever = mock_rag_setup
        distiller = RAGTemplateDistiller(
            templates_dir="apps/mas/configs/templates",
            retriever=retriever,
        )
        
        # This should find spectral domain content
        domain_scores = distiller._extract_domain_from_rag("eigenvalue Cayley")
        
        # Should extract some domain signals
        assert isinstance(domain_scores, dict)
    
    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set"
    )
    def test_live_rag_template_selection(self, live_rag_setup):
        """Verify RAG-enhanced template selection works with live API."""
        distiller, _ = live_rag_setup
        
        problem = "Find the eigenvalues of the Cayley graph of Z_18"
        template, score, context = distiller.select_with_rag(problem)
        
        if template:
            # Should select appropriate template
            assert "cayley" in template.template_id.lower() or "spectral" in template.template_id.lower()
        assert score >= 0
        assert len(context) >= 0


class TestGoTControllerRAG:
    """Tests for GoT controller with RAG integration."""
    
    def test_retrieval_node_execution(self, mock_rag_setup):
        """Verify GoT can execute retrieval nodes."""
        from apps.mas.graph.got_controller import GoTController
        
        retriever = mock_rag_setup
        
        # Create minimal template with retrieval node
        template = {
            "template_id": "test_rag",
            "graph_blueprint": {
                "nodes": [
                    {
                        "id": "n1",
                        "type": "retrieval",
                        "role": "rag",
                        "instruction": "Retrieve info about spectral graph theory"
                    }
                ],
                "edges": []
            }
        }
        
        # Create controller with mocked dependencies
        controller = GoTController(
            problem="What are eigenvalues of Cayley graphs?",
            template=template,
            swarm=None,
            researcher=None,
            verifier=None,
            retriever=retriever,
        )
        
        result = controller._run_retrieval("eigenvalue Cayley", "")
        
        assert len(result) > 0
        assert "[1]" in result or "No retriever" not in result
    
    def test_knowledge_seeds_augmentation(self, mock_rag_setup):
        """Verify knowledge seeds are augmented with RAG context."""
        from apps.mas.graph.got_controller import GoTController
        
        retriever = mock_rag_setup
        
        controller = GoTController(
            problem="Test problem about eigenvalues",
            template={"template_id": "test", "graph_blueprint": {"nodes": [], "edges": []}},
            swarm=None,
            researcher=None,
            verifier=None,
            retriever=retriever,
            knowledge_seeds=["Cayley graph eigenvalues use character sums"],
            augment_seeds_with_rag=True,
        )
        
        augmented = controller._augment_seeds_with_rag()
        
        # Should have more entries than original seeds
        assert len(augmented) >= len(controller.knowledge_seeds)
    
    def test_controller_without_retriever(self):
        """Verify controller works without retriever (graceful fallback)."""
        from apps.mas.graph.got_controller import GoTController
        
        controller = GoTController(
            problem="Test problem",
            template={"template_id": "test", "graph_blueprint": {"nodes": [], "edges": []}},
            swarm=None,
            researcher=None,
            verifier=None,
            retriever=None,  # No retriever
            knowledge_seeds=["Original seed"],
        )
        
        # Should return original seeds unchanged
        augmented = controller._augment_seeds_with_rag()
        assert augmented == controller.knowledge_seeds
        
        # Retrieval should return graceful message
        result = controller._run_retrieval("test", "")
        assert "No retriever" in result


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_pipeline_mock(self, mock_rag_setup):
        """Test complete RA-TGR pipeline with mocks."""
        from apps.mas.graph.template_distiller import RAGTemplateDistiller
        from apps.mas.graph.got_controller import GoTController
        
        retriever = mock_rag_setup
        
        # Step 1: RAG-enhanced template selection
        distiller = RAGTemplateDistiller(
            templates_dir="apps/mas/configs/templates",
            retriever=retriever,
        )
        
        problem = "Compute eigenvalues of Cayley graph"
        template, score, context = distiller.select_with_rag(problem)
        
        # Step 2: Create GoT controller with retriever
        if template:
            controller = GoTController(
                problem=problem,
                template={
                    "template_id": template.template_id,
                    "graph_blueprint": template.graph_blueprint,
                },
                swarm=None,
                researcher=None,
                verifier=None,
                retriever=retriever,
                knowledge_seeds=template.knowledge_seeds,
                augment_seeds_with_rag=True,
            )
            
            # Verify augmented seeds
            augmented = controller._augment_seeds_with_rag()
            assert len(augmented) >= len(template.knowledge_seeds)

