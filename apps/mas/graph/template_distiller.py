from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .template_generator import TemplateGenerator, GeneratedTemplateCache


def _default_emit(stage: str, content: str) -> None:
    """No-op logger fallback."""
    return None


@dataclass
class TemplateSpec:
    template_id: str
    domain_tags: List[str]
    description: str
    knowledge_seeds: List[str]
    graph_blueprint: Dict[str, Any]
    path: str


class TemplateDistiller:
    """
    Lightweight semantic matcher that selects a TGR template
    based on keyword overlap with the incoming problem.
    """

    def __init__(
        self,
        templates_dir: str,
        thinking_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.templates_dir = templates_dir
        self._emit = thinking_callback or _default_emit
        self.templates: List[TemplateSpec] = self._load_templates()

    def _load_templates(self) -> List[TemplateSpec]:
        templates: List[TemplateSpec] = []
        pattern = os.path.join(self.templates_dir, "*.json")
        for path in glob.glob(pattern):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                spec = TemplateSpec(
                    template_id=raw.get("template_id", os.path.basename(path)),
                    domain_tags=[t.lower() for t in raw.get("domain_tags", [])],
                    description=raw.get("description", ""),
                    knowledge_seeds=raw.get("knowledge_seeds", []),
                    graph_blueprint=raw.get("graph_blueprint", {}),
                    path=path,
                )
                templates.append(spec)
            except Exception as e:
                self._emit("template_load_error", f"Failed to load {path}: {e}")
        self._emit("template_load_complete", f"Loaded {len(templates)} templates from {self.templates_dir}")
        return templates

    @staticmethod
    def _normalize(text: str) -> str:
        t = text.lower()
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _score(self, problem: str, template: TemplateSpec) -> int:
        """
        Simple heuristic scoring: keyword overlap plus domain cues.
        """
        p = self._normalize(problem)
        tokens = set(p.split())
        score = 0
        for tag in template.domain_tags:
            if tag in p:
                score += 3
            elif tag in tokens:
                score += 2
        # Additional cues for known archetypes
        spectral_cues = {"eigenvalue", "spectrum", "cayley", "abelian", "roots", "unity"}
        rank1_cues = {"rank-1", "rank1", "frobenius", "admissible", "matrix", "matrices", "inner", "product"}
        knot_cues = {"knot", "quandle", "figure", "coloring", "colorable"}
        hotel_cues = {"hotel", "guest", "light", "toggle", "cat"}
        subgroup_cues = {"free product", "subgroup", "index", "c2", "c_2", "c5", "c_5", "s7", "s_7"}
        e8_cues = {"artin", "e8", "torsion", "order 10", "order-10", "center", "a/z", "quotient"}
        if any(cue in p for cue in spectral_cues) and "cayley" in template.template_id:
            score += 2
        if any(cue in p for cue in rank1_cues) and "rank1" in template.template_id:
            score += 2
        if any(cue in p for cue in knot_cues) and "knot" in template.template_id:
            score += 2
        if any(cue in p for cue in hotel_cues) and "hotel" in template.template_id:
            score += 2
        if any(cue in p for cue in subgroup_cues) and "free_product" in template.template_id:
            score += 2
        if any(cue in p for cue in e8_cues) and "artin_e8_torsion" in template.template_id:
            score += 3
        return score

    def select(self, problem: str) -> Optional[TemplateSpec]:
        if not self.templates:
            self._emit("template_missing", "No templates loaded")
            return None
        best: Optional[TemplateSpec] = None
        best_score = 0
        for tpl in self.templates:
            s = self._score(problem, tpl)
            if s > best_score:
                best = tpl
                best_score = s
        if best is None or best_score <= 0:
            self._emit("template_no_match", "No template matched the problem")
            return None
        self._emit("template_selected", f"Selected template {best.template_id} (score={best_score})")
        return best

    def select_with_score(self, problem: str) -> Tuple[Optional[TemplateSpec], int]:
        tpl = self.select(problem)
        if tpl is None:
            return None, 0
        score = self._score(problem, tpl)
        return tpl, score


class RAGTemplateDistiller(TemplateDistiller):
    """
    Enhanced distiller that uses RAG to:
    1. Find similar problems/content in knowledge base
    2. Extract domain signals from retrieved context
    3. Score templates based on semantic similarity
    
    This extends the keyword-based TemplateDistiller with retrieval-augmented
    template selection for improved accuracy on edge cases.
    """
    
    # Minimum score to select a template (avoid false positives on factual questions)
    MIN_TEMPLATE_SCORE = 5
    
    # Domain keywords to look for in retrieved content
    DOMAIN_KEYWORDS = {
        "spectral": ["eigenvalue", "spectrum", "cayley", "character", "representation"],
        "matrix": ["rank", "frobenius", "admissible", "inner product", "trace"],
        "knot": ["knot", "quandle", "coloring", "figure-8", "trefoil"],
        "group": ["subgroup", "index", "free product", "cyclic", "abelian"],
        "topology": ["topology", "manifold", "homology", "homotopy"],
    }
    
    # Keywords that indicate a factual/QA question (NOT a math problem)
    FACTUAL_SIGNALS = [
        # Question words for factual queries
        "which", "who", "what", "when", "where", "how many", "how much",
        # Named entities / historical events
        "airlines", "flight", "mh17", "mh-17", "malaysia", "ukraine", "russia",
        "missile", "buk", "war", "conflict", "attack", "crash", "downing",
        "president", "prime minister", "country", "nation", "government",
        "company", "corporation", "organization", "team", "player",
        "movie", "film", "actor", "actress", "singer", "album", "song",
        "book", "author", "wrote", "published", "founded", "invented",
        "born", "died", "married", "elected", "appointed", "awarded",
        "capital", "population", "located", "headquarters",
    ]
    
    # Keywords that indicate a math/logic problem
    MATH_SIGNALS = [
        "prove", "proof", "theorem", "lemma", "corollary", "axiom",
        "calculate", "compute", "evaluate", "solve", "simplify",
        "equation", "inequality", "formula", "expression",
        "derivative", "integral", "limit", "sum", "product",
        "matrix", "vector", "eigenvalue", "determinant", "rank",
        "group", "ring", "field", "module", "algebra",
        "topology", "manifold", "homology", "homotopy",
        "probability", "expectation", "variance", "distribution",
        "n choose k", "factorial", "binomial", "permutation",
        "graph", "vertex", "edge", "cycle", "path", "tree",
        "knot", "quandle", "coloring", "figure-8", "trefoil",
    ]
    
    @staticmethod
    def _is_factual_question(problem: str) -> bool:
        """
        Detect if the question is a factual/QA question that should NOT
        be routed through math templates.
        """
        p = problem.lower()
        
        # Count factual and math signals
        factual_count = sum(1 for s in RAGTemplateDistiller.FACTUAL_SIGNALS if s in p)
        math_count = sum(1 for s in RAGTemplateDistiller.MATH_SIGNALS if s in p)
        
        # If more factual signals than math signals, it's a factual question
        if factual_count > math_count and factual_count >= 2:
            return True
        
        # Strong factual signals (named entities, events)
        strong_factual = ["airlines", "flight", "mh17", "malaysia", "missile", "buk",
                         "president", "prime minister", "war", "conflict", "crash"]
        if any(s in p for s in strong_factual):
            return True
        
        return False
    
    def __init__(
        self,
        templates_dir: str,
        retriever: Optional[Any] = None,  # HybridRetriever
        thinking_callback: Optional[Callable[[str, str], None]] = None,
        rag_top_k: int = 5,
        rag_boost_weight: float = 1.0,
        template_generator: Optional["TemplateGenerator"] = None,
        generated_cache: Optional["GeneratedTemplateCache"] = None,
        enable_dynamic_generation: bool = False,
    ):
        """
        Initialize the RAG-enhanced template distiller.
        
        Args:
            templates_dir: Path to templates directory
            retriever: HybridRetriever instance for RAG
            thinking_callback: Optional callback for logging
            rag_top_k: Number of documents to retrieve
            rag_boost_weight: Weight for RAG-based score boost
            template_generator: Optional TemplateGenerator for dynamic creation
            generated_cache: Optional cache for generated templates
            enable_dynamic_generation: Whether to generate templates when none match
        """
        super().__init__(templates_dir, thinking_callback)
        self.retriever = retriever
        self.rag_top_k = rag_top_k
        self.rag_boost_weight = rag_boost_weight
        self.template_generator = template_generator
        self.generated_cache = generated_cache
        self.enable_dynamic_generation = enable_dynamic_generation
        
        # Load any cached generated templates
        if self.generated_cache:
            for cached_tpl in self.generated_cache.list_all():
                if cached_tpl.template_id not in {t.template_id for t in self.templates}:
                    self.templates.append(cached_tpl)
                    self._emit("template_loaded_from_cache", f"Loaded cached template: {cached_tpl.template_id}")
    
    def _extract_domain_from_rag(self, problem: str) -> Dict[str, float]:
        """
        Retrieve relevant docs and extract domain indicators.
        
        Args:
            problem: The problem text
        
        Returns:
            Dictionary mapping domain names to relevance scores
        """
        if not self.retriever:
            return {}
        
        try:
            chunks = self.retriever.fusion_search(problem, k=self.rag_top_k)
        except Exception as e:
            self._emit("rag_error", f"RAG retrieval failed: {e}")
            return {}
        
        if not chunks:
            return {}
        
        # Analyze retrieved content for domain keywords
        domain_scores: Dict[str, float] = {}
        
        for chunk in chunks:
            chunk_text = chunk.text.lower() if hasattr(chunk, 'text') else str(chunk).lower()
            chunk_score = chunk.score if hasattr(chunk, 'score') else 1.0
            
            for domain, keywords in self.DOMAIN_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in chunk_text:
                        # Weight by chunk relevance score
                        domain_scores[domain] = domain_scores.get(domain, 0) + chunk_score
        
        return domain_scores
    
    def _get_rag_context(self, problem: str) -> List[str]:
        """
        Retrieve context snippets from RAG.
        
        Args:
            problem: The problem text
        
        Returns:
            List of context snippet strings
        """
        if not self.retriever:
            return []
        
        try:
            chunks = self.retriever.fusion_search(problem, k=self.rag_top_k)
            return [
                chunk.text[:300] if hasattr(chunk, 'text') else str(chunk)[:300]
                for chunk in chunks
            ]
        except Exception as e:
            self._emit("rag_context_error", f"Failed to get RAG context: {e}")
            return []
    
    def _compute_rag_boost(
        self,
        template: TemplateSpec,
        domain_scores: Dict[str, float],
    ) -> float:
        """
        Compute RAG-based score boost for a template.
        
        Args:
            template: The template to score
            domain_scores: Domain relevance scores from RAG
        
        Returns:
            Boost value to add to base score
        """
        boost = 0.0
        template_id_lower = template.template_id.lower()
        
        # Map template IDs to domains
        template_domains = {
            "spectral": ["cayley", "spectral", "eigenvalue"],
            "matrix": ["rank1", "matrix", "frobenius"],
            "knot": ["knot", "quandle", "figure8"],
            "group": ["subgroup", "free_product", "group"],
            "topology": ["topology", "e8", "artin"],
        }
        
        for domain, score in domain_scores.items():
            if domain in template_domains:
                # Check if template matches this domain
                for pattern in template_domains[domain]:
                    if pattern in template_id_lower:
                        boost += score * self.rag_boost_weight
                        break
        
        # Also boost based on domain tag matches
        for tag in template.domain_tags:
            for domain, score in domain_scores.items():
                if domain in tag or tag in domain:
                    boost += score * 0.5 * self.rag_boost_weight
        
        return boost
    
    def select_with_rag(
        self,
        problem: str,
        allow_generation: Optional[bool] = None,
    ) -> Tuple[Optional[TemplateSpec], int, List[str]]:
        """
        Select template using RAG-augmented scoring.
        
        If no template matches and dynamic generation is enabled, will attempt
        to generate a new template using LLM.
        
        Args:
            problem: The problem text
            allow_generation: Override for enable_dynamic_generation
        
        Returns:
            Tuple of (template, combined_score, context_snippets)
        """
        self._emit("rag_distiller_start", f"RAG-enhanced template selection for: {problem[:100]}...")
        
        # 0. Check if this is a factual/QA question (not a math problem)
        if self._is_factual_question(problem):
            self._emit("rag_factual_detected", "Detected factual/QA question - skipping math templates")
            context_snippets = self._get_rag_context(problem)
            return None, 0, context_snippets
        
        # 1. Get RAG context
        context_snippets = self._get_rag_context(problem)
        self._emit("rag_context_retrieved", f"Retrieved {len(context_snippets)} context snippets")
        
        # 2. Extract domain signals
        domain_scores = self._extract_domain_from_rag(problem)
        if domain_scores:
            self._emit("rag_domains", f"Domain signals: {domain_scores}")
        
        # 3. Score all templates with RAG boost
        best_tpl: Optional[TemplateSpec] = None
        best_score = 0
        
        for tpl in self.templates:
            # Base keyword score
            base_score = self._score(problem, tpl)
            
            # RAG boost
            rag_boost = self._compute_rag_boost(tpl, domain_scores)
            
            combined_score = base_score + int(rag_boost)
            
            if combined_score > best_score:
                best_tpl = tpl
                best_score = combined_score
        
        # 4. Apply minimum score threshold to avoid false positives
        if best_score < self.MIN_TEMPLATE_SCORE:
            self._emit(
                "rag_below_threshold",
                f"Best template {best_tpl.template_id if best_tpl else 'none'} score={best_score} < min={self.MIN_TEMPLATE_SCORE}"
            )
            
            # 5. Try dynamic template generation if enabled
            should_generate = allow_generation if allow_generation is not None else self.enable_dynamic_generation
            if should_generate and self.template_generator:
                generated = self._try_generate_template(problem, context_snippets)
                if generated:
                    return generated, self.MIN_TEMPLATE_SCORE, context_snippets
            
            return None, 0, context_snippets
        
        if best_tpl:
            self._emit(
                "rag_template_selected",
                f"Selected template {best_tpl.template_id} (score={best_score})"
            )
        else:
            self._emit("rag_no_match", "No template matched with RAG augmentation")
        
        return best_tpl, best_score, context_snippets
    
    def _try_generate_template(
        self,
        problem: str,
        rag_context: List[str],
    ) -> Optional[TemplateSpec]:
        """
        Attempt to generate a template for the problem.
        
        Args:
            problem: The problem text
            rag_context: RAG context snippets
        
        Returns:
            Generated TemplateSpec if successful, None otherwise
        """
        if not self.template_generator:
            return None
        
        self._emit("template_gen_attempt", f"Attempting dynamic template generation for: {problem[:80]}...")
        
        try:
            result = self.template_generator.generate(problem, rag_context)
            
            if result.success and result.template:
                # Add to templates list for future matching
                self.templates.append(result.template)
                self._emit(
                    "template_gen_success",
                    f"Generated template {result.template.template_id} (attempts={result.attempts})"
                )
                return result.template
            else:
                self._emit(
                    "template_gen_failed",
                    f"Generation failed: {result.error or 'unknown error'}"
                )
        except Exception as e:
            self._emit("template_gen_error", f"Generation error: {str(e)[:150]}")
        
        return None
    
    def select_with_score(self, problem: str) -> Tuple[Optional[TemplateSpec], int]:
        """
        Override to use RAG-enhanced selection.
        
        Falls back to base method if retriever is not available.
        """
        if self.retriever:
            tpl, score, _ = self.select_with_rag(problem)
            return tpl, score
        else:
            return super().select_with_score(problem)
