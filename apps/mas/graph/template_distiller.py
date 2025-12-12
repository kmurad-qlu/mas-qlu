from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


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
    
    # Domain keywords to look for in retrieved content
    DOMAIN_KEYWORDS = {
        "spectral": ["eigenvalue", "spectrum", "cayley", "character", "representation"],
        "matrix": ["rank", "frobenius", "admissible", "inner product", "trace"],
        "knot": ["knot", "quandle", "coloring", "figure-8", "trefoil"],
        "group": ["subgroup", "index", "free product", "cyclic", "abelian"],
        "topology": ["topology", "manifold", "homology", "homotopy"],
    }
    
    def __init__(
        self,
        templates_dir: str,
        retriever: Optional[Any] = None,  # HybridRetriever
        thinking_callback: Optional[Callable[[str, str], None]] = None,
        rag_top_k: int = 5,
        rag_boost_weight: float = 1.0,
    ):
        """
        Initialize the RAG-enhanced template distiller.
        
        Args:
            templates_dir: Path to templates directory
            retriever: HybridRetriever instance for RAG
            thinking_callback: Optional callback for logging
            rag_top_k: Number of documents to retrieve
            rag_boost_weight: Weight for RAG-based score boost
        """
        super().__init__(templates_dir, thinking_callback)
        self.retriever = retriever
        self.rag_top_k = rag_top_k
        self.rag_boost_weight = rag_boost_weight
    
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
    ) -> Tuple[Optional[TemplateSpec], int, List[str]]:
        """
        Select template using RAG-augmented scoring.
        
        Args:
            problem: The problem text
        
        Returns:
            Tuple of (template, combined_score, context_snippets)
        """
        self._emit("rag_distiller_start", f"RAG-enhanced template selection for: {problem[:100]}...")
        
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
        
        if best_tpl:
            self._emit(
                "rag_template_selected",
                f"Selected template {best_tpl.template_id} (score={best_score})"
            )
        else:
            self._emit("rag_no_match", "No template matched with RAG augmentation")
        
        return best_tpl, best_score, context_snippets
    
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
