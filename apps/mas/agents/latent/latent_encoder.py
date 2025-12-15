"""
Latent Encoder for generating embeddings and attention maps.

Provides utilities for encoding agent outputs into dense vector
representations and extracting attention patterns.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, TYPE_CHECKING
import re
import hashlib
from collections import Counter

from .latent_state import LatentState, AttentionMap, ReasoningStep, EmbeddingType

if TYPE_CHECKING:
    # Avoid circular imports - embedder is optional
    pass


class LatentEncoder:
    """
    Generates latent representations from agent outputs.
    
    Can use external embedders (like Codestral) when available,
    or fall back to simple TF-IDF-like embeddings.
    """
    
    # Uncertainty markers that reduce confidence
    UNCERTAINTY_MARKERS = [
        "i'm not sure", "possibly", "might be", "uncertain",
        "approximately", "roughly", "i think", "maybe",
        "could be", "perhaps", "not certain", "unclear",
        "appears to", "seems to", "may have", "likely",
    ]
    
    # Confidence markers that increase confidence
    CONFIDENCE_MARKERS = [
        "definitely", "certainly", "clearly", "obviously",
        "the answer is", "therefore", "thus", "hence",
        "confirmed", "verified", "established", "proven",
    ]
    
    def __init__(
        self,
        embedder: Optional[Any] = None,
        embedding_dim: int = 768,
        use_cache: bool = True,
    ):
        """
        Initialize the latent encoder.
        
        Args:
            embedder: Optional external embedder (e.g., Codestral embedder)
            embedding_dim: Dimension for fallback embeddings
            use_cache: Whether to cache embeddings
        """
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        self.use_cache = use_cache
        self._cache: Dict[str, EmbeddingType] = {}
    
    def encode_text(self, text: str) -> EmbeddingType:
        """
        Generate embedding for text.
        
        Uses external embedder if available, otherwise falls back
        to a simple hash-based embedding.
        """
        if not text:
            return [0.0] * self.embedding_dim
        
        # Check cache
        cache_key = self._get_cache_key(text)
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try external embedder first
        if self.embedder is not None:
            try:
                embedding = self._encode_with_embedder(text)
                if embedding:
                    if self.use_cache:
                        self._cache[cache_key] = embedding
                    return embedding
            except Exception:
                pass  # Fall back to simple embedding
        
        # Fallback: simple hash-based embedding
        embedding = self._simple_embed(text)
        
        if self.use_cache:
            self._cache[cache_key] = embedding
        
        return embedding
    
    def _encode_with_embedder(self, text: str) -> Optional[EmbeddingType]:
        """Use external embedder to generate embedding."""
        if self.embedder is None:
            return None
        
        # Handle different embedder interfaces
        if hasattr(self.embedder, 'embed'):
            result = self.embedder.embed(text)
            if isinstance(result, list):
                return result
            # Convert numpy array to list
            return list(result)
        elif hasattr(self.embedder, 'encode'):
            result = self.embedder.encode(text)
            if isinstance(result, list):
                return result
            return list(result)
        
        return None
    
    def _simple_embed(self, text: str) -> EmbeddingType:
        """
        Generate a simple hash-based embedding.
        
        This is a fallback when no external embedder is available.
        Not as good as neural embeddings but provides some signal.
        """
        # Normalize text
        text_lower = text.lower()
        words = re.findall(r'\w+', text_lower)
        
        # Initialize embedding
        embedding = [0.0] * self.embedding_dim
        
        if not words:
            return embedding
        
        # Hash each word and add to embedding
        for word in words:
            # Use MD5 hash to get deterministic values
            word_hash = hashlib.md5(word.encode()).hexdigest()
            for i, char in enumerate(word_hash):
                idx = (i * 16 + int(char, 16)) % self.embedding_dim
                embedding[idx] += 1.0 / len(words)
        
        # Normalize
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Use first 500 chars to limit memory
        return hashlib.md5(text[:500].encode()).hexdigest()
    
    def estimate_confidence(self, text: str) -> float:
        """
        Estimate confidence from response text.
        
        Uses heuristics based on uncertainty/confidence markers.
        """
        if not text:
            return 0.5
        
        text_lower = text.lower()
        confidence = 0.7  # Base confidence
        
        # Check uncertainty markers
        for marker in self.UNCERTAINTY_MARKERS:
            if marker in text_lower:
                confidence -= 0.05
        
        # Check confidence markers
        for marker in self.CONFIDENCE_MARKERS:
            if marker in text_lower:
                confidence += 0.05
        
        # Clamp to [0.1, 1.0]
        return max(0.1, min(1.0, confidence))
    
    def extract_uncertainty_regions(self, text: str) -> List[str]:
        """
        Extract parts of the text that express uncertainty.
        """
        regions = []
        text_lower = text.lower()
        
        for marker in self.UNCERTAINTY_MARKERS:
            if marker in text_lower:
                # Find the sentence containing the marker
                sentences = re.split(r'[.!?]', text)
                for sentence in sentences:
                    if marker in sentence.lower():
                        # Extract a short snippet
                        snippet = sentence.strip()[:100]
                        if snippet and snippet not in regions:
                            regions.append(snippet)
        
        return regions[:5]  # Limit to 5 regions
    
    def extract_attention(self, text: str, context: str = "") -> AttentionMap:
        """
        Extract attention weights showing which parts of context matter.
        
        Uses keyword overlap and entity extraction to determine focus.
        """
        attention = AttentionMap()
        
        # Extract named entities (simple capitalized word sequences)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entity_counts = Counter(entities)
        
        # Calculate weights based on frequency
        max_count = max(entity_counts.values()) if entity_counts else 1
        for entity, count in entity_counts.items():
            weight = min(1.0, count / max_count * 0.8 + 0.2)
            attention.entity_weights[entity] = weight
        
        # Extract key phrases (quoted text, emphasized terms)
        quoted = re.findall(r'"([^"]+)"', text)
        attention.key_phrases = list(set(quoted))[:10]
        
        # If context provided, compute chunk relevance
        if context:
            # Split context into chunks and compute overlap
            context_chunks = context.split('\n\n')
            text_words = set(re.findall(r'\w+', text.lower()))
            
            for i, chunk in enumerate(context_chunks[:10]):
                chunk_words = set(re.findall(r'\w+', chunk.lower()))
                if chunk_words:
                    overlap = len(text_words & chunk_words) / len(chunk_words)
                    if overlap > 0.1:  # Minimum threshold
                        attention.chunk_weights[f"chunk_{i}"] = min(1.0, overlap)
        
        return attention
    
    def extract_reasoning_steps(self, text: str) -> List[ReasoningStep]:
        """
        Extract reasoning steps from structured output.
        
        Looks for numbered steps, bullet points, or explicit markers.
        """
        steps = []
        
        # Pattern 1: Numbered steps (1. 2. 3.)
        numbered = re.findall(r'(\d+)\.\s*([^.]+\.)', text)
        for num, content in numbered[:10]:
            step_type = self._classify_step(content)
            steps.append(ReasoningStep(
                step_id=f"step_{num}",
                step_type=step_type,
                input_summary="",
                conclusion=content.strip()[:200],
                confidence=self.estimate_confidence(content),
            ))
        
        # Pattern 2: Explicit markers
        markers = {
            "definition": [r'define[ds]?\s+as', r'means\s+that'],
            "calculation": [r'calculat(?:e|ing|ed)', r'comput(?:e|ing|ed)', r'=\s*\d'],
            "inference": [r'therefore', r'thus', r'hence', r'we can conclude'],
            "verification": [r'verify', r'check', r'confirm', r'validated'],
        }
        
        for step_type, patterns in markers.items():
            for pattern in patterns:
                matches = re.findall(rf'([^.]*{pattern}[^.]*\.)', text, re.IGNORECASE)
                for match in matches[:3]:
                    step_id = f"{step_type}_{len(steps)}"
                    # Avoid duplicates
                    if not any(s.conclusion == match.strip()[:200] for s in steps):
                        steps.append(ReasoningStep(
                            step_id=step_id,
                            step_type=step_type,
                            input_summary="",
                            conclusion=match.strip()[:200],
                            confidence=self.estimate_confidence(match),
                        ))
        
        return steps[:10]  # Limit to 10 steps
    
    def _classify_step(self, content: str) -> str:
        """Classify a reasoning step by its content."""
        content_lower = content.lower()
        
        if any(kw in content_lower for kw in ['define', 'definition', 'means']):
            return "definition"
        elif any(kw in content_lower for kw in ['calculate', 'compute', '=', 'equals']):
            return "calculation"
        elif any(kw in content_lower for kw in ['verify', 'check', 'confirm']):
            return "verification"
        elif any(kw in content_lower for kw in ['retrieve', 'search', 'found']):
            return "retrieval"
        else:
            return "inference"
    
    def create_latent_state(
        self,
        agent_id: str,
        task_id: str,
        text_output: str,
        context: str = "",
        model_name: str = "",
        latency_ms: float = 0.0,
        include_embedding: bool = True,
        include_attention: bool = True,
        include_reasoning: bool = False,
    ) -> LatentState:
        """
        Create a complete latent state from agent output.
        
        This is the main entry point for encoding agent outputs.
        """
        import time
        
        embedding = None
        if include_embedding:
            embedding = self.encode_text(text_output)
        
        attention = None
        if include_attention:
            attention = self.extract_attention(text_output, context)
        
        reasoning_steps = []
        if include_reasoning:
            reasoning_steps = self.extract_reasoning_steps(text_output)
        
        return LatentState(
            agent_id=agent_id,
            task_id=task_id,
            text_output=text_output,
            embedding=embedding,
            confidence=self.estimate_confidence(text_output),
            uncertainty_regions=self.extract_uncertainty_regions(text_output),
            attention=attention,
            reasoning_steps=reasoning_steps,
            model_name=model_name,
            latency_ms=latency_ms,
            timestamp=time.time(),
        )
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

