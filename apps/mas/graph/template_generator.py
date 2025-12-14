"""
Dynamic template generation for TGR.

Uses LLM to generate custom reasoning templates when existing
templates are inadequate for a given problem.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from .template_distiller import TemplateSpec

if TYPE_CHECKING:
    from ..infra.openrouter.client import OpenRouterClient


# System prompt for template generation
TEMPLATE_GEN_SYSTEM = """You are a reasoning template architect. Your job is to design structured reasoning graphs for solving complex problems.

A reasoning template consists of:
1. **domain_tags**: Keywords that identify when this template applies
2. **knowledge_seeds**: Key facts/principles needed to solve this type of problem
3. **graph_blueprint**: A DAG of reasoning nodes

Node types:
- **definition**: Establish concepts, notation, constraints (role: logic)
- **enumeration**: List cases, elements, possibilities (role: logic)
- **calculation**: Compute values, run code (role: research)
- **aggregation**: Synthesize results from multiple nodes (role: logic)
- **verification**: Validate the final answer (role: verifier)
- **retrieval**: Fetch relevant documents (role: rag)

Output strict JSON only. No markdown, no explanation."""

TEMPLATE_GEN_PROMPT = """Generate a reasoning template for this problem:

PROBLEM:
{problem}

CONTEXT FROM KNOWLEDGE BASE:
{rag_context}

Requirements:
1. Create 3-6 nodes that break down the reasoning steps
2. Connect nodes with edges (source → target)
3. Include at least one definition node, one calculation/enumeration node, and one verification node
4. Provide 2-4 knowledge seeds specific to this problem domain
5. Choose domain_tags that would match similar problems

Output JSON with this exact schema:
{{
  "template_id": "generated_{hash}",
  "domain_tags": ["tag1", "tag2", ...],
  "description": "Brief description of what problems this template solves",
  "knowledge_seeds": [
    "Key principle or formula 1",
    "Key principle or formula 2"
  ],
  "graph_blueprint": {{
    "entrypoint": "node_id",
    "nodes": [
      {{
        "id": "n1_define",
        "type": "definition",
        "role": "logic",
        "instruction": "Define the problem setup and key constraints..."
      }},
      {{
        "id": "n2_calculate",
        "type": "calculation",
        "role": "research",
        "instruction": "Compute the required values using Python..."
      }},
      {{
        "id": "n3_verify",
        "type": "verification",
        "role": "verifier",
        "instruction": "Verify the result and output final answer as #### number"
      }}
    ],
    "edges": [
      {{"source": "n1_define", "target": "n2_calculate"}},
      {{"source": "n2_calculate", "target": "n3_verify"}}
    ]
  }}
}}"""


def _default_emit(stage: str, content: str) -> None:
    """Default no-op emit function."""
    pass


@dataclass
class GenerationResult:
    """Result of template generation attempt."""
    success: bool
    template: Optional[TemplateSpec]
    error: Optional[str] = None
    attempts: int = 1


class TemplateValidator:
    """
    Validates generated templates for correctness and executability.
    """
    
    VALID_NODE_TYPES = {"definition", "enumeration", "calculation", "aggregation", "verification", "retrieval"}
    VALID_ROLES = {"logic", "math", "research", "verifier", "rag", "qa"}
    MIN_NODES = 2
    MAX_NODES = 10
    
    def validate(self, template: TemplateSpec) -> Tuple[bool, List[str]]:
        """
        Validate a template for correctness.
        
        Args:
            template: The template to validate
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors: List[str] = []
        
        # Check required fields
        if not template.template_id:
            errors.append("Missing template_id")
        if not template.domain_tags:
            errors.append("Missing domain_tags")
        if not template.knowledge_seeds:
            errors.append("Missing knowledge_seeds")
        
        blueprint = template.graph_blueprint
        if not blueprint:
            errors.append("Missing graph_blueprint")
            return False, errors
        
        nodes = blueprint.get("nodes", [])
        edges = blueprint.get("edges", [])
        
        # Check node count
        if len(nodes) < self.MIN_NODES:
            errors.append(f"Too few nodes ({len(nodes)} < {self.MIN_NODES})")
        if len(nodes) > self.MAX_NODES:
            errors.append(f"Too many nodes ({len(nodes)} > {self.MAX_NODES})")
        
        # Build node ID set
        node_ids = set()
        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                errors.append("Node missing 'id' field")
                continue
            if node_id in node_ids:
                errors.append(f"Duplicate node id: {node_id}")
            node_ids.add(node_id)
            
            # Check node type
            node_type = node.get("type", "logic")
            if node_type not in self.VALID_NODE_TYPES:
                errors.append(f"Invalid node type '{node_type}' for node {node_id}")
            
            # Check role
            role = node.get("role", "logic")
            if role not in self.VALID_ROLES:
                errors.append(f"Invalid role '{role}' for node {node_id}")
            
            # Check instruction
            if not node.get("instruction"):
                errors.append(f"Node {node_id} missing instruction")
        
        # Check edges reference valid nodes
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source not in node_ids:
                errors.append(f"Edge references unknown source: {source}")
            if target not in node_ids:
                errors.append(f"Edge references unknown target: {target}")
            if source == target:
                errors.append(f"Self-loop detected: {source}")
        
        # Check for cycles (topological sort)
        if not errors:
            has_cycle = self._detect_cycle(node_ids, edges)
            if has_cycle:
                errors.append("Graph contains a cycle")
        
        # Check entrypoint
        entrypoint = blueprint.get("entrypoint")
        if entrypoint and entrypoint not in node_ids:
            errors.append(f"Entrypoint '{entrypoint}' not in nodes")
        
        # Check for terminal nodes (nodes with no outgoing edges)
        if not errors:
            sources = {e.get("source") for e in edges}
            terminals = node_ids - sources
            if not terminals:
                errors.append("No terminal nodes found")
        
        return len(errors) == 0, errors
    
    def _detect_cycle(self, node_ids: set, edges: List[Dict[str, str]]) -> bool:
        """Detect if the graph has a cycle using DFS."""
        adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        for edge in edges:
            src, tgt = edge.get("source"), edge.get("target")
            if src and tgt:
                adj[src].append(tgt)
        
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in node_ids}
        
        def dfs(node: str) -> bool:
            color[node] = GRAY
            for neighbor in adj.get(node, []):
                if color.get(neighbor) == GRAY:
                    return True  # Back edge = cycle
                if color.get(neighbor) == WHITE:
                    if dfs(neighbor):
                        return True
            color[node] = BLACK
            return False
        
        for nid in node_ids:
            if color[nid] == WHITE:
                if dfs(nid):
                    return True
        return False


class GeneratedTemplateCache:
    """
    Cache for successfully generated templates.
    
    Persists templates to disk for reuse.
    """
    
    DEFAULT_PATH = "apps/mas/data/generated_templates"
    
    def __init__(self, cache_path: Optional[str] = None):
        self.cache_path = Path(cache_path or self.DEFAULT_PATH)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, TemplateSpec] = {}
        self._load_from_disk()
    
    def _load_from_disk(self) -> None:
        """Load cached templates from disk."""
        for path in self.cache_path.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                spec = TemplateSpec(
                    template_id=data.get("template_id", path.stem),
                    domain_tags=[t.lower() for t in data.get("domain_tags", [])],
                    description=data.get("description", ""),
                    knowledge_seeds=data.get("knowledge_seeds", []),
                    graph_blueprint=data.get("graph_blueprint", {}),
                    path=str(path),
                )
                self._memory_cache[spec.template_id] = spec
            except Exception:
                continue
    
    def get(self, template_id: str) -> Optional[TemplateSpec]:
        """Get a template by ID."""
        return self._memory_cache.get(template_id)
    
    def save(self, template: TemplateSpec) -> None:
        """Save a template to cache."""
        self._memory_cache[template.template_id] = template
        
        # Persist to disk
        file_path = self.cache_path / f"{template.template_id}.json"
        data = {
            "template_id": template.template_id,
            "domain_tags": template.domain_tags,
            "description": template.description,
            "knowledge_seeds": template.knowledge_seeds,
            "graph_blueprint": template.graph_blueprint,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        template.path = str(file_path)
    
    def find_by_tags(self, tags: List[str]) -> List[TemplateSpec]:
        """Find templates matching any of the given tags."""
        tags_lower = {t.lower() for t in tags}
        matches = []
        for template in self._memory_cache.values():
            template_tags = {t.lower() for t in template.domain_tags}
            if tags_lower & template_tags:
                matches.append(template)
        return matches
    
    def list_all(self) -> List[TemplateSpec]:
        """List all cached templates."""
        return list(self._memory_cache.values())
    
    def clear(self) -> None:
        """Clear all cached templates."""
        for path in self.cache_path.glob("*.json"):
            try:
                os.remove(path)
            except OSError:
                pass
        self._memory_cache.clear()


class TemplateGenerator:
    """
    Generates reasoning templates using LLM.
    
    When no existing template matches a problem, this class uses an LLM
    to generate a custom template based on the problem structure.
    """
    
    def __init__(
        self,
        client: "OpenRouterClient",
        model_name: str = "mistralai/mistral-large-2512",
        thinking_callback: Optional[Callable[[str, str], None]] = None,
        cache: Optional[GeneratedTemplateCache] = None,
        max_attempts: int = 2,
    ):
        """
        Initialize the template generator.
        
        Args:
            client: LLM client for generation
            model_name: Model to use for generation
            thinking_callback: Callback for logging
            cache: Optional cache for storing successful templates
            max_attempts: Maximum generation attempts before giving up
        """
        self.client = client
        self.model_name = model_name
        self._emit = thinking_callback or _default_emit
        self.cache = cache or GeneratedTemplateCache()
        self.max_attempts = max_attempts
        self.validator = TemplateValidator()
    
    def _generate_template_id(self, problem: str) -> str:
        """Generate a unique template ID based on problem hash."""
        problem_hash = hashlib.sha256(problem.encode()).hexdigest()[:8]
        return f"generated_{problem_hash}"
    
    def generate(
        self,
        problem: str,
        rag_context: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Generate a template for the given problem.
        
        Args:
            problem: The problem to generate a template for
            rag_context: Optional RAG context snippets
        
        Returns:
            GenerationResult with the template or error
        """
        self._emit("template_gen_start", f"Generating template for: {problem[:100]}...")
        
        # Format RAG context
        context_str = ""
        if rag_context:
            context_str = "\n".join([f"- {c[:300]}" for c in rag_context[:5]])
        else:
            context_str = "(No additional context available)"
        
        template_id = self._generate_template_id(problem)
        
        for attempt in range(1, self.max_attempts + 1):
            self._emit("template_gen_attempt", f"Attempt {attempt}/{self.max_attempts}")
            
            # Build the prompt
            prompt = TEMPLATE_GEN_PROMPT.format(
                problem=problem[:1000],
                rag_context=context_str,
                hash=template_id.split("_")[1],
            )
            
            try:
                response = self.client.complete_chat(
                    messages=[
                        {"role": "system", "content": TEMPLATE_GEN_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model_name,
                    temperature=0.3,
                    max_tokens=2000,
                )
                
                raw_text = response.text.strip()
                
                # Extract JSON from response
                template_data = self._parse_template_json(raw_text)
                if not template_data:
                    self._emit("template_gen_parse_error", f"Failed to parse JSON: {raw_text[:200]}")
                    continue
                
                # Create TemplateSpec
                template = TemplateSpec(
                    template_id=template_data.get("template_id", template_id),
                    domain_tags=[t.lower() for t in template_data.get("domain_tags", [])],
                    description=template_data.get("description", "Generated template"),
                    knowledge_seeds=template_data.get("knowledge_seeds", []),
                    graph_blueprint=template_data.get("graph_blueprint", {}),
                    path="",
                )
                
                # Validate
                is_valid, errors = self.validator.validate(template)
                
                if is_valid:
                    # Cache the successful template
                    self.cache.save(template)
                    self._emit("template_gen_success", f"Generated template: {template.template_id}")
                    return GenerationResult(
                        success=True,
                        template=template,
                        attempts=attempt,
                    )
                else:
                    self._emit("template_gen_invalid", f"Validation errors: {errors}")
                    # For retry, we could potentially pass errors back to LLM
                    
            except Exception as e:
                self._emit("template_gen_error", f"Generation failed: {str(e)[:150]}")
        
        return GenerationResult(
            success=False,
            template=None,
            error="Max generation attempts exceeded",
            attempts=self.max_attempts,
        )
    
    def _parse_template_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse template JSON from LLM response.
        
        Handles various formats including:
        - Raw JSON
        - JSON in markdown code blocks
        - JSON with surrounding text
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from code block
        if "```json" in text:
            try:
                start = text.index("```json") + 7
                end = text.index("```", start)
                return json.loads(text[start:end].strip())
            except (ValueError, json.JSONDecodeError):
                pass
        
        # Try extracting from any code block
        if "```" in text:
            try:
                start = text.index("```") + 3
                # Skip language identifier if present
                if text[start:start+1].isalpha():
                    start = text.index("\n", start) + 1
                end = text.index("```", start)
                return json.loads(text[start:end].strip())
            except (ValueError, json.JSONDecodeError):
                pass
        
        # Try finding JSON object in text
        try:
            start = text.index("{")
            # Find matching closing brace
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[start:i+1])
        except (ValueError, json.JSONDecodeError):
            pass
        
        return None
    
    def generate_from_examples(
        self,
        problem: str,
        similar_traces: List[Dict[str, Any]],
    ) -> GenerationResult:
        """
        Generate a template based on successful similar traces.
        
        Uses patterns from successful executions to inform template design.
        """
        if not similar_traces:
            return self.generate(problem)
        
        # Extract patterns from traces
        node_types_used = set()
        roles_used = set()
        avg_node_count = 0
        
        for trace in similar_traces[:5]:
            nodes = trace.get("nodes", [])
            avg_node_count += len(nodes)
            for node in nodes:
                node_types_used.add(node.get("type", "logic"))
                roles_used.add(node.get("role", "logic"))
        
        if similar_traces:
            avg_node_count = avg_node_count // len(similar_traces)
        
        # Add pattern hints to context
        pattern_hint = (
            f"Similar successful solutions used node types: {', '.join(node_types_used)}, "
            f"roles: {', '.join(roles_used)}, "
            f"average {avg_node_count} nodes."
        )
        
        return self.generate(problem, rag_context=[pattern_hint])

