"""
Trace storage for TGR execution traces.

Provides persistence and querying capabilities for execution traces
to support the distillation learning loop.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .trace_recorder import ExecutionTrace


class TraceStore:
    """
    Persistent storage for execution traces.
    
    Stores traces as JSON files organized by template_id for efficient
    querying and pattern analysis.
    """
    
    DEFAULT_PATH = "apps/mas/data/traces"
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the trace store.
        
        Args:
            storage_path: Path to store traces. Defaults to apps/mas/data/traces
        """
        self.storage_path = Path(storage_path or self.DEFAULT_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create index file if it doesn't exist
        self._index_path = self.storage_path / "index.json"
        if not self._index_path.exists():
            self._save_index({"traces": [], "stats": {}})
    
    def _load_index(self) -> Dict[str, Any]:
        """Load the trace index."""
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"traces": [], "stats": {}}
    
    def _save_index(self, index: Dict[str, Any]) -> None:
        """Save the trace index."""
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
    
    def _get_trace_path(self, trace_id: str, template_id: str) -> Path:
        """Get the file path for a trace."""
        # Organize by template_id for efficient querying
        template_dir = self.storage_path / (template_id or "unknown")
        template_dir.mkdir(parents=True, exist_ok=True)
        return template_dir / f"{trace_id}.json"
    
    def save(self, trace: ExecutionTrace) -> None:
        """
        Persist a trace to disk.
        
        Args:
            trace: The execution trace to save
        """
        # Save trace file
        trace_path = self._get_trace_path(trace.trace_id, trace.template_id)
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, indent=2)
        
        # Update index
        index = self._load_index()
        index["traces"].append({
            "trace_id": trace.trace_id,
            "template_id": trace.template_id,
            "timestamp": trace.timestamp.isoformat(),
            "verified": trace.verified,
            "problem_hash": trace.problem_hash(),
            "path": str(trace_path.relative_to(self.storage_path)),
        })
        
        # Update stats
        stats = index.get("stats", {})
        template_stats = stats.get(trace.template_id, {
            "total": 0,
            "verified": 0,
            "avg_duration_ms": 0.0,
        })
        template_stats["total"] += 1
        if trace.verified:
            template_stats["verified"] += 1
        # Rolling average for duration
        n = template_stats["total"]
        template_stats["avg_duration_ms"] = (
            (template_stats["avg_duration_ms"] * (n - 1) + trace.total_duration_ms) / n
        )
        stats[trace.template_id] = template_stats
        index["stats"] = stats
        
        self._save_index(index)
    
    def load(self, trace_id: str, template_id: Optional[str] = None) -> Optional[ExecutionTrace]:
        """
        Load a specific trace by ID.
        
        Args:
            trace_id: The trace ID to load
            template_id: Optional template ID to narrow search
        
        Returns:
            The ExecutionTrace if found, None otherwise
        """
        # If template_id provided, look directly
        if template_id:
            trace_path = self._get_trace_path(trace_id, template_id)
            if trace_path.exists():
                with open(trace_path, "r", encoding="utf-8") as f:
                    return ExecutionTrace.from_dict(json.load(f))
            return None
        
        # Otherwise, search index
        index = self._load_index()
        for entry in index.get("traces", []):
            if entry["trace_id"] == trace_id:
                trace_path = self.storage_path / entry["path"]
                if trace_path.exists():
                    with open(trace_path, "r", encoding="utf-8") as f:
                        return ExecutionTrace.from_dict(json.load(f))
        return None
    
    def query(
        self,
        template_id: Optional[str] = None,
        verified_only: bool = True,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[ExecutionTrace]:
        """
        Query traces matching criteria.
        
        Args:
            template_id: Filter by template ID
            verified_only: Only return verified traces
            since: Only return traces after this timestamp
            limit: Maximum number of traces to return
        
        Returns:
            List of matching ExecutionTrace objects
        """
        index = self._load_index()
        matching: List[ExecutionTrace] = []
        
        for entry in reversed(index.get("traces", [])):  # Most recent first
            if len(matching) >= limit:
                break
            
            # Apply filters
            if template_id and entry.get("template_id") != template_id:
                continue
            if verified_only and not entry.get("verified"):
                continue
            if since:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time < since:
                    continue
            
            # Load full trace
            trace_path = self.storage_path / entry["path"]
            if trace_path.exists():
                try:
                    with open(trace_path, "r", encoding="utf-8") as f:
                        trace = ExecutionTrace.from_dict(json.load(f))
                        matching.append(trace)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return matching
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate statistics across all traces.
        
        Returns:
            Dictionary with statistics:
            - total_traces: Total number of traces
            - verified_traces: Number of verified traces
            - templates: Per-template statistics
            - recent_success_rate: Success rate in last 100 traces
        """
        index = self._load_index()
        traces = index.get("traces", [])
        stats = index.get("stats", {})
        
        total = len(traces)
        verified = sum(1 for t in traces if t.get("verified"))
        
        # Recent success rate (last 100)
        recent = traces[-100:] if len(traces) > 100 else traces
        recent_verified = sum(1 for t in recent if t.get("verified"))
        recent_rate = recent_verified / len(recent) if recent else 0.0
        
        return {
            "total_traces": total,
            "verified_traces": verified,
            "overall_success_rate": verified / total if total > 0 else 0.0,
            "recent_success_rate": recent_rate,
            "templates": stats,
            "unique_templates": len(stats),
        }
    
    def get_template_traces(
        self,
        template_id: str,
        verified_only: bool = True,
        limit: int = 50,
    ) -> List[ExecutionTrace]:
        """
        Get traces for a specific template.
        
        Convenience method for pattern analysis.
        """
        return self.query(template_id=template_id, verified_only=verified_only, limit=limit)
    
    def get_similar_problems(
        self,
        problem_hash: str,
        limit: int = 5,
    ) -> List[ExecutionTrace]:
        """
        Find traces with similar problems (by hash).
        
        Useful for few-shot example selection.
        """
        index = self._load_index()
        matching: List[ExecutionTrace] = []
        
        for entry in reversed(index.get("traces", [])):
            if len(matching) >= limit:
                break
            if entry.get("problem_hash") == problem_hash and entry.get("verified"):
                trace_path = self.storage_path / entry["path"]
                if trace_path.exists():
                    try:
                        with open(trace_path, "r", encoding="utf-8") as f:
                            trace = ExecutionTrace.from_dict(json.load(f))
                            matching.append(trace)
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        return matching
    
    def prune_old_traces(
        self,
        keep_verified: int = 1000,
        keep_unverified: int = 100,
    ) -> int:
        """
        Prune old traces to manage storage.
        
        Keeps the most recent verified and unverified traces.
        
        Args:
            keep_verified: Number of verified traces to keep
            keep_unverified: Number of unverified traces to keep
        
        Returns:
            Number of traces pruned
        """
        index = self._load_index()
        traces = index.get("traces", [])
        
        # Separate verified and unverified
        verified = [t for t in traces if t.get("verified")]
        unverified = [t for t in traces if not t.get("verified")]
        
        # Determine which to prune
        to_prune: List[Dict[str, Any]] = []
        if len(verified) > keep_verified:
            to_prune.extend(verified[:-keep_verified])
        if len(unverified) > keep_unverified:
            to_prune.extend(unverified[:-keep_unverified])
        
        # Delete files
        for entry in to_prune:
            trace_path = self.storage_path / entry["path"]
            if trace_path.exists():
                try:
                    os.remove(trace_path)
                except OSError:
                    pass
        
        # Update index
        pruned_ids = {t["trace_id"] for t in to_prune}
        index["traces"] = [t for t in traces if t["trace_id"] not in pruned_ids]
        self._save_index(index)
        
        return len(to_prune)
    
    def count(self, template_id: Optional[str] = None, verified_only: bool = False) -> int:
        """
        Count traces matching criteria.
        
        Args:
            template_id: Filter by template ID
            verified_only: Only count verified traces
        
        Returns:
            Number of matching traces
        """
        index = self._load_index()
        count = 0
        
        for entry in index.get("traces", []):
            if template_id and entry.get("template_id") != template_id:
                continue
            if verified_only and not entry.get("verified"):
                continue
            count += 1
        
        return count

