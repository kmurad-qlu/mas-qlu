"""
Hybrid timeline inference (events -> constraints -> bounded date/range).

Goal: answer questions like:
  - "When was Darryl Prue sent home?"
by extracting dated events from evidence and computing a bounded interval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..infra.openrouter.client import OpenRouterClient


@dataclass
class TimelineEvidence:
    id: str
    url: str
    quote: str


@dataclass
class TimelineEvent:
    id: str
    description: str
    start_date: Optional[str] = None  # ISO YYYY-MM-DD
    end_date: Optional[str] = None  # ISO YYYY-MM-DD
    entity: Optional[str] = None
    evidence_ids: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.evidence_ids is None:
            self.evidence_ids = []


@dataclass
class TimelineConstraint:
    target_event: str
    relation: str  # "after" or "before"
    ref_event: str
    ref_point: str  # "start" or "end"
    inclusive: bool
    evidence_ids: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.evidence_ids is None:
            self.evidence_ids = []


@dataclass
class TimelineExtraction:
    target_event_id: str
    events: List[TimelineEvent]
    constraints: List[TimelineConstraint]
    evidence: List[TimelineEvidence]


@dataclass
class TimelineAnswer:
    answer_type: str  # "date" | "range" | "unknown" | "inconsistent"
    lower: Optional[date] = None
    upper: Optional[date] = None
    lower_inclusive: bool = False
    upper_inclusive: bool = False
    citations: List[TimelineEvidence] = None  # type: ignore[assignment]
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.citations is None:
            self.citations = []

    def format_citations(self, prefix: str = "W") -> str:
        lines: List[str] = []
        for i, ev in enumerate(self.citations, 1):
            if ev.url and ev.quote:
                lines.append(f"[{prefix}{i}] {ev.url} — \"{ev.quote[:180]}\"")
            elif ev.url:
                lines.append(f"[{prefix}{i}] {ev.url}")
        return "\n".join(lines).strip()

    def format_answer(self) -> str:
        if self.answer_type == "date" and self.lower:
            return self.lower.isoformat()
        if self.answer_type == "range" and self.lower and self.upper:
            # Human-friendly range
            return f"between {self.lower.isoformat()} and {self.upper.isoformat()}"
        if self.answer_type == "inconsistent":
            return "inconsistent evidence (cannot form a valid date range)"
        return "unknown"


SYSTEM_TIMELINE = (
    "You are a Temporal Evidence Extractor.\n"
    "Your job is to extract dated events and explicit temporal constraints from the evidence.\n"
    "Do NOT guess dates. Only extract what is supported by the evidence.\n"
    "Output STRICT JSON only.\n"
)

TIMELINE_PROMPT = """
Question: {question}

Evidence:
{evidence}

Task:
1) Identify the target event implied by the question (give it an id, e.g. 'sent_home').
2) Extract dated events from the evidence with ISO dates.
   - If an event has a range, provide start_date and end_date.
3) Extract temporal constraints about the target event using ONLY the evidence.
   - Use constraints like: target after ref_event end, target before ref_event start.
4) Provide evidence quotes for each extracted event/constraint.

Return JSON with this schema:
{{
  "target_event_id": "...",
  "events": [
    {{"id":"...","description":"...","entity":"...","start_date":"YYYY-MM-DD|null","end_date":"YYYY-MM-DD|null","evidence_ids":["E1"]}}
  ],
  "constraints": [
    {{"target_event":"...","relation":"after|before","ref_event":"...","ref_point":"start|end","inclusive":false,"evidence_ids":["E2"]}}
  ],
  "evidence": [
    {{"id":"E1","url":"...","quote":"..."}}
  ]
}}
""".strip()


def _parse_iso_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    t = s.strip()
    if not t or t.lower() == "null":
        return None
    # Prefer ISO
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return date(y, mo, d)
    # Fallback: try common formats
    for fmt in ("%Y/%m/%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(t, fmt).date()
        except Exception:
            continue
    return None


def extract_timeline(
    *,
    client: OpenRouterClient,
    model: str,
    question: str,
    evidence_text: str,
    emit: Optional[Callable[[str, str], None]] = None,
) -> Optional[TimelineExtraction]:
    """
    Use an LLM to extract events + constraints as structured JSON.
    """
    emit = emit or (lambda *_: None)
    if not evidence_text or not evidence_text.strip():
        emit("timeline_skip", "No evidence text provided; skipping timeline extraction")
        return None

    prompt = TIMELINE_PROMPT.format(question=question, evidence=evidence_text[:12000])
    res = client.complete_chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_TIMELINE},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=1800,
    )
    raw = (res.text or "").strip()
    emit("timeline_raw", raw[:1200])

    # Strip fenced code if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            raw = "\n".join(lines[1:-1]).strip()

    try:
        obj = json.loads(raw)
    except Exception as e:
        emit("timeline_parse_error", f"Failed to parse JSON: {str(e)[:200]}")
        return None

    try:
        events = [
            TimelineEvent(
                id=str(ev.get("id")),
                description=str(ev.get("description") or ""),
                entity=(str(ev.get("entity")) if ev.get("entity") else None),
                start_date=(str(ev.get("start_date")) if ev.get("start_date") else None),
                end_date=(str(ev.get("end_date")) if ev.get("end_date") else None),
                evidence_ids=list(ev.get("evidence_ids") or []),
            )
            for ev in (obj.get("events") or [])
        ]
        constraints = [
            TimelineConstraint(
                target_event=str(c.get("target_event")),
                relation=str(c.get("relation")),
                ref_event=str(c.get("ref_event")),
                ref_point=str(c.get("ref_point")),
                inclusive=bool(c.get("inclusive", False)),
                evidence_ids=list(c.get("evidence_ids") or []),
            )
            for c in (obj.get("constraints") or [])
        ]
        evidence = [
            TimelineEvidence(
                id=str(evv.get("id")),
                url=str(evv.get("url") or ""),
                quote=str(evv.get("quote") or ""),
            )
            for evv in (obj.get("evidence") or [])
        ]
        target_event_id = str(obj.get("target_event_id") or "")
        if not target_event_id:
            return None
        ext = TimelineExtraction(
            target_event_id=target_event_id,
            events=events,
            constraints=constraints,
            evidence=evidence,
        )
        emit("timeline_facts", f"Extracted events={len(events)} constraints={len(constraints)} evidence={len(evidence)} target={target_event_id}")
        return ext
    except Exception as e:
        emit("timeline_struct_error", f"Failed to build extraction: {str(e)[:200]}")
        return None


def solve_timeline(extraction: TimelineExtraction) -> TimelineAnswer:
    """
    Deterministic solver: compute a bounded date/range for the target event.
    """
    ev_by_id: Dict[str, TimelineEvent] = {e.id: e for e in extraction.events if e.id}
    evidence_by_id: Dict[str, TimelineEvidence] = {e.id: e for e in extraction.evidence if e.id}

    target = extraction.target_event_id
    lower: Optional[date] = None
    upper: Optional[date] = None
    lower_incl = False
    upper_incl = False
    used_evidence_ids: List[str] = []

    # Direct date on target?
    if target in ev_by_id:
        tev = ev_by_id[target]
        sd = _parse_iso_date(tev.start_date)
        ed = _parse_iso_date(tev.end_date) or sd
        if sd and ed:
            if sd == ed:
                used_evidence_ids.extend(tev.evidence_ids)
                return TimelineAnswer(
                    answer_type="date",
                    lower=sd,
                    upper=sd,
                    lower_inclusive=True,
                    upper_inclusive=True,
                    citations=[evidence_by_id[eid] for eid in tev.evidence_ids if eid in evidence_by_id],
                    rationale="Target event had an explicit date in extracted events.",
                )
            used_evidence_ids.extend(tev.evidence_ids)
            return TimelineAnswer(
                answer_type="range",
                lower=sd,
                upper=ed,
                lower_inclusive=True,
                upper_inclusive=True,
                citations=[evidence_by_id[eid] for eid in tev.evidence_ids if eid in evidence_by_id],
                rationale="Target event had an explicit date range in extracted events.",
            )

    for c in extraction.constraints:
        if c.target_event != target:
            continue
        ref = ev_by_id.get(c.ref_event)
        if not ref:
            continue
        ref_date = _parse_iso_date(ref.start_date if c.ref_point == "start" else ref.end_date)
        if not ref_date:
            continue
        used_evidence_ids.extend(c.evidence_ids)
        used_evidence_ids.extend(ref.evidence_ids)

        if c.relation == "after":
            # lower bound
            if lower is None or ref_date > lower or (ref_date == lower and c.inclusive and not lower_incl):
                lower = ref_date
                lower_incl = c.inclusive
        elif c.relation == "before":
            if upper is None or ref_date < upper or (ref_date == upper and c.inclusive and not upper_incl):
                upper = ref_date
                upper_incl = c.inclusive

    if lower and upper:
        # Convert exclusive bounds into inclusive day-range bounds where possible.
        low = lower + timedelta(days=1) if not lower_incl else lower
        up = upper - timedelta(days=1) if not upper_incl else upper
        if low > up:
            return TimelineAnswer(
                answer_type="inconsistent",
                citations=[evidence_by_id[eid] for eid in dict.fromkeys(used_evidence_ids) if eid in evidence_by_id],
                rationale="Computed range was empty after applying exclusivity.",
            )
        return TimelineAnswer(
            answer_type="range",
            lower=low,
            upper=up,
            lower_inclusive=True,
            upper_inclusive=True,
            citations=[evidence_by_id[eid] for eid in dict.fromkeys(used_evidence_ids) if eid in evidence_by_id],
            rationale="Computed bounded range from extracted temporal constraints.",
        )
    if lower:
        return TimelineAnswer(
            answer_type="range",
            lower=(lower + timedelta(days=1) if not lower_incl else lower),
            upper=None,
            lower_inclusive=True,
            upper_inclusive=False,
            citations=[evidence_by_id[eid] for eid in dict.fromkeys(used_evidence_ids) if eid in evidence_by_id],
            rationale="Only a lower bound was available from constraints.",
        )
    if upper:
        return TimelineAnswer(
            answer_type="range",
            lower=None,
            upper=(upper - timedelta(days=1) if not upper_incl else upper),
            lower_inclusive=False,
            upper_inclusive=True,
            citations=[evidence_by_id[eid] for eid in dict.fromkeys(used_evidence_ids) if eid in evidence_by_id],
            rationale="Only an upper bound was available from constraints.",
        )
    return TimelineAnswer(answer_type="unknown", rationale="No usable dated constraints extracted.")


