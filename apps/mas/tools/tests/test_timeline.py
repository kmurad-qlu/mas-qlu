from __future__ import annotations

from apps.mas.tools.timeline import (
    TimelineConstraint,
    TimelineEvidence,
    TimelineEvent,
    TimelineExtraction,
    solve_timeline,
)


def test_solve_timeline_bounded_range_from_constraints():
    ext = TimelineExtraction(
        target_event_id="sent_home",
        events=[
            TimelineEvent(
                id="prue_stint",
                description="Darryl Prue stint",
                entity="Darryl Prue",
                start_date="1995-07-01",
                end_date="1995-07-09",
                evidence_ids=["E1"],
            ),
            TimelineEvent(
                id="coles_game",
                description="Alexander Coles played",
                entity="Alexander Coles",
                start_date="1995-07-16",
                end_date="1995-07-16",
                evidence_ids=["E2"],
            ),
        ],
        constraints=[
            TimelineConstraint(
                target_event="sent_home",
                relation="after",
                ref_event="prue_stint",
                ref_point="end",
                inclusive=False,
                evidence_ids=["E1"],
            ),
            TimelineConstraint(
                target_event="sent_home",
                relation="before",
                ref_event="coles_game",
                ref_point="start",
                inclusive=False,
                evidence_ids=["E2"],
            ),
        ],
        evidence=[
            TimelineEvidence(id="E1", url="https://en.wikipedia.org/wiki/1995_Ginebra_San_Miguel_season", quote="Darryl Prue ... July 1–9"),
            TimelineEvidence(id="E2", url="https://en.wikipedia.org/wiki/1995_Ginebra_San_Miguel_season", quote="Alexander Coles ... July 16"),
        ],
    )

    ans = solve_timeline(ext)
    assert ans.answer_type == "range"
    assert ans.lower is not None and ans.upper is not None
    # exclusive bounds => (1995-07-10 .. 1995-07-15)
    assert ans.lower.isoformat() == "1995-07-10"
    assert ans.upper.isoformat() == "1995-07-15"


