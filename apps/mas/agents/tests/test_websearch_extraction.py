from __future__ import annotations

from apps.mas.tools.search import WebResult


def test_extract_latest_album_candidates_prefers_quoted_album():
    from apps.mas.agents.websearch import WebSearchAgent

    agent = WebSearchAgent(client=None)  # type: ignore[arg-type]
    artist = "Taylor Swift"

    results = [
        WebResult(
            title='Taylor Swift releases new album "The Life of a Showgirl" in October 2025',
            body='The album "The Life of a Showgirl" is her latest studio release.',
            url="https://example.com/news1",
            source="NEWS",
            date="2025-10-03",
        ),
        WebResult(
            title='Review: "The Life of a Showgirl" is Taylor Swift’s boldest album yet',
            body='Critics discuss the themes of "The Life of a Showgirl".',
            url="https://example.org/review",
            source="WEB",
            date=None,
        ),
        WebResult(
            title="Spotify Wrapped 2025 top artists, songs and albums revealed",
            body="Unrelated roundup.",
            url="https://example.net/wrapped",
            source="NEWS",
            date="2025-12-03",
        ),
    ]

    cands = agent._extract_album_candidates(artist, results)
    assert cands, "expected at least one candidate"
    assert cands[0]["value"] == "The Life of a Showgirl"
    assert cands[0]["confidence"] >= 0.5


def test_extract_song_performer_candidates_from_news_title():
    from apps.mas.agents.websearch import WebSearchAgent

    agent = WebSearchAgent(client=None)  # type: ignore[arg-type]
    song = "The Fate of Ophelia"

    results = [
        WebResult(
            title="Taylor Swift References a Shakespearean Figure in 'The Fate of Ophelia': Lyrics, Explained",
            body="A breakdown of lyrics.",
            url="https://news.example.com/article",
            source="NEWS",
            date="2025-10-03",
        ),
        WebResult(
            title="'The Fate of Ophelia': Taylor Swift reveals the meaning behind the lyrics",
            body="More context.",
            url="https://news.example.org/aol",
            source="NEWS",
            date="2025-10-03",
        ),
    ]

    cands = agent._extract_song_performer_candidates(song, results)
    assert cands, "expected at least one performer candidate"
    assert cands[0]["value"] == "Taylor Swift"


def test_extract_title_holder_candidates_from_wins_headline():
    from apps.mas.agents.websearch import WebSearchAgent

    agent = WebSearchAgent(client=None)  # type: ignore[arg-type]
    results = [
        WebResult(
            title="Lando Norris wins Formula 1 world championship in the final race of 2025",
            body="The McLaren driver clinched his first title.",
            url="https://sports.example.com/f1",
            source="NEWS",
            date="2025-12-08",
        ),
        WebResult(
            title="Formula 1: Lando Norris wins 2025 world championship",
            body="Norris, 26, finished third in Abu Dhabi.",
            url="https://sports.example.org/f1",
            source="NEWS",
            date="2025-12-07",
        ),
    ]

    cands = agent._extract_title_holder_candidates("Who is the current F1 champion?", results)
    assert cands and cands[0]["value"] == "Lando Norris"


