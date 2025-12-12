from __future__ import annotations


def test_intent_detection_latest_album():
    from apps.mas.agents.websearch import WebSearchAgent

    agent = WebSearchAgent(client=None)  # type: ignore[arg-type]
    assert agent._detect_intent("Which is the latest album of Taylor Swift?") == "latest_album_of_artist"
    assert agent._detect_intent("What is the newest album by Hasan Raheem?") == "latest_album_of_artist"


def test_intent_detection_song_performer():
    from apps.mas.agents.websearch import WebSearchAgent

    agent = WebSearchAgent(client=None)  # type: ignore[arg-type]
    assert agent._detect_intent('Who sang "The Fate of Ophelia"?') == "who_sang_or_performed_song"
    assert agent._detect_intent("Who performed 'Bohemian Rhapsody'?") == "who_sang_or_performed_song"


def test_intent_detection_current_title_holder():
    from apps.mas.agents.websearch import WebSearchAgent

    agent = WebSearchAgent(client=None)  # type: ignore[arg-type]
    assert agent._detect_intent("Who is the current F1 champion?") == "current_title_holder"
    assert agent._detect_intent("Who is the current Prime Minister of the UK?") == "current_title_holder"


def test_entity_extraction_album_artist():
    from apps.mas.agents.websearch import WebSearchAgent

    agent = WebSearchAgent(client=None)  # type: ignore[arg-type]
    assert agent._extract_artist_from_album_question("Which is the latest album of Taylor Swift?") == "Taylor Swift"
    assert agent._extract_artist_from_album_question("Taylor Swift's latest album?") == "Taylor Swift"


def test_entity_extraction_song_title():
    from apps.mas.agents.websearch import WebSearchAgent

    agent = WebSearchAgent(client=None)  # type: ignore[arg-type]
    assert agent._extract_song_title('Who sang "The Fate of Ophelia"?') == "The Fate of Ophelia"
    assert agent._extract_song_title("Who performed 'The Fate of Ophelia' and why?") == "The Fate of Ophelia"


