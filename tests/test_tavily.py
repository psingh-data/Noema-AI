import pytest

import ai.tavily as tavily
from ai.tavily import (
    ACADEMIC_DOMAINS,
    format_tavily_answer,
    search_tavily,
    should_use_tavily,
)
from core.conversation import continue_conversation


FACTUAL_CASES = (
    "What are the current Germany student visa requirements?",
    "What is the application deadline for Osnabruck University?",
    "What is the latest AI news?",
    "Which laptop should I buy for data science?",
    "Compare laptops for machine learning.",
    "What is the data scientist salary in Germany?",
    "What software tools are currently used for MLOps?",
    "What regulations apply to AI products?",
)


@pytest.mark.parametrize("query", FACTUAL_CASES)
def test_required_factual_queries_trigger_tavily(query):
    reply = continue_conversation(query)
    assert reply.route.knowledge_route == "internet"
    assert should_use_tavily(reply.route)


@pytest.mark.parametrize(
    "query",
    (
        "My grandfather died and I miss him.",
        "I feel lonely tonight.",
        "I don't need advice. I just want to vent.",
        "Everything feels overwhelming.",
        "AI news is overwhelming me.",
        "I feel lost and hopeless about my career.",
        "I am the king.",
    ),
)
def test_emotional_and_casual_queries_do_not_trigger_tavily(query):
    reply = continue_conversation(query)
    assert not should_use_tavily(reply.route)


def test_tavily_returns_summary_sources_urls_and_confidence(monkeypatch):
    captured = {}

    def fake_search(payload, api_key):
        captured["payload"] = payload
        captured["api_key"] = api_key
        return {
            "answer": "Germany requires student visa applicants to follow current official requirements.",
            "results": [
                {
                    "title": "German Federal Foreign Office",
                    "url": "https://www.auswaertiges-amt.de/",
                    "score": 0.92,
                },
                {
                    "title": "Make it in Germany",
                    "url": "https://www.make-it-in-germany.com/",
                    "score": 0.84,
                },
                {
                    "title": "BAMF",
                    "url": "https://www.bamf.de/",
                    "score": 0.78,
                },
            ],
        }

    monkeypatch.setattr(tavily, "_create_tavily_search", fake_search)
    result = search_tavily("Germany student visa requirements", api_key="test-key")

    assert result.succeeded
    assert result.confidence == "High"
    assert len(result.sources) == 3
    assert result.sources[0].title == "German Federal Foreign Office"
    assert result.sources[0].url == "https://www.auswaertiges-amt.de/"
    assert captured["payload"]["include_answer"] == "advanced"
    assert captured["payload"]["max_results"] == 5
    assert captured["api_key"] == "test-key"
    formatted = format_tavily_answer(result)
    assert "**Summary**" in formatted
    assert "**Sources**" in formatted
    assert "https://www.auswaertiges-amt.de/" in formatted
    advice = format_tavily_answer(result, advice=True)
    assert advice.index("**Practical guidance**") < advice.index(
        "**Why this guidance:**"
    )


def test_ai_news_uses_news_topic_and_recent_window(monkeypatch):
    captured = {}

    def fake_search(payload, _api_key):
        captured.update(payload)
        return {
            "answer": "A sourced AI news summary.",
            "results": [
                {
                    "title": "AI News Source",
                    "url": "https://example.com/ai-news",
                    "score": 0.8,
                }
            ],
        }

    monkeypatch.setattr(tavily, "_create_tavily_search", fake_search)
    result = search_tavily("What is the latest AI news?", api_key="test-key")

    assert result.succeeded
    assert captured["topic"] == "news"
    assert captured["time_range"] == "week"


def test_tavily_failure_returns_no_factual_summary(monkeypatch):
    def failed_search(_payload, _api_key):
        raise OSError("network unavailable")

    monkeypatch.setattr(tavily, "_create_tavily_search", failed_search)
    result = search_tavily("Current visa rules", api_key="test-key")

    assert not result.succeeded
    assert result.summary == ""
    assert result.sources == ()
    assert result.confidence == "Low"


def test_research_route_uses_tavily_academic_domains(monkeypatch):
    captured = {}

    def fake_search(payload, _api_key):
        captured.update(payload)
        return {
            "answer": "Cognitive reframing is commonly studied within CBT.",
            "results": [
                {
                    "title": "CBT Review",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/00000000/",
                    "score": 0.88,
                    "content": "A review reports evidence across several conditions.",
                },
                {
                    "title": "Meta-analysis",
                    "url": "https://doi.org/10.1000/example",
                    "score": 0.76,
                    "content": "Effects vary by population and outcome.",
                },
            ],
        }

    monkeypatch.setattr(tavily, "_create_tavily_search", fake_search)
    result = search_tavily(
        "What does research say about cognitive reframing?",
        api_key="test-key",
        research=True,
    )

    assert result.succeeded
    assert captured["max_results"] == 8
    assert captured["include_domains"] == list(ACADEMIC_DOMAINS)
    assert result.sources[0].excerpt.startswith("A review reports")
    formatted = format_tavily_answer(result, research=True)
    assert "plain-language version" in formatted
    assert "**Research summary**" in formatted
    assert "**Sources**" in formatted
    assert "**Evidence excerpt:**" in formatted
    assert "not a complete systematic review" in formatted

    advice_formatted = format_tavily_answer(result, research=True, advice=True)
    assert advice_formatted.index("**Direct answer**") < advice_formatted.index(
        "**Research summary**"
    )
    assert "Evidence-based options" in advice_formatted


def test_research_route_triggers_tavily():
    reply = continue_conversation(
        "What does research say about cognitive reframing?"
    )
    assert reply.route.knowledge_route == "research papers"
    assert should_use_tavily(reply.route)
