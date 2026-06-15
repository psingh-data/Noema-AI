from core.response_metadata import build_response_metadata
from core.router import RouteDecision


def route(intent: str, knowledge: str) -> RouteDecision:
    return RouteDecision(
        intent=intent,
        response_mode="Test",
        knowledge_route=knowledge,
        confidence=0.8,
        reason="test",
    )


def test_emotional_response_reports_internal_sources_and_medium_confidence():
    metadata = build_response_metadata(
        route=route("grief", "conversation context"),
        history_used=False,
        private_reference_used=False,
        internet_used=False,
        research_used=False,
        retrieval_succeeded=False,
    )
    assert metadata.sources == ("Safety Rules", "Internal Knowledge")
    assert metadata.confidence_level == "Medium"


def test_failed_current_lookup_reports_no_external_sources_and_low_confidence():
    metadata = build_response_metadata(
        route=route("current factual search", "internet"),
        history_used=False,
        private_reference_used=False,
        internet_used=False,
        research_used=False,
        retrieval_succeeded=False,
    )
    assert "No external sources used" in metadata.sources
    assert metadata.confidence_level == "Low"


def test_successful_research_route_reports_papers():
    metadata = build_response_metadata(
        route=route("research paper question", "research papers"),
        history_used=True,
        private_reference_used=False,
        internet_used=False,
        research_used=True,
        retrieval_succeeded=True,
    )
    assert "Research Papers" in metadata.sources
    assert "Conversation Context" in metadata.sources
    assert metadata.confidence_level == "Medium"


def test_successful_tavily_lookup_reports_provider_confidence():
    metadata = build_response_metadata(
        route=route("current factual search", "internet"),
        history_used=False,
        private_reference_used=False,
        internet_used=True,
        research_used=False,
        retrieval_succeeded=True,
        retrieval_provider="Tavily",
        confidence_override="High",
    )
    assert "Tavily" in metadata.sources
    assert "Internal Knowledge" not in metadata.sources
    assert metadata.confidence_level == "High"


def test_safety_route_uses_local_resources_with_high_confidence():
    metadata = build_response_metadata(
        route=route("crisis / safety", "local crisis resources"),
        history_used=False,
        private_reference_used=False,
        internet_used=False,
        research_used=False,
        retrieval_succeeded=True,
    )
    assert "Local Crisis Resources" in metadata.sources
    assert metadata.confidence_level == "High"
