"""Transparent metadata about how a Noema response was produced."""

from __future__ import annotations

from dataclasses import dataclass

from core.router import RouteDecision


@dataclass(frozen=True)
class ResponseMetadata:
    internet_used: bool
    research_used: bool
    sources: tuple[str, ...]
    confidence_level: str
    confidence_reason: str


def build_response_metadata(
    *,
    route: RouteDecision,
    history_used: bool,
    private_reference_used: bool,
    internet_used: bool,
    research_used: bool,
    retrieval_succeeded: bool,
    memory_used: bool = False,
    foundational_used: bool = False,
    cache_used: bool = False,
    retrieval_provider: str | None = None,
    confidence_override: str | None = None,
) -> ResponseMetadata:
    sources = ["Safety Rules"]
    if route.intent == "crisis / safety":
        sources.append("Local Crisis Resources")
    elif internet_used:
        provider = retrieval_provider or "Internet Search"
        sources.append(
            f"{provider} (cached)" if cache_used else provider
        )
    elif research_used:
        sources.extend(("Research Papers", "Internal Knowledge"))
    else:
        sources.append("Internal Knowledge")
        if foundational_used:
            sources.append("Foundational Literature (not a live search)")
        elif route.knowledge_route in {"internet", "research papers"}:
            sources.append("No external sources used")

    if memory_used:
        sources.append("User-approved Session Memory")
    if history_used:
        sources.append("Conversation Context")
    if private_reference_used:
        sources.append("Private Clinical Reference")

    if route.intent == "crisis / safety":
        level = "High"
        reason = "A direct safety phrase activated verified local safety rules."
    elif route.knowledge_route in {"internet", "research papers"}:
        if foundational_used:
            level = "Medium"
            reason = (
                "The works are established, but no live or comprehensive literature "
                "search was performed."
            )
        elif not retrieval_succeeded:
            level = "Low"
            reason = "Current or research information could not be verified live."
        elif research_used:
            level = "Medium"
            reason = (
                "Research sources were retrieved, but a short search may not capture "
                "the complete literature."
            )
        else:
            level = "Medium"
            reason = (
                "Sources were retrieved and cited, but current facts can change."
                if not cache_used
                else "A still-valid cached answer is backed by previously cited sources."
            )
    elif route.intent in {
        "grief",
        "anxiety / stress",
        "overwhelm",
        "emotional reflection",
        "cognitive challenge",
    }:
        level = "Medium"
        reason = "The response is based on limited text and a tentative interpretation."
    else:
        level = "High"
        reason = "The route uses stable conversational reasoning without current facts."

    return ResponseMetadata(
        internet_used=internet_used,
        research_used=research_used,
        sources=tuple(dict.fromkeys(sources)),
        confidence_level=confidence_override or level,
        confidence_reason=reason,
    )
