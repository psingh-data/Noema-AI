"""Validated Tavily retrieval for current facts and academic research."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

from core.router import RouteDecision


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_DENIED_INTENTS = {
    "grief",
    "emotional reflection",
    "venting",
    "anxiety / stress",
    "overwhelm",
    "casual conversation",
}
NEWS_MARKERS = (
    "news",
    "current event",
    "latest development",
    "what happened today",
)
ACADEMIC_DOMAINS = (
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "semanticscholar.org",
    "api.semanticscholar.org",
    "crossref.org",
    "api.crossref.org",
    "arxiv.org",
    "doi.org",
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "apa.org",
)


@dataclass(frozen=True)
class TavilySource:
    title: str
    url: str
    score: float
    excerpt: str = ""


@dataclass(frozen=True)
class TavilyResult:
    summary: str
    sources: tuple[TavilySource, ...]
    confidence: str
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return bool(self.summary and self.sources and not self.error)


def tavily_key_configured(api_key: str | None = None) -> bool:
    return bool(api_key or os.getenv("TAVILY_API_KEY"))


def should_use_tavily(route: RouteDecision) -> bool:
    return (
        route.knowledge_route in {"internet", "research papers"}
        and route.intent not in TAVILY_DENIED_INTENTS
    )


def _request_payload(query: str, *, research: bool = False) -> dict:
    normalized = query.lower()
    is_news = any(marker in normalized for marker in NEWS_MARKERS)
    payload = {
        "query": query,
        "search_depth": "advanced",
        "include_answer": "advanced",
        "include_raw_content": False,
        "max_results": 5,
        "topic": "news" if is_news else "general",
    }
    if research:
        payload["max_results"] = 8
        payload["include_domains"] = list(ACADEMIC_DOMAINS)
    if is_news:
        payload["time_range"] = "week"
    return payload


def _create_tavily_search(payload: dict, api_key: str) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        TAVILY_SEARCH_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def _confidence(sources: tuple[TavilySource, ...]) -> str:
    if not sources:
        return "Low"
    average_score = sum(source.score for source in sources) / len(sources)
    if len(sources) >= 3 and average_score >= 0.7:
        return "High"
    if len(sources) >= 2 and average_score >= 0.45:
        return "Medium"
    return "Low"


def _parse_sources(payload: dict) -> tuple[TavilySource, ...]:
    sources: list[TavilySource] = []
    seen: set[str] = set()
    for result in payload.get("results", []):
        title = str(result.get("title") or "").strip()
        url = str(result.get("url") or "").strip()
        if not title or not url or url in seen:
            continue
        seen.add(url)
        try:
            score = float(result.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        excerpt = " ".join(str(result.get("content") or "").split())
        sources.append(
            TavilySource(
                title=title,
                url=url,
                score=score,
                excerpt=excerpt[:500],
            )
        )
    return tuple(sources)


def format_tavily_answer(
    result: TavilyResult,
    *,
    research: bool = False,
    advice: bool = False,
) -> str:
    if research:
        source_lines = []
        for source in result.sources:
            detail = (
                f"  \n**Evidence excerpt:** {source.excerpt}"
                if source.excerpt
                else ""
            )
            source_lines.append(
                f"- [{source.title}]({source.url}){detail}"
            )
        return (
            f"**Research summary**\n\n{result.summary}\n\n"
            "**Academic sources reviewed**\n"
            + "\n".join(source_lines)
            + "\n\nThis is a focused search, not a complete systematic review."
        )
    source_lines = "\n".join(
        f"- [{source.title}]({source.url})" for source in result.sources
    )
    if advice:
        return (
            "It makes sense to want a clear, practical next step.\n\n"
            f"**Practical guidance**\n\n{result.summary}\n\n"
            "**Why this guidance:** It is based on the current sources listed below, "
            "rather than an unsupported guess.\n\n"
            f"**Sources**\n{source_lines}"
        )
    return f"**Summary**\n\n{result.summary}\n\n**Sources**\n{source_lines}"


def search_tavily(
    query: str,
    *,
    api_key: str | None = None,
    research: bool = False,
) -> TavilyResult:
    resolved_key = api_key or os.getenv("TAVILY_API_KEY")
    if not resolved_key:
        return TavilyResult(
            summary="",
            sources=(),
            confidence="Low",
            error="Tavily is not configured.",
        )

    try:
        payload = _create_tavily_search(
            _request_payload(query, research=research),
            resolved_key,
        )
        summary = str(payload.get("answer") or "").strip()
        sources = _parse_sources(payload)
        if not summary or not sources:
            return TavilyResult(
                summary="",
                sources=(),
                confidence="Low",
                error="Tavily returned no verifiable summary and sources.",
            )
        return TavilyResult(
            summary=summary,
            sources=sources,
            confidence=_confidence(sources),
        )
    except Exception as exc:
        return TavilyResult(
            summary="",
            sources=(),
            confidence="Low",
            error=f"{type(exc).__name__}: {exc}",
        )
