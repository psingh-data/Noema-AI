"""Deterministic response critic for routing, advice, safety, and tone."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.router import RouteDecision


BANNED_THERAPY_PHRASES = (
    "what feels most present",
    "say a little more about that",
    "the effect on your daily life matters",
    "there is no need to solve this right away",
)

UNCERTAINTY_MARKERS = (
    "couldn't verify",
    "could not verify",
    "can't confirm",
    "cannot confirm",
    "don't want to guess",
    "do not want to guess",
    "uncertain",
    "live web access",
    "need live verification",
    "needs live verification",
    "will not invent",
    "retrieve and verify",
)

ADVICE_CONTENT_MARKERS = (
    "try ",
    "start ",
    "first",
    "practical",
    "plan",
    "recommend",
    "next step",
    "1.",
)

DECISION_CONTENT_MARKERS = (
    "compare",
    "option",
    "tradeoff",
    "trade-off",
    "benefit",
    "cost",
    "risk",
    "value",
    "against",
    "versus",
    "fits",
    "fit",
    "whereas",
    "while",
)

GENERIC_NEXT_STEP_MARKERS = (
    "recommend",
    "next step",
    "start",
    "try",
    "choose",
    "prefer",
    "plan",
)

GRIEF_VALIDATION_MARKERS = (
    "grief",
    "missing",
    "miss",
    "loss",
    "grandfather",
    "not a sign",
    "makes sense",
)

THERAPY_SUPPORT_MARKERS = (
    "therapist",
    "therapy",
    "counseling",
    "counselling",
    "support",
)

THERAPY_CONDITION_MARKERS = (
    "sleep",
    "work",
    "study",
    "relationships",
    "daily",
    "functioning",
    "unbearable",
    "intense",
    "distress",
    "stuck",
)

WORKPLACE_DECISION_MARKERS = (
    "do not recommend quitting impulsively",
    "not recommend quitting impulsively",
    "before resigning",
    "still employed",
)

BUSINESS_DECISION_MARKERS = (
    "do not quit impulsively",
    "quitting tomorrow",
    "runway",
    "validation",
    "paying customers",
    "customer feedback",
    "repeated revenue",
)

DOCUMENTATION_MARKERS = (
    "document",
    "records",
    "promotion criteria",
    "written",
    "achievements",
    "feedback",
)

JOB_MARKET_MARKERS = (
    "resume",
    "job market",
    "other roles",
    "other companies",
    "exit path",
    "applications",
)

HEALTH_SAFETY_MARKERS = (
    "doctor",
    "clinician",
    "qualified",
    "professional",
    "medical",
    "assessment",
)

RELATIONSHIP_MARKERS = (
    "relationship",
    "partner",
    "girlfriend",
    "fighting",
    "communication",
    "communicate",
    "boundary",
    "boundaries",
    "observe",
    "repair",
    "compatibility",
)

ABSOLUTE_CLAIM_MARKERS = (
    "definitely",
    "always",
    "never",
    "they are",
    "you must",
)

MIXED_BUCKET_MARKERS = (
    "bucket",
    "separate",
    "emotional load",
    "main track",
    "side experiment",
)

ADVICE_REQUEST_MARKERS = (
    "what should i do",
    "suggest something",
    "suggest me",
    "give me advice",
    "recommend",
    "any advice",
    "any ideas",
    "what to do",
    "how can i",
    "how do i",
    "need steps",
    "be practical",
)

DECISION_REQUEST_MARKERS = (
    "should i",
    "which should i",
    "help me decide",
    "help me choose",
    "confused between",
    "choose between",
    "decide between",
    "torn between",
    "whether i should",
    "is it the right move",
    "is this the right move",
    "should i do it",
    "if you were me",
    "would you",
)

GENERIC_DECISION_PHRASES = (
    "there is a real decision here, and i do not want to invent details",
    "tell me the options",
    "once the choice is clear",
    "is there a specific choice connected to this",
)

DISTORTION_MARKERS = (
    "i am a failure because",
    "i feel like a failure because",
    "failed once so",
    "failed two interviews",
    "not smart enough",
    "i will never",
    "everyone",
    "everyone my age is ahead",
    "nobody",
    "my life is ruined",
    "life is already ruined",
    "my whole future is ruined",
    "i always",
)

TOPIC_MARKERS = {
    "grief": ("miss my", "grandfather", "passed away", "died", "grief"),
    "education": ("cognitive science", "data science", "university", "study"),
    "workplace": ("boss", "company", "promotion", "promotes", "hr", "quit"),
    "career": ("career", "job", "interview", "resume"),
    "business": ("business", "startup", "brand", "customers"),
    "relationship": ("girlfriend", "boyfriend", "partner", "relationship", "fighting"),
    "self_esteem": ("failure", "failed", "worthless", "not good enough"),
}

TOPIC_ALIASES = {
    "career": "workplace",
    "finance": "general",
    "family": "relationship",
}


@dataclass(frozen=True)
class CriticResult:
    passed: bool
    failures: tuple[str, ...]
    question_count: int


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    normalized = " ".join(text.lower().split())
    return any(marker in normalized for marker in markers)


def _topic_detected(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(
        marker in normalized
        for markers in TOPIC_MARKERS.values()
        for marker in markers
    )


def _topic(route: RouteDecision) -> str:
    return TOPIC_ALIASES.get(getattr(route, "topic", "general"), getattr(route, "topic", "general"))


def _has_recommendation_or_next_step(text: str) -> bool:
    return _has_any(text, GENERIC_NEXT_STEP_MARKERS)


def _named_options(user_input: str) -> tuple[str, str] | None:
    normalized = " ".join(user_input.strip().split())
    patterns = (
        r"(?:should i|whether i should)\s+(.+?)\s+or\s+(.+?)[?.]?$",
        r"between\s+(.+?)\s+and\s+(.+?)[?.]?$",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            left = re.sub(
                r"^(study|choose|pick|take|go to)\s+",
                "",
                match.group(1).strip(" ?.!"),
                flags=re.IGNORECASE,
            )
            right = re.sub(
                r"^(study|choose|pick|take|go to)\s+",
                "",
                match.group(2).strip(" ?.!"),
                flags=re.IGNORECASE,
            )
            return left, right
    return None


def _decision_failures_by_topic(
    *,
    user_input: str,
    response: str,
    route: RouteDecision,
) -> list[str]:
    lowered = response.lower()
    topic = _topic(route)
    failures: list[str] = []

    if route.intent == "mixed complex life problem":
        if not _has_any(lowered, MIXED_BUCKET_MARKERS):
            failures.append("mixed complex response did not identify multiple buckets")
        if "priority" not in lowered and "prioritize" not in lowered:
            failures.append("mixed complex response did not prioritize")
        if not _has_recommendation_or_next_step(lowered):
            failures.append("mixed complex response did not give a practical next step")
        return failures

    if topic == "grief":
        if not _has_any(lowered, GRIEF_VALIDATION_MARKERS):
            failures.append("grief decision response did not validate grief")
        if not _has_any(lowered, THERAPY_SUPPORT_MARKERS):
            failures.append("grief decision response did not mention therapy/counseling/support")
        if not _has_any(lowered, THERAPY_CONDITION_MARKERS):
            failures.append("grief decision response did not explain when therapy may help")
        if not _has_recommendation_or_next_step(lowered):
            failures.append("grief decision response did not give a gentle recommendation")
        return failures

    if topic in {"workplace", "career"}:
        if not _has_any(lowered, WORKPLACE_DECISION_MARKERS):
            failures.append("workplace decision response may encourage impulsive quitting")
        if not _has_any(lowered, DOCUMENTATION_MARKERS):
            failures.append("workplace decision response did not suggest documentation or criteria")
        if not _has_any(lowered, JOB_MARKET_MARKERS):
            failures.append("workplace decision response did not suggest exploring options/job market")
        if not _has_recommendation_or_next_step(lowered):
            failures.append("workplace decision response did not give a practical next step")
        return failures

    if topic == "business":
        if not _has_any(lowered, BUSINESS_DECISION_MARKERS):
            failures.append("business decision response did not assess runway, validation, or impulsive quitting risk")
        if not _has_recommendation_or_next_step(lowered):
            failures.append("business decision response did not give a practical next step")
        return failures

    if topic == "education":
        options = _named_options(user_input)
        if options and not all(option.lower() in lowered for option in options):
            failures.append("education decision response did not mention both named options")
        if not any(marker in lowered for marker in DECISION_CONTENT_MARKERS):
            failures.append("education decision response did not compare fields/options")
        if not _has_recommendation_or_next_step(lowered):
            failures.append("education decision response did not give criteria or a recommendation")
        return failures

    if topic == "health":
        if "diagnos" in lowered:
            failures.append("health decision response implied diagnosis")
        if not _has_any(lowered, HEALTH_SAFETY_MARKERS):
            failures.append("health decision response did not suggest professional help when appropriate")
        if not _has_recommendation_or_next_step(lowered):
            failures.append("health decision response did not give a safe practical next step")
        return failures

    if topic == "relationship":
        if not _has_any(lowered, RELATIONSHIP_MARKERS):
            failures.append("relationship decision response did not acknowledge relationship context")
        if not _has_any(lowered, ("communicat", "boundar", "repair", "pattern", "compatib")):
            failures.append("relationship decision response did not discuss communication, boundaries, repair, or patterns")
        if _has_any(lowered, ABSOLUTE_CLAIM_MARKERS):
            failures.append("relationship decision response made an overly absolute claim")
        if not _has_recommendation_or_next_step(lowered):
            failures.append("relationship decision response did not give a practical next step")
        return failures

    if not any(marker in lowered for marker in DECISION_CONTENT_MARKERS):
        failures.append("decision request did not compare options or tradeoffs")
    if not _has_recommendation_or_next_step(lowered):
        failures.append("decision request did not include a recommendation or next step")
    return failures


def critique_response(
    *,
    user_input: str,
    response: str,
    route: RouteDecision,
    internet_used: bool = False,
    research_used: bool = False,
    safety_used: bool = False,
) -> CriticResult:
    lowered = response.lower()
    failures: list[str] = []
    question_count = response.count("?")
    asks_advice = _has_any(user_input, ADVICE_REQUEST_MARKERS)
    asks_decision = _has_any(user_input, DECISION_REQUEST_MARKERS) and (
        " or " in user_input.lower()
        or "between" in user_input.lower()
        or route.intent == "decision support"
    )

    if asks_advice and not any(marker in lowered for marker in ADVICE_CONTENT_MARKERS):
        failures.append("explicit advice request did not receive practical advice")
    if asks_decision or route.intent == "mixed complex life problem":
        failures.extend(
            _decision_failures_by_topic(
                user_input=user_input,
                response=response,
                route=route,
            )
        )
    if (asks_advice or asks_decision) and question_count > 1:
        failures.append("advice or decision response asks more than one question")
    if asks_decision and _topic_detected(user_input) and _has_any(
        response,
        GENERIC_DECISION_PHRASES,
    ):
        failures.append("topic-specific decision request received a generic decision template")
    if asks_decision and " or " in user_input.lower() and _has_any(
        response,
        ("tell me the options", "name the options"),
    ):
        failures.append("response asks for options that were already named")
    if _has_any(user_input, DISTORTION_MARKERS) and not _has_any(
        response,
        (
            "overgeneralization",
            "catastrophizing",
            "mind reading",
            "all-or-nothing",
            "one event",
            "not your whole",
        ),
    ):
        failures.append("obvious cognitive distortion was not addressed")

    found_banned = [
        phrase for phrase in BANNED_THERAPY_PHRASES if phrase in lowered
    ]
    if found_banned:
        failures.append("overused therapy language: " + ", ".join(found_banned))

    retrieval_uncertain = any(marker in lowered for marker in UNCERTAINTY_MARKERS)
    if route.knowledge_route == "internet" and not internet_used and not retrieval_uncertain:
        failures.append("current facts required internet or an explicit uncertainty notice")
    if (
        route.knowledge_route == "research papers"
        and not research_used
        and not retrieval_uncertain
        and "foundational" not in lowered
    ):
        failures.append("research request lacked retrieval or an uncertainty notice")
    if route.knowledge_route not in {"internet", "research papers"} and (
        route.intent
        in {"grief", "venting", "emotional reflection", "casual conversation"}
        and (internet_used or research_used)
    ):
        failures.append("pure emotional or casual support should not use retrieval")
    if route.intent == "crisis / safety" and not safety_used:
        failures.append("crisis language did not activate the safety system")
    if route.intent == "crisis / safety" and not _has_any(
        user_input,
        (
            "kill myself",
            "end my life",
            "take my life",
            "suicidal",
            "want to die",
            "wish i was dead",
            "wish i were dead",
            "hurt myself",
            "harm myself",
            "self harm",
            "feel like dying",
            "life is not worth living",
            "rather be dead",
            "feel unsafe with myself",
            "disappear forever",
            "nobody would care if i disappeared",
            "might do something bad to myself",
        ),
    ):
        failures.append("crisis mode activated without direct safety evidence")

    return CriticResult(
        passed=not failures,
        failures=tuple(failures),
        question_count=question_count,
    )


def _limit_questions(response: str) -> str:
    seen = 0
    parts = re.split(r"(?<=[.!?])\s+", response)
    kept: list[str] = []
    for part in parts:
        if "?" in part:
            seen += part.count("?")
            if seen > 1:
                continue
        kept.append(part)
    return " ".join(kept).strip()


def repair_response(
    *,
    user_input: str,
    response: str,
    route: RouteDecision,
    failures: tuple[str, ...],
) -> str:
    """Repair once with a strict deterministic structure."""
    lowered_failures = " ".join(failures)
    topic = _topic(route)
    if route.intent == "mixed complex life problem" or topic in {
        "grief",
        "workplace",
        "career",
        "education",
        "health",
        "relationship",
        "business",
    }:
        repaired = response
        if "more than one question" in lowered_failures:
            repaired = _limit_questions(repaired)
        if repaired.strip():
            return repaired
    if "practical advice" in lowered_failures:
        return (
            "It makes sense that you want a concrete way forward.\n\n"
            "**Start here:** write the problem in one sentence, choose the smallest "
            "reversible action available today, and set a short time to test it. If "
            "the choice affects health, safety, money, or legal rights, verify the "
            "facts with a qualified professional before committing.\n\n"
            "**Why:** a reversible step lowers pressure while giving you real "
            "information for the next decision.\n\n"
            "What constraint matters most right now?"
        )
    if "compare options" in lowered_failures:
        return (
            "Compare the options on the same five points: "
            "benefits, costs, risks, fit with your values, and how reversible each "
            "path is. Prefer the option that fits your priorities and has acceptable "
            "downsides, not simply the one that removes anxiety fastest.\n\n"
            "Which tradeoff matters most to you?"
        )

    repaired = response
    replacements = {
        "What feels most present for you?": "What part matters most right now?",
        "Say a little more about that.": "What happened next?",
        "The effect on your daily life matters here.": (
            "It may help to notice whether this is changing your day-to-day life."
        ),
        "There is no need to solve this right away.": "We can take this one step at a time.",
    }
    for phrase, replacement in replacements.items():
        repaired = repaired.replace(phrase, replacement)
    if "more than one question" in lowered_failures:
        repaired = _limit_questions(repaired)
    return repaired
