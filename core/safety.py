"""Safety screening.

This module intentionally runs before every other analyzer. Keyword rules are
only a conservative first layer for the Phase 1 prototype, not a clinical risk
assessment.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.crisis_resources import crisis_response


@dataclass(frozen=True)
class SafetyResult:
    level: str
    matched_phrases: tuple[str, ...]

    @property
    def is_crisis(self) -> bool:
        return self.level in {"concern", "immediate"}


IMMEDIATE_PHRASES = (
    "kill myself",
    "end my life",
    "take my life",
    "suicide plan",
    "going to die tonight",
    "hurt myself now",
    "about to kill myself",
    "going to kill myself",
    "ready to die",
    "might do something bad to myself tonight",
)

CONCERN_PHRASES = (
    "want to die",
    "don't want to live",
    "do not want to live",
    "wish i was dead",
    "wish i were dead",
    "hurt myself",
    "hurting myself",
    "harm myself",
    "harming myself",
    "feel like hurting myself",
    "feel like harming myself",
    "self harm",
    "suicidal",
    "no reason to live",
    "better off without me",
    "feel like dying",
    "feeling like dying",
    "life is not worth living",
    "life isn't worth living",
    "rather be dead",
    "can't go on",
    "cannot go on",
    "want to disappear forever",
    "nobody would care if i disappeared",
    "feel unsafe with myself",
    "unsafe with myself",
    "don't want to be here anymore",
    "do not want to be here anymore",
)

INTENT_PHRASES = (
    "i want to",
    "i am going to",
    "i'm going to",
    "i plan to",
    "planning to",
    "i might",
    "i may",
    "about to",
    "ready to",
    "tonight",
    "today",
    "now",
    "soon",
)

SELF_HARM_OBJECTS = (
    "kill myself",
    "end my life",
    "take my life",
    "suicide",
    "hurt myself",
    "harm myself",
    "self harm",
    "self-harm",
)

AMBIGUOUS_CONCERN_PHRASES = (
    "can't go on",
    "cannot go on",
    "don't want to be here anymore",
    "do not want to be here anymore",
    "want to disappear forever",
    "nobody would care if i disappeared",
    "feel like dying",
    "feeling like dying",
)

ORDINARY_CONTEXT_MARKERS = (
    "with this assignment",
    "with this exam",
    "with this project",
    "with this job",
    "with this class",
    "with it",
    "at this party",
    "at this meeting",
    "in this class",
    "in this city",
    "in this job",
    "on this app",
    "watching this",
)

NON_CRISIS_CONTEXT_MARKERS = (
    "grandfather",
    "grandmother",
    "grief",
    "died",
    "passed away",
    "career",
    "job",
    "university",
    "study",
    "business",
    "relationship",
    "girlfriend",
    "boyfriend",
    "partner",
    "who i am",
    "identity",
    "adhd",
    "focus",
    "free will",
    "point of life",
    "meaning of life",
    "existential",
)

NEGATED_PHRASES = (
    "not suicidal",
    "not going to kill myself",
    "don't want to die",
    "do not want to die",
    "never wanted to die",
    "no thoughts of suicide",
    "not about suicide",
    "not self harm",
    "not self-harm",
    "i mean this place",
)


def _has_actual_self_harm_intent(text: str) -> bool:
    return any(marker in text for marker in SELF_HARM_OBJECTS) and any(
        marker in text for marker in INTENT_PHRASES
    )


def assess_safety(text: str) -> SafetyResult:
    """Return a conservative safety classification for user-provided text."""
    normalized = " ".join(text.lower().split())
    screened = normalized
    for phrase in NEGATED_PHRASES:
        screened = screened.replace(phrase, " ")
    screened = " ".join(screened.split())
    ordinary_context = any(marker in screened for marker in ORDINARY_CONTEXT_MARKERS)
    non_crisis_context = any(marker in screened for marker in NON_CRISIS_CONTEXT_MARKERS)

    immediate = tuple(phrase for phrase in IMMEDIATE_PHRASES if phrase in screened)
    if immediate:
        return SafetyResult(level="immediate", matched_phrases=immediate)

    concern = tuple(
        phrase
        for phrase in CONCERN_PHRASES
        if phrase in screened
        and not (
            phrase in AMBIGUOUS_CONCERN_PHRASES
            and (ordinary_context or non_crisis_context)
            and not _has_actual_self_harm_intent(screened)
        )
    )
    if concern:
        return SafetyResult(level="concern", matched_phrases=concern)

    return SafetyResult(level="none", matched_phrases=())


def generate_crisis_response(level: str, country_code: str = "IN") -> str:
    """Provide direct, supportive guidance without pretending to be emergency care."""
    return crisis_response(
        country_code=country_code,
        immediate=level == "immediate",
    )
