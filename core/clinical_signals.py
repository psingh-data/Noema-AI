"""Cross-cutting symptom signals for adaptive conversation.

The domains are informed by APA DSM-5-TR cross-cutting assessment areas and
WHO disability concepts. They guide follow-up questions only. They are not a
diagnostic criteria engine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClinicalSignal:
    domain: str
    score: int
    label: str


DOMAIN_TERMS = {
    "depressed mood": (
        "sad",
        "low",
        "hopeless",
        "empty",
        "worthless",
        "guilty",
        "no interest",
        "lost interest",
        "nothing feels enjoyable",
    ),
    "anxiety and fear": (
        "anxious",
        "worried",
        "panic",
        "afraid",
        "fear",
        "nervous",
        "on edge",
        "dread",
    ),
    "anger and irritability": (
        "angry",
        "furious",
        "irritable",
        "rage",
        "losing my temper",
        "frustrated",
    ),
    "elevated mood or activation": (
        "unusually energetic",
        "full of energy",
        "need less sleep",
        "don't need sleep",
        "do not need sleep",
        "racing thoughts",
        "talking very fast",
        "invincible",
        "reckless",
        "impulsive spending",
    ),
    "sleep": (
        "can't sleep",
        "cannot sleep",
        "insomnia",
        "sleeping too much",
        "oversleeping",
        "nightmares",
        "waking up",
        "poor sleep",
        "no sleep",
    ),
    "physical symptoms": (
        "headache",
        "stomach",
        "pain",
        "dizzy",
        "nausea",
        "heart racing",
        "chest tight",
        "physical symptoms",
    ),
    "concentration and memory": (
        "can't focus",
        "cannot focus",
        "concentrate",
        "forgetful",
        "memory",
        "brain fog",
        "can't decide",
        "cannot decide",
    ),
    "repetitive thoughts or behaviors": (
        "can't stop thinking",
        "cannot stop thinking",
        "intrusive thought",
        "obsessing",
        "compulsion",
        "checking repeatedly",
        "repeating",
        "ritual",
    ),
    "detachment or unreality": (
        "not real",
        "unreal",
        "outside my body",
        "detached from myself",
        "watching myself",
        "disconnected from reality",
        "lost time",
    ),
    "unusual perceptions or beliefs": (
        "hearing voices",
        "hear voices",
        "seeing things",
        "being watched",
        "people are following",
        "someone controls",
        "messages meant for me",
        "reading my thoughts",
    ),
    "substance use": (
        "alcohol",
        "drinking",
        "cannabis",
        "weed",
        "cocaine",
        "opioid",
        "pills",
        "drugs",
        "substance",
        "getting high",
    ),
    "relationships and sense of self": (
        "hate myself",
        "don't know who i am",
        "do not know who i am",
        "people always leave",
        "relationship",
        "abandoned",
        "unstable",
    ),
    "daily functioning": (
        "can't work",
        "cannot work",
        "can't study",
        "cannot study",
        "missing work",
        "missing class",
        "not eating",
        "not showering",
        "can't get out of bed",
        "cannot get out of bed",
        "not taking care of myself",
        "affecting my work",
        "affecting my studies",
        "affecting my life",
    ),
}

NEGATIONS = (
    "no ",
    "not ",
    "never ",
    "don't ",
    "do not ",
    "haven't ",
    "have not ",
    "isn't ",
    "is not ",
)


def _is_negated(text: str, phrase: str) -> bool:
    index = text.find(phrase)
    if index < 0:
        return False
    prefix = text[max(0, index - 45) : index]
    boundaries = (".", "!", "?", ";", ",")
    if any(mark in prefix for mark in boundaries):
        prefix = prefix[max(prefix.rfind(mark) for mark in boundaries) + 1 :]
    return any(marker in prefix for marker in NEGATIONS)


def detect_clinical_signals(text: str) -> list[ClinicalSignal]:
    normalized = " ".join(text.lower().split())
    signals: list[ClinicalSignal] = []

    for domain, terms in DOMAIN_TERMS.items():
        matches = [
            term
            for term in terms
            if term in normalized and not _is_negated(normalized, term)
        ]
        if matches:
            signals.append(
                ClinicalSignal(
                    domain=domain,
                    score=min(3, len(matches)),
                    label="present" if len(matches) == 1 else "prominent",
                )
            )

    return sorted(signals, key=lambda signal: signal.score, reverse=True)


def signal_names(signals: list[ClinicalSignal]) -> list[str]:
    return [signal.domain for signal in signals]
