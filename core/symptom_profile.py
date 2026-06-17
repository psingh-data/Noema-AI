"""DSM-informed symptom overlap profile.

This module is a symptom-reference layer only. It is not a diagnostic criteria
engine and must never be used to state or imply that a user has a disorder.
"""

from __future__ import annotations

from dataclasses import dataclass


SYMPTOM_KEYS = (
    "depressive_symptoms",
    "anxiety_symptoms",
    "panic_symptoms",
    "trauma_related_symptoms",
    "grief_related_symptoms",
    "obsessive_symptoms",
    "attention_regulation_symptoms",
    "burnout_indicators",
)


SYMPTOM_TERMS = {
    "depressive_symptoms": (
        "sad",
        "low",
        "hopeless",
        "empty",
        "worthless",
        "guilty",
        "no interest",
        "lost interest",
        "nothing feels enjoyable",
        "can't get out of bed",
        "cannot get out of bed",
        "not eating",
        "sleeping too much",
    ),
    "anxiety_symptoms": (
        "anxious",
        "worried",
        "nervous",
        "on edge",
        "dread",
        "afraid",
        "overthinking",
        "can't relax",
        "cannot relax",
    ),
    "panic_symptoms": (
        "panic",
        "heart racing",
        "chest tight",
        "can't breathe",
        "cannot breathe",
        "dizzy",
        "shaking",
        "sweating",
        "feel like i am dying",
    ),
    "trauma_related_symptoms": (
        "flashback",
        "nightmare",
        "triggered",
        "hypervigilant",
        "trauma",
        "intrusive memory",
        "avoid reminders",
        "unsafe",
    ),
    "grief_related_symptoms": (
        "died",
        "death",
        "passed away",
        "grief",
        "loss",
        "funeral",
        "miss my",
        "missing my",
        "still cry",
        "mourning",
    ),
    "obsessive_symptoms": (
        "intrusive thought",
        "can't stop thinking",
        "cannot stop thinking",
        "obsessing",
        "compulsion",
        "checking repeatedly",
        "ritual",
        "repeating",
    ),
    "attention_regulation_symptoms": (
        "can't focus",
        "cannot focus",
        "distracted",
        "forgetful",
        "disorganized",
        "procrastinating",
        "procrastination",
        "adhd-like",
        "adhd like",
        "brain fog",
    ),
    "burnout_indicators": (
        "burnout",
        "burned out",
        "exhausted",
        "drained",
        "overworked",
        "no energy",
        "work stress",
        "cynical",
        "can't keep going",
        "cannot keep going",
    ),
}


OVERLAP_LABELS = {
    "depressive_symptoms": "depression-related symptom areas",
    "anxiety_symptoms": "anxiety-related symptom areas",
    "panic_symptoms": "panic-related symptom areas",
    "trauma_related_symptoms": "trauma-related symptom areas",
    "grief_related_symptoms": "grief-related symptom areas",
    "obsessive_symptoms": "obsessive or repetitive-thought symptom areas",
    "attention_regulation_symptoms": "attention-regulation symptom areas",
    "burnout_indicators": "burnout-related indicators",
}


@dataclass(frozen=True)
class ClinicalOverlap:
    area: str
    confidence: str
    score: int
    explanation: str


def empty_symptom_profile() -> dict[str, int]:
    return {key: 0 for key in SYMPTOM_KEYS}


def confidence_for_score(score: int) -> str:
    if score >= 3:
        return "high"
    if score == 2:
        return "moderate"
    return "low"


def build_symptom_profile(text: str) -> dict[str, int]:
    normalized = " ".join(text.lower().split())
    profile = empty_symptom_profile()
    for key, terms in SYMPTOM_TERMS.items():
        matches = sum(term in normalized for term in terms)
        profile[key] = min(4, matches)
    return profile


def merge_symptom_profiles(
    existing: dict[str, int],
    current: dict[str, int],
) -> dict[str, int]:
    merged = empty_symptom_profile()
    for key in SYMPTOM_KEYS:
        merged[key] = max(int(existing.get(key, 0)), int(current.get(key, 0)))
    return merged


def possible_clinical_overlaps(
    profile: dict[str, int],
) -> tuple[ClinicalOverlap, ...]:
    overlaps: list[ClinicalOverlap] = []
    for key in SYMPTOM_KEYS:
        score = int(profile.get(key, 0))
        if score <= 0:
            continue
        label = OVERLAP_LABELS[key]
        confidence = confidence_for_score(score)
        overlaps.append(
            ClinicalOverlap(
                area=label,
                confidence=confidence,
                score=score,
                explanation=(
                    f"Some wording overlaps with {label} clinicians may assess. "
                    "This is not a diagnosis."
                ),
            )
        )
    return tuple(
        sorted(overlaps, key=lambda overlap: overlap.score, reverse=True)
    )


def overlap_summary(overlaps: tuple[ClinicalOverlap, ...]) -> str:
    if not overlaps:
        return ""
    top = overlaps[:2]
    areas = " and ".join(overlap.area for overlap in top)
    guidance = (
        f"Some of the experiences you describe overlap with {areas}. "
        "That does not mean any condition is present; it means these are areas a "
        "clinician might ask about more carefully."
    )
    if any(overlap.confidence == "high" for overlap in top):
        guidance += (
            " Because the pattern sounds significant, a professional evaluation "
            "could help you understand what is happening and what support fits."
        )
    return guidance


def overlap_dicts(
    overlaps: tuple[ClinicalOverlap, ...],
) -> list[dict[str, str | int]]:
    return [
        {
            "area": overlap.area,
            "confidence": overlap.confidence,
            "score": overlap.score,
            "explanation": overlap.explanation,
        }
        for overlap in overlaps
    ]
