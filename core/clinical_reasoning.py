"""Non-diagnostic clinical reasoning helpers for Noema.

This module generates hypotheses, not conclusions. It is meant to help Noema
explain that several patterns can produce similar experiences, while keeping
diagnosis and assessment with qualified professionals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PossibleExplanation:
    label: str
    confidence: str
    reason: str


STYLE_BY_INTENT = {
    "decision support": "decision-focused",
    "practical advice": "practical",
    "intervention_request": "psychoeducational",
    "research paper question": "psychoeducational",
    "current factual search": "practical",
    "identity_exploration": "narrative",
    "achievement_self_worth": "reflective",
    "existential_question": "narrative",
    "ethical_dilemma": "decision-focused",
    "structured_problem_solving": "coaching",
    "failed_intervention_repair": "coaching",
    "user_frustration_repair": "reflective",
    "conversation_continuity": "reflective",
    "cognitive challenge": "reflective",
    "grief": "reflective",
    "venting": "reflective",
    "casual conversation": "exploratory",
}

STYLE_SEQUENCE = (
    "reflective",
    "practical",
    "psychoeducational",
    "exploratory",
    "narrative",
    "coaching",
    "decision-focused",
)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    normalized = " ".join(text.lower().split())
    return any(marker in normalized for marker in markers)


def select_response_style(
    *,
    intent: str,
    emotion: str,
    last_5_styles: list[str],
) -> str:
    """Pick a response style while avoiding repetition."""
    preferred = STYLE_BY_INTENT.get(intent)
    if preferred and preferred not in last_5_styles[-2:]:
        return preferred
    if emotion in {"sadness", "grief", "loneliness"}:
        candidates = ("reflective", "narrative", "practical")
    elif emotion in {"anxiety", "overwhelm"}:
        candidates = ("practical", "coaching", "psychoeducational")
    else:
        candidates = STYLE_SEQUENCE
    for style in candidates:
        if style not in last_5_styles:
            return style
    for style in STYLE_SEQUENCE:
        if style not in last_5_styles[-2:]:
            return style
    return preferred or "reflective"


def update_style_history(last_5_styles: list[str], style: str) -> None:
    if not style:
        return
    last_5_styles.append(style)
    if len(last_5_styles) > 5:
        del last_5_styles[:-5]


def possible_explanations(
    text: str,
    symptom_profile: dict[str, int],
) -> tuple[PossibleExplanation, ...]:
    """Return possible explanations for a user experience without diagnosing."""
    normalized = " ".join(text.lower().split())
    explanations: list[PossibleExplanation] = []

    focus_markers = (
        "can't focus",
        "cannot focus",
        "difficulty focusing",
        "hard to focus",
        "concentrate",
        "attention",
        "adhd",
    )
    if _contains_any(normalized, focus_markers):
        explanations.append(
            PossibleExplanation(
                "ADHD-like attention regulation",
                "low" if "adhd" not in normalized else "moderate",
                "Attention difficulty can overlap with ADHD, especially when it is long-standing and appears across school, work, home, and relationships.",
            )
        )
        explanations.append(
            PossibleExplanation(
                "sleep or exhaustion",
                "moderate",
                "Poor sleep, fatigue, or irregular routines can make attention feel much worse even without ADHD.",
            )
        )
        explanations.append(
            PossibleExplanation(
                "anxiety or overwhelm",
                "moderate",
                "Worry and overload can consume mental bandwidth, making focus feel unavailable.",
            )
        )
        explanations.append(
            PossibleExplanation(
                "depression-related low energy",
                "low" if symptom_profile.get("depressive_symptoms", 0) < 2 else "moderate",
                "Low mood, emptiness, and loss of energy can also show up as concentration problems.",
            )
        )

    if symptom_profile.get("burnout_indicators", 0) >= 2 or _contains_any(
        normalized,
        ("burnout", "burned out", "exhausted", "tired of everything"),
    ):
        explanations.append(
            PossibleExplanation(
                "burnout or chronic stress load",
                "moderate",
                "Sustained pressure can create exhaustion, cynicism, and reduced effectiveness.",
            )
        )

    if symptom_profile.get("grief_related_symptoms", 0) >= 2:
        explanations.append(
            PossibleExplanation(
                "grief-related emotional load",
                "moderate",
                "Loss can affect energy, sleep, concentration, motivation, and the sense of future.",
            )
        )

    if symptom_profile.get("anxiety_symptoms", 0) >= 2 and not any(
        item.label == "anxiety or overwhelm" for item in explanations
    ):
        explanations.append(
            PossibleExplanation(
                "anxiety or nervous-system alertness",
                "moderate",
                "When the body is on alert, the mind often scans for threat instead of resting on one task.",
            )
        )

    if symptom_profile.get("depressive_symptoms", 0) >= 2 and not any(
        item.label == "depression-related low energy" for item in explanations
    ):
        explanations.append(
            PossibleExplanation(
                "depression-related symptom overlap",
                "moderate",
                "Emptiness, worthlessness, fatigue, and low motivation can overlap with symptoms clinicians assess for depression.",
            )
        )

    return tuple(explanations[:5])


def explanation_dicts(
    explanations: tuple[PossibleExplanation, ...],
) -> list[dict[str, str]]:
    return [
        {
            "label": explanation.label,
            "confidence": explanation.confidence,
            "reason": explanation.reason,
        }
        for explanation in explanations
    ]
