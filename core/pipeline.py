"""Single orchestration point for the complete Noema Phase 1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from core.bias_detector import BiasResult, detect_biases
from core.classifier import CategoryResult, classify_category
from core.emotion_detector import EmotionResult, detect_emotion
from core.response_generator import generate_response
from core.safety import SafetyResult, assess_safety, generate_crisis_response
from core.tone_controller import select_tone


@dataclass(frozen=True)
class ReflectionResult:
    response: str
    safety: SafetyResult
    emotion: EmotionResult
    category: CategoryResult
    biases: list[BiasResult]
    tone: str
    support_urgency: str


def process_reflection(text: str) -> ReflectionResult:
    """Analyze text safely and produce a supportive response."""
    safety = assess_safety(text)

    # Crisis content bypasses ordinary interpretation and advice.
    if safety.is_crisis:
        return ReflectionResult(
            response=generate_crisis_response(safety.level),
            safety=safety,
            emotion=EmotionResult("distress", "high", 0.8),
            category=CategoryResult("crisis / safety", 0.9),
            biases=[],
            tone="safety-first",
            support_urgency="urgent",
        )

    emotion = detect_emotion(text)
    category = classify_category(text)
    biases = detect_biases(text)
    tone = select_tone(emotion.emotion, category.category, emotion.intensity)
    response = generate_response(
        emotion=emotion.emotion,
        category=category.category,
        intensity=emotion.intensity,
        tone=tone,
        biases=biases,
    )
    support_urgency = "professional" if emotion.intensity == "high" else "routine"
    return ReflectionResult(
        response=response,
        safety=safety,
        emotion=emotion,
        category=category,
        biases=biases,
        tone=tone,
        support_urgency=support_urgency,
    )
