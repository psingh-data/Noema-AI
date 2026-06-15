"""Select a response style from the analysis."""

from __future__ import annotations


def select_tone(emotion: str, category: str, intensity: str) -> str:
    if intensity == "high" or emotion == "overwhelm":
        return "short and grounding"
    if category == "grief" or emotion in {"grief", "sadness"}:
        return "gentle and compassionate"
    if emotion == "anxiety" or category == "anxiety":
        return "calm and reassuring"
    if category in {"career", "study", "decision making", "motivation"}:
        return "practical and encouraging"
    return "warm and reflective"

