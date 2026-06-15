"""Transparent, rule-based emotion and intensity estimates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class EmotionResult:
    emotion: str
    intensity: str
    confidence: float


EMOTION_KEYWORDS = {
    "grief": (
        "died",
        "death",
        "grief",
        "funeral",
        "passed away",
        "lost my",
        "miss my",
        "missing my",
        "memory of my",
    ),
    "anxiety": ("anxious", "worried", "nervous", "panic", "afraid", "uncertain"),
    "overwhelm": (
        "overwhelmed",
        "too much",
        "can't cope",
        "cannot cope",
        "can't handle",
        "everything feels impossible",
    ),
    "sadness": ("sad", "unhappy", "empty", "crying", "hopeless", "down"),
    "anger": ("angry", "furious", "annoyed", "frustrated", "resentful"),
    "loneliness": ("lonely", "alone", "isolated", "no one understands"),
    "confusion": ("confused", "don't know", "do not know", "torn", "unsure"),
    "hope": ("hopeful", "optimistic", "looking forward", "getting better"),
    "motivation": ("motivated", "determined", "ready to", "inspired"),
}

HIGH_INTENSITY_MARKERS = (
    "extremely",
    "unbearable",
    "completely",
    "impossible",
    "can't take it",
    "cannot take it",
    "out of control",
    "breaking down",
)

MEDIUM_INTENSITY_MARKERS = ("very", "really", "so ", "constantly", "exhausted")


@lru_cache(maxsize=1)
def _dataset_lexicon() -> dict[str, dict[str, float]]:
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "training"
        / "emotion_lexicon.json"
    )
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    lexicon: dict[str, dict[str, float]] = {}
    for emotion, entries in payload.get("emotions", {}).items():
        lexicon[emotion] = {
            str(entry["token"]): float(entry["score"])
            for entry in entries
            if entry.get("token")
        }
    return lexicon


def detect_emotion(text: str) -> EmotionResult:
    normalized = " ".join(text.lower().split())
    scores = {
        emotion: sum(1 for keyword in keywords if keyword in normalized)
        for emotion, keywords in EMOTION_KEYWORDS.items()
    }
    emotion, score = max(scores.items(), key=lambda item: item[1], default=("neutral", 0))

    if score == 0:
        tokens = set(re.findall(r"[a-z']+", normalized))
        learned_scores = {
            label: sum(weights.get(token, 0.0) for token in tokens)
            for label, weights in _dataset_lexicon().items()
            if label != "neutral"
        }
        learned_emotion, learned_score = max(
            learned_scores.items(),
            key=lambda item: item[1],
            default=("neutral", 0.0),
        )
        if learned_score >= 2.0:
            emotion = learned_emotion
            score = 1
        else:
            emotion = "neutral"

    marker_count = sum(marker in normalized for marker in HIGH_INTENSITY_MARKERS)
    if marker_count or score >= 3:
        intensity = "high"
    elif score >= 2 or any(marker in normalized for marker in MEDIUM_INTENSITY_MARKERS):
        intensity = "medium"
    else:
        intensity = "low"

    confidence = 0.25 if score == 0 else min(0.95, 0.45 + (score * 0.15))
    return EmotionResult(emotion=emotion, intensity=intensity, confidence=confidence)
