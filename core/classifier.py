"""Rule-based life-category classification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryResult:
    category: str
    confidence: float


CATEGORY_KEYWORDS = {
    "grief": (
        "died",
        "death",
        "funeral",
        "passed away",
        "loss",
        "miss my",
        "missing my",
        "memory of my",
        "broke down",
    ),
    "anxiety": ("anxiety", "anxious", "panic", "worried"),
    "overwhelm": ("overwhelmed", "suffocated", "suffocating", "too much", "can't cope", "burnout"),
    "career": ("job", "career", "interview", "work", "boss", "promotion"),
    "study": ("exam", "college", "university", "assignment", "study", "grades"),
    "relationship": (
        "partner",
        "relationship",
        "breakup",
        "boyfriend",
        "girlfriend",
        "spouse",
    ),
    "health": ("health", "illness", "diagnosis", "pain", "doctor"),
    "self-esteem": (
        "worthless",
        "not good enough",
        "failure",
        "hate myself",
        "insecure",
    ),
    "decision making": ("decide", "decision", "choose", "choice", "torn between"),
    "motivation": ("motivation", "procrastinating", "can't start", "discipline"),
    "loneliness": ("lonely", "alone", "isolated", "no friends"),
    "personal growth": ("grow", "improve myself", "habit", "self awareness"),
}


def classify_category(text: str) -> CategoryResult:
    normalized = " ".join(text.lower().split())
    scores = {
        category: sum(1 for keyword in keywords if keyword in normalized)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    category, score = max(scores.items(), key=lambda item: item[1], default=("", 0))
    if score == 0:
        return CategoryResult(category="general reflection", confidence=0.25)
    return CategoryResult(category=category, confidence=min(0.95, 0.5 + score * 0.15))
