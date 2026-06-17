"""Shared dataset normalization helpers for Noema's pre-training pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


SCHEMA_FIELDS = (
    "id",
    "dataset_source",
    "user_input",
    "target_intent",
    "target_emotion",
    "target_mode",
    "emotion_intensity",
    "should_use_internet",
    "should_use_research",
    "should_use_safety",
    "bad_response_trap",
    "ideal_response",
    "response_requirements",
    "details_panel_expected",
    "split",
)

ADVICE_PHRASES = (
    "what should i do",
    "suggest something",
    "suggest me",
    "give me advice",
    "recommend",
    "any advice",
    "any ideas",
    "how can i",
    "how do i",
)

DECISION_PHRASES = (
    "should i",
    "is this the right move",
    "help me decide",
    "help me choose",
    "choose between",
    "decide between",
    "torn between",
    "whether i should",
)

INTENT_ALIASES = {
    "practical_advice": "advice",
    "practical advice": "advice",
    "decision support": "decision_support",
    "emotional reflection": "emotional_reflection",
    "current facts": "current_facts",
    "current factual search": "current_facts",
    "research paper question": "research",
    "crisis / safety": "crisis_safety",
    "casual conversation": "casual",
    "anxiety / stress": "emotional_reflection",
    "overwhelm": "emotional_reflection",
    "general conversation": "casual",
    "general knowledge": "current_facts",
    "career / education": "current_facts",
    "health / wellness information": "current_facts",
    "acute_grief_long_support": "grief",
    "ongoing_grief_long_support": "grief",
    "business_decision_long": "decision_support",
    "career_decision_long": "decision_support",
    "education_decision_long": "decision_support",
    "relationship_decision_long": "decision_support",
    "workplace_discrimination_long": "workplace_discrimination",
    "cognitive_distortion_long": "cognitive_challenge",
    "casual_long_but_natural": "casual",
    "therapy_recommendation": "intervention_request",
}

MODE_BY_INTENT = {
    "advice": "Give me advice",
    "decision_support": "Help me make a decision",
    "grief": "Help me understand my feelings",
    "venting": "Just listen",
    "emotional_reflection": "Help me understand my feelings",
    "current_facts": "Research Assistant",
    "research": "Research Assistant",
    "crisis_safety": "Safety",
    "casual": "Friend",
    "cognitive_challenge": "Challenge my thinking",
    "workplace_discrimination": "Give me advice",
    "response_quality": "Response quality",
    "emotion_detection": "Emotion classification",
    "identity_exploration": "Help me understand my feelings",
    "achievement_self_worth": "Challenge my thinking",
    "existential_question": "Reflective conversation",
    "ethical_dilemma": "Structured reflection",
    "structured_problem_solving": "Give me advice",
    "intervention_request": "Research Assistant",
    "failed_intervention_repair": "Give me advice",
    "user_frustration_repair": "Repair conversation",
    "conversation_continuity": "Conversation continuity",
}

GOEMOTIONS_MAP = {
    "grief": "grief",
    "sadness": "sadness",
    "fear": "anxiety",
    "nervousness": "anxiety",
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "joy": "positive",
    "excitement": "positive",
    "optimism": "hopeful",
    "neutral": "neutral",
    "embarrassment": "shame",
    "remorse": "guilt",
    "caring": "connection",
    "love": "connection",
}

EMOTION_ALIASES = {
    "afraid": "anxiety",
    "anxious": "anxiety",
    "terrified": "anxiety",
    "furious": "anger",
    "annoyed": "anger",
    "devastated": "grief",
    "sentimental": "connection",
    "nostalgic": "connection",
    "hope": "hopeful",
    "joyful": "positive",
    "happy": "positive",
    "ashamed": "shame",
    "guilty": "guilt",
    "lonely": "loneliness",
}


def stable_id(source: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:20]
    return f"{source}-{digest}"


def deterministic_split(value: str) -> str:
    bucket = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    if not value:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [clean_text(value)]
        return as_list(parsed)
    return [clean_text(value)]


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def canonical_intent(value: Any) -> str:
    normalized = clean_text(value).lower().replace("-", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    return INTENT_ALIASES.get(normalized, normalized or "casual")


def priority_intent(user_input: str, original_intent: str) -> tuple[str, float]:
    """Apply the explicit advice/decision override before reflection labels."""
    text = clean_text(user_input).lower()
    named_decision = (
        any(phrase in text for phrase in DECISION_PHRASES)
        and (" or " in text or "between" in text or "right move" in text)
    )
    if named_decision:
        return "decision_support", 2.0
    if any(phrase in text for phrase in ADVICE_PHRASES):
        return "advice", 2.0
    if any(phrase in text for phrase in DECISION_PHRASES):
        return "decision_support", 2.0
    return canonical_intent(original_intent), 1.0


def map_emotion(labels: Iterable[str] | str) -> str:
    if isinstance(labels, str):
        labels = [labels]
    mapped: list[str] = []
    for label in labels:
        normalized = clean_text(label).lower()
        mapped_label = GOEMOTIONS_MAP.get(
            normalized,
            EMOTION_ALIASES.get(normalized, normalized),
        )
        if mapped_label:
            mapped.append(mapped_label)
    if not mapped:
        return "neutral"
    priority = (
        "grief",
        "anxiety",
        "anger",
        "sadness",
        "shame",
        "guilt",
        "loneliness",
        "connection",
        "hopeful",
        "positive",
        "neutral",
    )
    return next((label for label in priority if label in mapped), mapped[0])


def normalized_record(
    *,
    record_id: str,
    dataset_source: str,
    user_input: str,
    target_intent: str = "",
    target_emotion: str = "",
    target_mode: str = "",
    emotion_intensity: str = "",
    should_use_internet: bool = False,
    should_use_research: bool = False,
    should_use_safety: bool = False,
    bad_response_trap: str = "",
    ideal_response: str = "",
    response_requirements: Iterable[str] = (),
    details_panel_expected: dict[str, Any] | None = None,
    split: str = "train",
    priority_weight: float = 1.0,
    training_eligible: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    intent = canonical_intent(target_intent)
    research_flag = bool(should_use_research)
    return {
        "id": clean_text(record_id),
        "dataset_source": clean_text(dataset_source),
        "user_input": clean_text(user_input),
        "target_intent": intent,
        "target_emotion": clean_text(target_emotion).lower(),
        "target_mode": clean_text(target_mode) or MODE_BY_INTENT.get(intent, ""),
        "emotion_intensity": clean_text(emotion_intensity).lower(),
        "should_use_internet": bool(should_use_internet) or research_flag,
        "should_use_research": research_flag,
        "should_use_safety": bool(should_use_safety),
        "bad_response_trap": clean_text(bad_response_trap),
        "ideal_response": clean_text(ideal_response),
        "response_requirements": [
            clean_text(item) for item in response_requirements if clean_text(item)
        ],
        "details_panel_expected": details_panel_expected or {},
        "split": split if split in {"train", "validation", "test"} else "train",
        "priority_weight": float(priority_weight),
        "training_eligible": bool(training_eligible),
        "metadata": metadata or {},
    }


def last_dialogue_turn(transcript: str, speaker: str) -> str:
    pattern = rf"(?:^|\n\n){re.escape(speaker)}:\s*(.*?)(?=\n\n(?:Human|Assistant):|\Z)"
    matches = re.findall(pattern, transcript or "", flags=re.DOTALL)
    return clean_text(matches[-1]) if matches else ""


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                yield value
