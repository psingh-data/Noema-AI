"""Select intent-specific examples without exposing the full dataset to prompts."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = PROJECT_ROOT / "data" / "processed" / "fewshot_examples.json"

RUNTIME_INTENT_MAP = {
    "practical advice": "advice",
    "decision support": "decision_support",
    "emotional reflection": "emotional_reflection",
    "current factual search": "current_facts",
    "research paper question": "research",
    "crisis / safety": "crisis_safety",
    "casual conversation": "casual",
    "mixed complex life problem": "mixed_complex_life_problem",
    "identity exploration": "identity_exploration",
    "achievement self worth": "achievement_self_worth",
    "existential question": "existential_question",
    "ethical dilemma": "ethical_dilemma",
    "structured problem solving": "structured_problem_solving",
    "intervention request": "intervention_request",
    "therapy recommendation": "intervention_request",
    "failed intervention repair": "failed_intervention_repair",
    "user frustration repair": "user_frustration_repair",
    "conversation continuity": "conversation_continuity",
}


def canonical_runtime_intent(intent: str) -> str:
    normalized = " ".join(intent.lower().replace("_", " ").split())
    return RUNTIME_INTENT_MAP.get(normalized, normalized.replace(" ", "_"))


@lru_cache(maxsize=2)
def load_fewshot_library(path: str = str(DEFAULT_LIBRARY)) -> dict[str, Any]:
    library_path = Path(path)
    if not library_path.exists():
        return {"version": 0, "intents": {}}
    return json.loads(library_path.read_text(encoding="utf-8"))


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", text.lower())
        if len(token) > 2
    }


def select_fewshot_examples(
    intent: str,
    user_input: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    canonical = canonical_runtime_intent(intent)
    group = load_fewshot_library().get("intents", {}).get(canonical, {})
    examples = list(group.get("strong_examples", []))
    if not examples:
        return []
    desired = limit if limit is not None else (5 if canonical == "decision_support" else 3)
    query_tokens = _tokens(user_input)

    def score(example: dict[str, Any]) -> tuple[float, float]:
        example_tokens = _tokens(str(example.get("user_input", "")))
        overlap = len(query_tokens & example_tokens) / max(len(query_tokens), 1)
        return overlap, float(example.get("priority_weight", 1.0))

    return sorted(examples, key=score, reverse=True)[:desired]


def expected_structure(intent: str) -> list[str]:
    canonical = canonical_runtime_intent(intent)
    group = load_fewshot_library().get("intents", {}).get(canonical, {})
    return list(group.get("expected_response_structure", []))


def format_examples_for_prompt(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "No reviewed few-shot examples are available for this intent."
    blocks = []
    for index, example in enumerate(examples, start=1):
        blocks.append(
            f"Example {index}\n"
            f"User: {example.get('user_input', '')}\n"
            f"Good response: {example.get('ideal_response', '')}"
        )
    return "\n\n".join(blocks)
