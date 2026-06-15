"""Build a reviewed, intent-specific few-shot library from normalized data."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dataset_common import iter_jsonl


TARGET_INTENTS = (
    "grief",
    "advice",
    "decision_support",
    "workplace_discrimination",
    "cognitive_challenge",
    "venting",
    "casual",
    "current_facts",
    "research",
    "crisis_safety",
    "mixed_complex_life_problem",
)

EXPECTED_STRUCTURES = {
    "grief": [
        "Acknowledge the loss naturally.",
        "Explore what the person misses or is carrying.",
        "Do not rush into coping tips unless advice was requested.",
        "Ask at most one accessible question.",
    ],
    "advice": [
        "Validate briefly.",
        "Give practical suggestions immediately.",
        "Explain why the suggestions may help.",
        "Ask at most one clarifying question after the advice.",
    ],
    "decision_support": [
        "Name the options already provided.",
        "Compare benefits, costs, risks, values, and reversibility.",
        "Offer a practical decision process or recommendation.",
        "Ask at most one question after comparing options.",
    ],
    "workplace_discrimination": [
        "Validate the experience without making unverified legal conclusions.",
        "Recommend documenting dates, words, witnesses, and impact.",
        "Suggest careful HR, union, regulator, or legal guidance as appropriate.",
        "Consider a quiet job search when safety or retaliation is a concern.",
    ],
    "cognitive_challenge": [
        "Validate the emotion before challenging the conclusion.",
        "Identify absolute or unsupported thinking tentatively.",
        "Compare evidence for and against the thought.",
        "Offer a more balanced alternative.",
    ],
    "venting": [
        "Listen without advice or reframing.",
        "Reflect the specific frustration naturally.",
        "Ask at most one question that helps the person continue.",
    ],
    "casual": [
        "Respond like ordinary conversation.",
        "Do not force emotional analysis.",
        "Match playful or neutral tone appropriately.",
    ],
    "current_facts": [
        "Use live retrieval.",
        "Summarize verified facts and distinguish uncertainty.",
        "Include source titles and URLs.",
        "Do not invent current details when retrieval fails.",
    ],
    "research": [
        "Search academic or primary sources.",
        "Summarize findings with limitations.",
        "Include paper or source titles and URLs.",
        "Do not present a search as exhaustive.",
    ],
    "crisis_safety": [
        "Prioritize immediate safety over every other mode.",
        "Use direct, calm language.",
        "Provide verified country-appropriate crisis resources.",
        "Encourage immediate human contact and emergency help when needed.",
    ],
    "mixed_complex_life_problem": [
        "Summarize the whole situation briefly.",
        "Separate the issues into emotional load, practical choices, and unknowns.",
        "Prioritize the nearest consequence.",
        "Give a recommendation.",
        "Ask at most one question.",
    ],
}

FALLBACK_TRAPS = {
    "grief": [
        "Here are five ways to move on.",
        "There is no need to solve this right away. What feels most present?",
    ],
    "advice": [
        "What feels most present for you?",
        "Say a little more about that.",
        "The effect on your daily life matters here.",
    ],
    "decision_support": [
        "Is there a specific choice connected to this?",
        "Only you can decide.",
    ],
    "venting": ["Here are three solutions.", "Let us challenge that thought."],
    "casual": ["What emotion is underneath that statement?"],
    "current_facts": ["I think the latest rule is probably unchanged."],
    "research": ["Research proves this works for everyone."],
    "crisis_safety": ["Try journaling and see whether the feeling passes."],
    "mixed_complex_life_problem": ["What feels most present for you?"],
}

STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "been",
    "before",
    "being",
    "could",
    "did",
    "does",
    "felt",
    "from",
    "have",
    "having",
    "into",
    "just",
    "like",
    "more",
    "much",
    "really",
    "said",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "very",
    "was",
    "were",
    "what",
    "when",
    "with",
    "would",
}


def _quality_score(row: dict[str, Any]) -> tuple[float, int, int]:
    return (
        float(row.get("priority_weight", 1.0)),
        len(row.get("response_requirements", [])),
        len(str(row.get("ideal_response", ""))),
    )


def _compact_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "dataset_source": row["dataset_source"],
        "user_input": row["user_input"],
        "ideal_response": row["ideal_response"],
        "response_requirements": row["response_requirements"],
        "should_use_internet": row["should_use_internet"],
        "should_use_research": row["should_use_research"],
        "should_use_safety": row["should_use_safety"],
        "priority_weight": row.get("priority_weight", 1.0),
    }


def build(input_path: Path, output_path: Path) -> dict[str, Any]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    traps: dict[str, list[str]] = defaultdict(list)
    emotion_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    token_totals: Counter[str] = Counter()

    for row in iter_jsonl(input_path):
        intent = str(row.get("target_intent", ""))
        if (
            intent == "emotion_detection"
            and row.get("split") == "train"
            and row.get("target_emotion")
        ):
            emotion = str(row["target_emotion"])
            tokens = {
                token
                for token in re.findall(r"[a-z']+", str(row["user_input"]).lower())
                if len(token) >= 3 and token not in STOPWORDS
            }
            for token in tokens:
                emotion_token_counts[emotion][token] += 1
                token_totals[token] += 1
        if intent not in TARGET_INTENTS or row.get("split") != "train":
            continue
        if row.get("ideal_response"):
            candidates[intent].append(row)
        trap = str(row.get("bad_response_trap", "")).strip()
        if trap and trap not in traps[intent]:
            traps[intent].append(trap)

    library: dict[str, Any] = {
        "version": 1,
        "fine_tuning_performed": False,
        "selection_policy": (
            "Training-split examples only; explicit advice and named decisions have "
            "priority. Emotion-only datasets are excluded from response examples."
        ),
        "intents": {},
    }
    for intent in TARGET_INTENTS:
        ranked = sorted(candidates[intent], key=_quality_score, reverse=True)
        strong = [_compact_example(row) for row in ranked[:10]]
        selected_traps = traps[intent][:]
        for fallback in FALLBACK_TRAPS.get(intent, []):
            if fallback not in selected_traps:
                selected_traps.append(fallback)
        library["intents"][intent] = {
            "strong_examples": strong,
            "bad_response_traps": selected_traps[:5],
            "expected_response_structure": EXPECTED_STRUCTURES[intent],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(library, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    lexicon_path = output_path.parents[1] / "training" / "emotion_lexicon.json"
    lexicon_path.parent.mkdir(parents=True, exist_ok=True)
    lexicon: dict[str, list[dict[str, Any]]] = {}
    for emotion, counts in emotion_token_counts.items():
        ranked: list[tuple[float, str, int, float]] = []
        for token, count in counts.items():
            if count < 8:
                continue
            precision = count / token_totals[token]
            if precision < 0.5:
                continue
            score = precision * math.log1p(count)
            ranked.append((score, token, count, precision))
        lexicon[emotion] = [
            {
                "token": token,
                "score": round(score, 4),
                "count": count,
                "precision": round(precision, 4),
            }
            for score, token, count, precision in sorted(ranked, reverse=True)[:150]
        ]
    lexicon_path.write_text(
        json.dumps(
            {
                "version": 1,
                "source": "training-split emotion classification rows",
                "response_generation_allowed": False,
                "emotions": lexicon,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return {
        "output": str(output_path),
        "emotion_lexicon": str(lexicon_path),
        "intents": {
            intent: {
                "strong_examples": len(value["strong_examples"]),
                "bad_response_traps": len(value["bad_response_traps"]),
            }
            for intent, value in library["intents"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/noema_unified_dataset.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/fewshot_examples.json"),
    )
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
