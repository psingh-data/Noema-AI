"""Normalize Noema's source datasets without creating fine-tuning artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from dataset_common import (
    as_bool,
    as_dict,
    as_list,
    clean_text,
    deterministic_split,
    iter_jsonl,
    last_dialogue_turn,
    map_emotion,
    normalized_record,
    priority_intent,
    stable_id,
)


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2_147_483_647)

DEFAULT_SOURCES = {
    "noema_10k": Path(r"C:\Users\prabh\Downloads\noema_10000_training_dataset.jsonl"),
    "noema_seed": Path(r"C:\Users\prabh\Downloads\noema_training_seed_dataset.jsonl"),
    "goemotions_1": Path(r"C:\Users\prabh\Downloads\goemotions_1.csv"),
    "goemotions_2": Path(r"C:\Users\prabh\Downloads\goemotions_2.csv"),
    "goemotions_3": Path(r"C:\Users\prabh\Downloads\goemotions_3.csv"),
    "emotion_69k": Path(
        r"C:\Users\prabh\Downloads\archive (3)\emotion-emotion_69k.csv"
    ),
    "counselchat": Path(
        r"C:\Users\prabh\Downloads\archive (1)\counselchat-data.csv"
    ),
    "hh_helpful_train": Path(
        r"C:\Users\prabh\Downloads\archive (4)\helpful_train.jsonl"
    ),
    "hh_helpful_test": Path(
        r"C:\Users\prabh\Downloads\archive (4)\helpful_test.jsonl"
    ),
    "hh_harmless_train": Path(
        r"C:\Users\prabh\Downloads\archive (4)\harmless_train.jsonl"
    ),
    "hh_harmless_test": Path(
        r"C:\Users\prabh\Downloads\archive (4)\harmless_test.jsonl"
    ),
}

GOEMOTION_COLUMNS = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)


def normalize_noema(path: Path, source: str) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(iter_jsonl(path), start=1):
        user_input = clean_text(row.get("user_input") or row.get("input"))
        if not user_input:
            continue
        original_intent = row.get("intent") or row.get("target_intent") or "casual"
        intent, weight = priority_intent(user_input, str(original_intent))
        yield normalized_record(
            record_id=clean_text(row.get("id")) or stable_id(source, user_input),
            dataset_source=source,
            user_input=user_input,
            target_intent=intent,
            target_emotion=map_emotion(
                row.get("emotion") or row.get("target_emotion") or ""
            ),
            target_mode=row.get("mode") or row.get("target_mode") or "",
            emotion_intensity=row.get("emotion_intensity") or "",
            should_use_internet=as_bool(row.get("should_use_internet")),
            should_use_research=as_bool(row.get("should_use_research")),
            should_use_safety=as_bool(row.get("should_use_safety")),
            bad_response_trap=row.get("bad_response_trap") or "",
            ideal_response=row.get("ideal_response") or "",
            response_requirements=as_list(row.get("response_requirements")),
            details_panel_expected=as_dict(row.get("details_panel_expected")),
            split=row.get("split") or deterministic_split(user_input),
            priority_weight=weight,
            metadata={
                "original_intent": clean_text(original_intent),
                "priority_override": intent
                != clean_text(original_intent).lower().replace(" ", "_"),
                "source_row": index,
            },
        )


def normalize_goemotions(paths: list[Path]) -> Iterable[dict[str, Any]]:
    examples: dict[str, dict[str, Any]] = {}
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    unclear: Counter[str] = Counter()

    for path in paths:
        with path.open(
            "r", encoding="utf-8-sig", errors="replace", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                example_id = clean_text(row.get("id"))
                text = clean_text(row.get("text"))
                if not example_id or not text:
                    continue
                examples.setdefault(example_id, {"text": text})
                if as_bool(row.get("example_very_unclear")):
                    unclear[example_id] += 1
                    continue
                for label in GOEMOTION_COLUMNS:
                    if as_bool(row.get(label)):
                        votes[example_id][label] += 1

    for example_id, value in examples.items():
        if not votes[example_id] or unclear[example_id] > sum(votes[example_id].values()):
            continue
        max_vote = max(votes[example_id].values())
        labels = [label for label, count in votes[example_id].items() if count == max_vote]
        text = value["text"]
        yield normalized_record(
            record_id=f"goemotions-{example_id}",
            dataset_source="goemotions",
            user_input=text,
            target_intent="emotion_detection",
            target_emotion=map_emotion(labels),
            target_mode="Emotion classification",
            split=deterministic_split(example_id),
            priority_weight=1.0,
            training_eligible=False,
            metadata={"raw_labels": labels, "rater_votes": dict(votes[example_id])},
        )


def normalize_emotion_69k(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            text = clean_text(row.get("Situation") or row.get("situation"))
            raw_emotion = clean_text(row.get("emotion"))
            if not text:
                continue
            yield normalized_record(
                record_id=stable_id("emotion69k", f"{index}:{text}"),
                dataset_source="emotion_69k",
                user_input=text,
                target_intent="emotion_detection",
                target_emotion=map_emotion(raw_emotion),
                target_mode="Emotion classification",
                split=deterministic_split(text),
                priority_weight=0.8,
                training_eligible=False,
                metadata={"raw_emotion": raw_emotion},
            )


def _first(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def normalize_counselchat(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            title = _first(row, "questionTitle", "question_title", "title")
            body = _first(row, "questionText", "question", "text")
            user_input = clean_text(f"{title}. {body}" if title and body else title or body)
            answer = _first(row, "answerText", "answer", "response")
            if not user_input or not answer:
                continue
            intent, _ = priority_intent(user_input, "emotional_reflection")
            yield normalized_record(
                record_id=stable_id("counselchat", f"{index}:{user_input}"),
                dataset_source="counselchat",
                user_input=user_input,
                target_intent=intent,
                target_emotion="",
                ideal_response=answer,
                response_requirements=(
                    "Use only as a low-weight warmth example.",
                    "Do not copy therapist claims, diagnoses, or authority.",
                ),
                split=deterministic_split(user_input),
                priority_weight=0.15,
                training_eligible=True,
                metadata={"topic": _first(row, "topic")},
            )


def normalize_hh(path: Path, source: str, split: str) -> Iterable[dict[str, Any]]:
    quality_type = "harmless" if "harmless" in source else "helpful"
    for index, row in enumerate(iter_jsonl(path), start=1):
        chosen = str(row.get("chosen") or "")
        rejected = str(row.get("rejected") or "")
        user_input = last_dialogue_turn(chosen, "Human") or last_dialogue_turn(
            rejected, "Human"
        )
        ideal_response = last_dialogue_turn(chosen, "Assistant")
        bad_response = last_dialogue_turn(rejected, "Assistant")
        if not user_input or not ideal_response:
            continue
        yield normalized_record(
            record_id=stable_id(source, f"{index}:{user_input}:{ideal_response[:80]}"),
            dataset_source=source,
            user_input=user_input,
            target_intent="response_quality",
            target_mode="Response quality",
            should_use_safety=quality_type == "harmless",
            bad_response_trap=bad_response,
            ideal_response=ideal_response,
            response_requirements=(
                "Prefer the chosen response over the rejected response.",
                "Use this pair for response-quality comparison only.",
            ),
            split=split,
            priority_weight=0.25,
            training_eligible=True,
            metadata={"preference_type": quality_type},
        )


def available_sources() -> dict[str, Path]:
    return {name: path for name, path in DEFAULT_SOURCES.items() if path.exists()}


def all_records(sources: dict[str, Path]) -> Iterable[dict[str, Any]]:
    if "noema_10k" in sources:
        yield from normalize_noema(sources["noema_10k"], "noema_10k")
    if "noema_seed" in sources:
        yield from normalize_noema(sources["noema_seed"], "noema_seed")

    go_paths = [
        sources[name]
        for name in ("goemotions_1", "goemotions_2", "goemotions_3")
        if name in sources
    ]
    if go_paths:
        yield from normalize_goemotions(go_paths)
    if "emotion_69k" in sources:
        yield from normalize_emotion_69k(sources["emotion_69k"])
    if "counselchat" in sources:
        yield from normalize_counselchat(sources["counselchat"])

    for name in (
        "hh_helpful_train",
        "hh_helpful_test",
        "hh_harmless_train",
        "hh_harmless_test",
    ):
        if name in sources:
            split = "test" if name.endswith("_test") else "train"
            yield from normalize_hh(sources[name], name, split)


def prepare(output_root: Path) -> dict[str, Any]:
    sources = available_sources()
    if not sources:
        raise FileNotFoundError("None of the configured dataset files were found.")

    for folder in ("raw", "processed", "training", "evaluation"):
        (output_root / folder).mkdir(parents=True, exist_ok=True)

    manifest = {
        "fine_tuning_performed": False,
        "sources": {
            name: {"path": str(path), "bytes": path.stat().st_size}
            for name, path in sources.items()
        },
        "notes": [
            "GoEmotions and emotion_69k are emotion-classification only.",
            "CounselChat has low priority weight to avoid therapist-style dominance.",
            "HH-RLHF rows retain chosen and rejected responses as preference pairs.",
            "Explicit advice and named-decision wording overrides reflection labels.",
        ],
    }
    (output_root / "raw" / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    output = output_root / "processed" / "noema_unified_dataset.jsonl"
    counts: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    total = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in all_records(sources):
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            counts[record["dataset_source"]] += 1
            splits[record["split"]] += 1
            total += 1

    summary = {
        "output": str(output),
        "total_records": total,
        "records_by_source": dict(counts),
        "records_by_split": dict(splits),
        "fine_tuning_performed": False,
    }
    (output_root / "processed" / "ingestion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    summary = prepare(args.output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
