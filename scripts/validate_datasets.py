"""Validate Noema's unified dataset before any training preparation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_common import SCHEMA_FIELDS, iter_jsonl


BOOLEAN_FIELDS = (
    "should_use_internet",
    "should_use_research",
    "should_use_safety",
)
LIST_FIELDS = ("response_requirements",)
DICT_FIELDS = ("details_panel_expected",)
VALID_SPLITS = {"train", "validation", "test"}
EMOTION_ONLY_SOURCES = {"goemotions", "emotion_69k"}


def validate(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    total = 0

    for line_number, row in enumerate(iter_jsonl(path), start=1):
        total += 1
        missing = [field for field in SCHEMA_FIELDS if field not in row]
        if missing:
            errors.append(f"line {line_number}: missing fields {missing}")
            continue

        record_id = str(row["id"])
        if not record_id:
            errors.append(f"line {line_number}: empty id")
        elif record_id in seen_ids:
            errors.append(f"line {line_number}: duplicate id {record_id}")
        seen_ids.add(record_id)

        if not str(row["user_input"]).strip():
            errors.append(f"line {line_number}: empty user_input")
        if row["split"] not in VALID_SPLITS:
            errors.append(f"line {line_number}: invalid split {row['split']!r}")
        for field in BOOLEAN_FIELDS:
            if not isinstance(row[field], bool):
                errors.append(f"line {line_number}: {field} must be boolean")
        for field in LIST_FIELDS:
            if not isinstance(row[field], list):
                errors.append(f"line {line_number}: {field} must be a list")
        for field in DICT_FIELDS:
            if not isinstance(row[field], dict):
                errors.append(f"line {line_number}: {field} must be an object")

        source = str(row["dataset_source"])
        if source in EMOTION_ONLY_SOURCES:
            if row["ideal_response"] or row["bad_response_trap"]:
                errors.append(
                    f"line {line_number}: emotion-only source contains response text"
                )
            if row.get("training_eligible"):
                errors.append(
                    f"line {line_number}: emotion-only source marked generation eligible"
                )

        if row["should_use_research"] and not row["should_use_internet"]:
            warnings.append(
                f"line {line_number}: research is true while internet is false"
            )

        source_counts[source] += 1
        split_counts[str(row["split"])] += 1
        intent_counts[str(row["target_intent"])] += 1

    report = {
        "valid": not errors,
        "total_records": total,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors[:100],
        "warnings": warnings[:100],
        "records_by_source": dict(source_counts),
        "records_by_split": dict(split_counts),
        "records_by_intent": dict(intent_counts),
        "checks": {
            "required_schema": True,
            "unique_ids": not any("duplicate id" in error for error in errors),
            "emotion_sources_excluded_from_generation": not any(
                "emotion-only source" in error for error in errors
            ),
            "fine_tuning_performed": False,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/noema_unified_dataset.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/evaluation/validation_report.json"),
    )
    args = parser.parse_args()
    report = validate(args.input)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
