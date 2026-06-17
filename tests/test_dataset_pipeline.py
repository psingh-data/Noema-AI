import json
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dataset_common import normalized_record, priority_intent
from prepare_datasets import normalize_longform
from validate_datasets import validate


def test_research_records_always_enable_internet():
    record = normalized_record(
        record_id="research-1",
        dataset_source="test",
        user_input="What do recent studies say?",
        target_intent="research",
        should_use_research=True,
    )
    assert record["should_use_research"]
    assert record["should_use_internet"]


def test_explicit_advice_overrides_reflection_label():
    intent, weight = priority_intent(
        "I feel lost. What should I do?",
        "emotional_reflection",
    )
    assert intent == "advice"
    assert weight > 1


def test_named_decision_overrides_reflection_label():
    intent, weight = priority_intent(
        "Should I study or start a business?",
        "emotional_reflection",
    )
    assert intent == "decision_support"
    assert weight > 1


def test_validator_rejects_response_text_in_emotion_only_source(tmp_path):
    path = tmp_path / "bad.jsonl"
    record = normalized_record(
        record_id="emotion-1",
        dataset_source="goemotions",
        user_input="I feel sad.",
        target_intent="emotion_detection",
        target_emotion="sadness",
        ideal_response="This must not be used.",
        training_eligible=False,
    )
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    report = validate(path)
    assert not report["valid"]
    assert any("emotion-only source" in error for error in report["errors"])


def test_longform_dataset_normalizes_style_metadata(tmp_path):
    path = tmp_path / "longform.jsonl"
    row = {
        "id": "noema_long_test",
        "dataset_source": "noema_longform_synthetic_v1",
        "user_input": "Suggest me some therapies for grief.",
        "target_intent": "therapy_recommendation",
        "response_length_target": "long",
        "should_use_internet": True,
        "should_use_research": True,
        "should_preserve_context": True,
        "must_not_repeat_previous_intervention": False,
        "bad_response_trap": "Just try breathing.",
        "ideal_response": "Here are several support options...",
        "critic_checks": "response_is_not_too_short|uses_topic_specific_guidance",
        "split": "train",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    [record] = list(normalize_longform(path, "noema_longform"))
    assert record["dataset_source"] == "noema_longform_synthetic_v1"
    assert record["target_intent"] == "intervention_request"
    assert record["should_use_internet"]
    assert record["should_use_research"]
    assert record["metadata"]["response_length_target"] == "long"
    assert "response_is_not_too_short" in record["response_requirements"]
