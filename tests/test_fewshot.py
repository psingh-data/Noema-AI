from core.fewshot import (
    canonical_runtime_intent,
    expected_structure,
    select_fewshot_examples,
)


def test_runtime_intent_maps_to_dataset_group():
    assert canonical_runtime_intent("practical advice") == "advice"
    assert canonical_runtime_intent("research paper question") == "research"


def test_decision_selection_returns_reviewed_examples():
    examples = select_fewshot_examples(
        "decision support",
        "Should I go to university or start a business?",
    )
    assert len(examples) == 5
    assert all(example["ideal_response"] for example in examples)


def test_advice_structure_requires_advice_before_question():
    structure = expected_structure("practical advice")
    joined = " ".join(structure).lower()
    assert "practical suggestions immediately" in joined
    assert "after the advice" in joined
