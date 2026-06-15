from core.pipeline import process_reflection
from core.safety import assess_safety, generate_crisis_response


def test_clear_crisis_phrase_is_detected():
    result = assess_safety("I want to end my life.")
    assert result.level == "immediate"


def test_ordinary_distress_is_not_marked_as_crisis():
    result = assess_safety("My exam stress is exhausting.")
    assert result.level == "none"


def test_feeling_like_dying_is_treated_as_a_safety_concern():
    result = assess_safety("I feel like dying.")
    assert result.level == "concern"


def test_feeling_like_hurting_myself_is_treated_as_a_safety_concern():
    result = assess_safety("I feel like hurting myself.")
    assert result.level == "concern"


def test_mixed_negation_does_not_hide_another_safety_phrase():
    result = assess_safety("I am not suicidal, but I feel like dying.")
    assert result.level == "concern"


def test_clear_denial_is_not_treated_as_a_crisis():
    result = assess_safety("I am not suicidal and I do not want to die.")
    assert result.level == "none"


def test_ordinary_context_does_not_trigger_ambiguous_safety_phrase():
    result = assess_safety("I can't go on with this assignment tonight.")
    assert result.level == "none"


def test_location_context_does_not_trigger_here_anymore_phrase():
    result = assess_safety("I don't want to be here anymore at this party.")
    assert result.level == "none"


def test_phase_false_positive_examples_are_not_crisis():
    examples = (
        "I have not received a promotion in 4 years. Should I leave?",
        "I have Rs 2 lakh. Should I start a clothing brand or invest in my education?",
        "I love psychology but Data Science pays more.",
        (
            "My grandfather died last year. I am waiting for Germany admissions. "
            "I feel behind compared to my friends. Part of me wants to start a "
            "business. I am worried I am making the wrong decision with my life."
        ),
    )
    for text in examples:
        assert assess_safety(text).level == "none"


def test_india_crisis_response_is_available_without_internet():
    response = generate_crisis_response("concern", "IN")
    assert "112" in response
    assert "14416" in response


def test_crisis_bypasses_bias_analysis():
    result = process_reflection("I want to kill myself because I always fail.")
    assert result.category.category == "crisis / safety"
    assert result.biases == []
    assert "immediate danger" in result.response.lower()
