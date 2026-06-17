from core.bias_detector import detect_biases
from core.classifier import classify_category
from core.emotion_detector import detect_emotion
from core.pipeline import process_reflection
from core.symptom_profile import (
    build_symptom_profile,
    overlap_summary,
    possible_clinical_overlaps,
)


def test_unknown_input_becomes_general_reflection():
    result = classify_category("I have been thinking about yesterday.")
    assert result.category == "general reflection"


def test_empty_input_is_safe_to_analyze():
    result = process_reflection("")
    assert result.emotion.emotion == "neutral"
    assert result.category.category == "general reflection"


def test_anxiety_and_medium_intensity_are_detected():
    result = detect_emotion("I am really anxious and worried about tomorrow.")
    assert result.emotion == "anxiety"
    assert result.intensity == "medium"


def test_bias_language_is_tentative():
    biases = detect_biases("Everyone thinks I am a total failure.")
    assert biases
    result = process_reflection("Everyone thinks I am a total failure.")
    assert "there may be" in result.response.lower()


def test_high_intensity_response_stays_short():
    result = process_reflection("Everything feels impossible and I am extremely overwhelmed.")
    assert result.emotion.intensity == "high"
    assert len(result.response.split()) < 90


def test_symptom_profile_scores_overlap_without_diagnosis():
    profile = build_symptom_profile(
        "I feel hopeless, lost interest, cannot focus, and cannot get out of bed."
    )
    assert profile["depressive_symptoms"] >= 3
    assert profile["attention_regulation_symptoms"] >= 1
    overlaps = possible_clinical_overlaps(profile)
    summary = overlap_summary(overlaps).lower()
    assert "overlap" in summary
    assert "not mean any condition is present" in summary
    assert "professional evaluation" in summary
    assert "you have depression" not in summary
    assert "you have adhd" not in summary
    assert "you have bipolar" not in summary
