from core.bias_detector import detect_biases
from core.classifier import classify_category
from core.emotion_detector import detect_emotion
from core.pipeline import process_reflection


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

