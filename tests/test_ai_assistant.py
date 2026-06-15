import ai.assistant as assistant
from ai.assistant import generate_ai_response
from core.conversation import continue_conversation


def test_missing_api_key_uses_local_conversation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    local_reply = continue_conversation("I feel anxious about tomorrow.")

    result = generate_ai_response(
        user_text="I feel anxious about tomorrow.",
        history=[],
        local_reply=local_reply,
        excerpts=[],
        api_key=None,
    )

    assert result.text == local_reply.response
    assert result.mode == "local fallback"
    assert result.error is None


def test_emotional_message_never_enables_web_search(monkeypatch):
    captured = {}

    def fake_create(request, _api_key):
        captured.update(request)
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "I hear you."}],
                }
            ]
        }

    monkeypatch.setattr(assistant, "_create_response", fake_create)
    local_reply = continue_conversation("My cat died today.")
    result = generate_ai_response(
        user_text="My cat died today.",
        history=[],
        local_reply=local_reply,
        excerpts=[],
        api_key="test-key",
    )

    assert "tools" not in captured
    assert not result.internet_used
    assert not result.research_used


def test_current_fact_route_does_not_use_openai_web_tool(monkeypatch):
    captured = {}

    def fake_create(request, _api_key):
        captured.update(request)
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "This call should use internal generation only.",
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr(assistant, "_create_response", fake_create)
    local_reply = continue_conversation(
        "What are current Germany student visa rules?"
    )
    result = generate_ai_response(
        user_text="What are current Germany student visa rules?",
        history=[],
        local_reply=local_reply,
        excerpts=[],
        api_key="test-key",
    )

    assert "tools" not in captured
    assert not result.internet_used
    assert not result.retrieval_succeeded


def test_openai_prompt_enforces_advice_before_questions(monkeypatch):
    captured = {}

    def fake_create(request, _api_key):
        captured.update(request)
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Practical advice."}],
                }
            ]
        }

    monkeypatch.setattr(assistant, "_create_response", fake_create)
    local_reply = continue_conversation(
        "I don't know what to do. Suggest something."
    )
    generate_ai_response(
        user_text="I don't know what to do. Suggest something.",
        history=[],
        local_reply=local_reply,
        excerpts=[],
        api_key="test-key",
    )

    instructions = captured["instructions"]
    assert "Advice Priority Rule" in instructions
    assert "Do not ask an exploratory therapy-style question before giving advice" in instructions
