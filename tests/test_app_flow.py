from streamlit.testing.v1 import AppTest
import ai.tavily as tavily


def run_app():
    return AppTest.from_file("app.py").run(timeout=30)


def test_single_chatbox_has_no_forced_mode_controls():
    app = run_app()
    assert not app.exception
    assert len(app.chat_input) == 1
    assert app.chat_input[0].placeholder == "Think out loud or ask anything..."
    assert len(app.radio) == 0
    assert any(
        "helps you understand, decide, learn, and grow" in item.value
        for item in app.markdown
    )


def test_grief_details_show_no_internet_and_contextual_suggestions():
    app = run_app()
    app.chat_input[0].set_value("My cat died today.").run(timeout=30)

    message = app.session_state["messages"][-1]
    markdown = [item.value for item in app.markdown]
    buttons = [button.label for button in app.button]

    assert message["intent"] == "grief"
    assert not message["internet_used"]
    assert "**Internet used:** No" in markdown
    assert "**Research used:** No" in markdown
    assert "**Source:** None" in markdown
    assert "**Number of sources used:** 0" in markdown
    assert "**Confidence:** Medium" in markdown
    assert "Listen more" in buttons
    assert "Give advice" in buttons


def test_current_lookup_failure_is_transparent_and_low_confidence():
    app = run_app()
    app.chat_input[0].set_value(
        "What are current Germany student visa rules?"
    ).run(timeout=30)

    message = app.session_state["messages"][-1]
    lowered = message["content"].lower()

    assert message["intent"] == "current factual search"
    assert message["confidence_level"] == "Low"
    assert "No external sources used" in message["knowledge_sources"]
    assert "i don't want to guess" in lowered
    assert "api key" not in lowered


def test_crisis_message_routes_to_local_safety_details():
    app = run_app()
    app.chat_input[0].set_value("I feel like hurting myself.").run(timeout=30)

    message = app.session_state["messages"][-1]
    assert message["intent"] == "crisis / safety"
    assert message["confidence_level"] == "High"
    assert "Safety Rules" in message["knowledge_sources"]
    assert "Local Crisis Resources" in message["knowledge_sources"]
    assert "112" in message["content"]


def test_user_can_continue_naturally_without_mode_buttons():
    app = run_app()
    app.chat_input[0].set_value("I don't know what I need, I just feel weird.").run(
        timeout=30
    )
    app.chat_input[0].set_value("It has been like this all week.").run(timeout=30)

    assert not app.exception
    assert len(app.radio) == 0
    assert len(app.session_state["messages"]) == 4


def test_optional_suggestion_changes_next_emotional_response():
    app = run_app()
    app.chat_input[0].set_value(
        "My grandfather passed away months ago but I still cry."
    ).run(timeout=30)

    advice_button = next(button for button in app.button if button.label == "Give advice")
    advice_button.click().run(timeout=30)
    app.chat_input[0].set_value("The evenings are the hardest.").run(timeout=30)

    message = app.session_state["messages"][-1]
    assert message["support_mode"] == "Give me advice"
    assert "a gentle place to start" in message["content"].lower()


def test_research_suggestion_changes_next_knowledge_route():
    app = run_app()
    app.chat_input[0].set_value("How do I stop procrastinating?").run(timeout=30)

    research_button = next(
        button for button in app.button if button.label == "Find research evidence"
    )
    research_button.click().run(timeout=30)
    app.chat_input[0].set_value("Does it actually work?").run(timeout=30)

    message = app.session_state["messages"][-1]
    assert message["intent"] == "research paper question"
    assert message["knowledge_route"] == "research papers"


def test_memory_requires_an_explicit_button_click():
    app = run_app()
    app.chat_input[0].set_value("I am applying to cognitive science programs.").run(
        timeout=30
    )

    assert app.session_state["approved_memories"] == []
    remember = next(
        button for button in app.button
        if button.label == "Remember this for this session"
    )
    remember.click().run(timeout=30)
    assert app.session_state["approved_memories"] == [
        "I am applying to cognitive science programs."
    ]


def test_no_openai_still_keeps_tavily_research_available(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")

    def fake_search(_payload, _api_key):
        return {
            "answer": "A focused research summary from academic sources.",
            "results": [
                {
                    "title": "PubMed Review",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/00000000/",
                    "score": 0.9,
                    "content": "The review summarizes the available evidence.",
                },
                {
                    "title": "DOI Record",
                    "url": "https://doi.org/10.1000/example",
                    "score": 0.8,
                    "content": "The findings depend on population and method.",
                },
            ],
        }

    monkeypatch.setattr(tavily, "_create_tavily_search", fake_search)
    app = run_app()
    captions = [item.value for item in app.caption]
    assert "Core conversation: available" in captions
    assert "Optional OpenAI enhancement: unavailable" in captions
    assert "Live knowledge: available" in captions
    assert "Research retrieval: available" in captions

    app.chat_input[0].set_value(
        "What does research say about cognitive reframing?"
    ).run(timeout=30)
    message = app.session_state["messages"][-1]

    assert message["retrieval_provider"] == "Tavily"
    assert message["internet_used"]
    assert message["research_used"]
    assert len(message["source_links"]) == 2
    assert "focused research summary" in message["content"].lower()


def test_no_openai_keeps_emotional_and_decision_support(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = run_app()

    app.chat_input[0].set_value("My cat died today.").run(timeout=30)
    emotional = app.session_state["messages"][-1]
    assert emotional["intent"] == "grief"
    assert "grief does not follow a fixed timetable" in emotional["content"].lower()

    start_over = next(button for button in app.button if button.label == "Start over")
    start_over.click().run(timeout=30)
    app.chat_input[0].set_value(
        "Should I go to university or choose business?"
    ).run(timeout=30)
    decision = app.session_state["messages"][-1]
    assert decision["intent"] == "decision support"
    assert "university may offer" in decision["content"].lower()


def test_grief_therapist_decision_reaches_streamlit_without_generic_repair(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = run_app()

    app.chat_input[0].set_value(
        "I miss my grandfather every day. Should I see a therapist?"
    ).run(timeout=30)

    message = app.session_state["messages"][-1]
    lowered = message["content"].lower()

    assert message["intent"] == "decision support"
    assert message["critic_passed"]
    assert not message["critic_repaired"]
    assert "grief" in lowered
    assert "therapist" in lowered
    assert "tell me the options" not in lowered
