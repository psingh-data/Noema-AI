from critic import critique_response, repair_response
from core.router import RouteDecision


def route(intent: str, knowledge_route: str = "conversation context") -> RouteDecision:
    return RouteDecision(
        intent=intent,
        response_mode="Conversation",
        knowledge_route=knowledge_route,
        confidence=1.0,
        reason="test",
    )


def topic_route(
    intent: str,
    topic: str,
    knowledge_route: str = "conversation context",
) -> RouteDecision:
    return RouteDecision(
        intent=intent,
        response_mode="Conversation",
        knowledge_route=knowledge_route,
        confidence=1.0,
        reason="test",
        topic=topic,
    )


def test_critic_rejects_reflection_only_answer_to_advice_request():
    result = critique_response(
        user_input="What should I do? Suggest something.",
        response="What feels most present for you?",
        route=route("practical advice"),
    )
    assert not result.passed
    assert any("practical advice" in failure for failure in result.failures)
    assert any("therapy language" in failure for failure in result.failures)


def test_critic_accepts_direct_advice_with_one_question():
    result = critique_response(
        user_input="What should I do?",
        response=(
            "Start with one reversible next step. This lowers pressure while giving "
            "you useful information. What constraint matters most?"
        ),
        route=route("practical advice"),
    )
    assert result.passed


def test_critic_requires_uncertainty_when_live_retrieval_is_missing():
    result = critique_response(
        user_input="What are Germany's current visa requirements?",
        response="The requirements are unchanged.",
        route=route("current factual search", "internet"),
        internet_used=False,
    )
    assert not result.passed


def test_repair_adds_practical_structure_once():
    original = critique_response(
        user_input="Give me advice.",
        response="Say a little more about that.",
        route=route("practical advice"),
    )
    repaired = repair_response(
        user_input="Give me advice.",
        response="Say a little more about that.",
        route=route("practical advice"),
        failures=original.failures,
    )
    assert "start here" in repaired.lower()
    assert repaired.count("?") <= 1


def test_critic_rejects_generic_decision_when_topic_is_known():
    result = critique_response(
        user_input="Should I study Cognitive Science or Data Science?",
        response=(
            "There is a real decision here, and I do not want to invent details "
            "you have not given me. Tell me the options."
        ),
        route=route("decision support"),
    )
    assert not result.passed
    assert any("generic decision" in failure for failure in result.failures)


def test_critic_rejects_missed_obvious_cognitive_distortion():
    result = critique_response(
        user_input="I feel like a failure because I failed one interview.",
        response="That sounds difficult. What happened?",
        route=route("cognitive challenge"),
    )
    assert not result.passed
    assert any("cognitive distortion" in failure for failure in result.failures)


def test_critic_passes_grief_therapy_decision_response():
    response = (
        "Missing your grandfather every day can be a real part of grief.\n\n"
        "Seeing a therapist could be a good idea if grief feels unbearable, affects "
        "sleep, work, study, relationships, or daily functioning.\n\n"
        "My recommendation is to try one initial session and treat it as support."
    )
    result = critique_response(
        user_input="I miss my grandfather every day. Should I see a therapist?",
        response=response,
        route=topic_route("decision support", "grief"),
    )
    assert result.passed


def test_repair_preserves_grief_specific_response():
    response = (
        "Missing your grandfather every day can be a real part of grief. Therapy "
        "support may help if it affects sleep or daily functioning. My recommendation "
        "is to try one initial session."
    )
    repaired = repair_response(
        user_input="I miss my grandfather every day. Should I see a therapist?",
        response=response,
        route=topic_route("decision support", "grief"),
        failures=("some topic-specific repair needed",),
    )
    assert "grief" in repaired.lower()
    assert "therapy" in repaired.lower()
    assert "tell me the options" not in repaired.lower()


def test_critic_passes_education_field_comparison():
    response = (
        "You are choosing between Cognitive Science and Data Science. Cognitive "
        "Science fits mind, brain, behavior, and research. Data Science is more "
        "job-oriented through statistics, machine learning, and analytics. My "
        "conditional recommendation is to choose based on job clarity versus "
        "intellectual fit."
    )
    result = critique_response(
        user_input="Should I study Cognitive Science or Data Science?",
        response=response,
        route=topic_route("decision support", "education"),
    )
    assert result.passed


def test_critic_passes_workplace_decision_response():
    response = (
        "I do not recommend quitting impulsively tomorrow. Document promotion "
        "criteria, feedback, and achievements, update your resume, and explore the "
        "job market while you are still employed. My recommendation is to prepare "
        "an exit path before resigning."
    )
    result = critique_response(
        user_input=(
            "My boss only promotes women. I have been stuck for 5 years. "
            "I want to leave the company. Should I do it?"
        ),
        response=response,
        route=topic_route("decision support", "workplace"),
    )
    assert result.passed


def test_critic_rejects_diagnostic_claims():
    result = critique_response(
        user_input="I feel hopeless and cannot get out of bed.",
        response="You have depression. You should seek help.",
        route=topic_route("emotional reflection", "health"),
    )
    assert not result.passed
    assert any("diagnostic claim" in failure for failure in result.failures)


def test_repair_rewrites_diagnostic_claims_as_overlap_language():
    repaired = repair_response(
        user_input="I feel hopeless and cannot get out of bed.",
        response="You have depression. You should seek help.",
        route=topic_route("emotional reflection", "health"),
        failures=("response made a diagnostic claim instead of a possible overlap",),
    )
    lowered = repaired.lower()
    assert "you have depression" not in lowered
    assert "overlaps with depressive symptoms" in lowered
    assert "not a diagnosis" in lowered
