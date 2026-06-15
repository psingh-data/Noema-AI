from core.pipeline import process_reflection
from core.router import route_message
import pytest


def route(text: str, preferred_mode: str = "Just listen"):
    analysis = process_reflection(text)
    return route_message(
        text,
        preferred_mode=preferred_mode,
        category=analysis.category.category,
        emotion=analysis.emotion.emotion,
        has_bias=bool(analysis.biases),
        clinical_domains=(),
    )


def test_short_bold_statement_routes_to_casual_conversation():
    decision = route("I am the king")
    assert decision.intent == "casual conversation"
    assert decision.response_mode == "Friend"


def test_named_choice_routes_to_decision_support():
    decision = route("Should I go to university or just choose my business?")
    assert decision.intent == "decision support"
    assert decision.response_mode == "Help me make a decision"


def test_how_can_i_question_routes_to_advice():
    decision = route("How can I stop procrastinating?")
    assert decision.intent == "practical advice"


@pytest.mark.parametrize(
    "text",
    (
        "I don't know what to do. What should I do?",
        "I don't know what to do. Suggest something.",
        "Please give me advice.",
        "Help me decide what to do next.",
        "Recommend something practical.",
        "Any ideas?",
    ),
)
def test_explicit_advice_language_has_priority(text):
    decision = route(text)
    assert decision.intent == "practical advice"
    assert decision.response_mode == "Give me advice"


def test_explicit_no_advice_routes_to_venting():
    decision = route("I don't need advice. I just want to complain.")
    assert decision.intent == "venting"
    assert decision.response_mode == "Just listen"


def test_just_let_me_complain_routes_to_venting():
    decision = route("Just let me complain.")
    assert decision.intent == "venting"


def test_global_social_claim_routes_to_cognitive_challenge():
    decision = route("Everyone hates me.")
    assert decision.intent == "cognitive challenge"


def test_current_visa_question_routes_to_internet():
    decision = route("What are the visa requirements for Germany in 2026?")
    assert decision.intent == "current factual search"
    assert decision.knowledge_route == "internet"


def test_research_question_routes_to_papers():
    decision = route("What does research say about procrastination?")
    assert decision.intent == "research paper question"
    assert decision.knowledge_route == "research papers"


def test_university_deadline_routes_to_internet():
    decision = route("What's the deadline for Osnabruck Cognitive Science?")
    assert decision.intent == "current factual search"
    assert decision.knowledge_route == "internet"


def test_grief_today_does_not_route_to_internet():
    decision = route("My cat died today.")
    assert decision.intent == "grief"
    assert decision.knowledge_route != "internet"


def test_uncertain_emotional_message_routes_to_reflection():
    decision = route("I don't know what I need, I just feel weird.")
    assert decision.intent == "emotional reflection"


def test_cognitive_reframing_question_routes_to_research():
    decision = route("Does cognitive reframing work?")
    assert decision.intent == "research paper question"
    assert decision.knowledge_route == "research papers"


def test_product_recommendation_routes_to_current_search():
    decision = route("Which laptop should I buy for data science?")
    assert decision.intent == "current factual search"
    assert decision.knowledge_route == "internet"


def test_explicit_product_recommendation_uses_advice_mode_and_internet():
    decision = route("Recommend a laptop for data science.")
    assert decision.intent == "practical advice"
    assert decision.response_mode == "Give me advice"
    assert decision.knowledge_route == "internet"


def test_stable_career_question_routes_to_career_education():
    decision = route("What career path fits a cognitive science degree?")
    assert decision.intent == "career / education"
    assert decision.knowledge_route == "internet"


def test_stable_general_question_routes_to_general_knowledge():
    decision = route("Why is the sky blue?")
    assert decision.intent == "general knowledge"


def test_ai_news_routes_to_current_search():
    decision = route("What is the latest AI news?")
    assert decision.intent == "current factual search"
    assert decision.knowledge_route == "internet"


def test_software_information_routes_to_current_search():
    decision = route("What is the latest version of this software tool?")
    assert decision.intent == "current factual search"
    assert decision.knowledge_route == "internet"


def test_confused_between_routes_to_decision_support():
    decision = route("I am confused between changing company and switching careers.")
    assert decision.intent == "decision support"


def test_right_move_routes_to_decision_support():
    decision = route("I want to take a break. Is it the right move?")
    assert decision.intent == "decision support"


def test_need_steps_routes_to_advice():
    decision = route("I need steps, not just motivation.")
    assert decision.intent == "practical advice"


def test_foundational_papers_routes_to_research():
    decision = route("Give me foundational papers in organizational psychology.")
    assert decision.intent == "research paper question"


def test_named_decision_beats_generic_advice_request():
    decision = route("Should I talk to HR or take a break? What should I do?")
    assert decision.intent == "decision support"


def test_cognitive_distortion_beats_practical_advice_marker():
    decision = route("If this fails, my whole future is ruined. Please be practical.")
    assert decision.intent == "cognitive challenge"


def test_grief_plus_advice_routes_to_practical_grief_support():
    decision = route("I miss my sister so much today. Please be practical.")
    assert decision.intent == "practical advice"
    assert decision.knowledge_route == "conversation context"


def test_mixed_complex_life_problem_routes_to_new_intent():
    decision = route(
        "I feel overwhelmed about my career, university admissions, family pressure, "
        "money, and moving abroad. I don't know what to do about my future."
    )
    assert decision.intent == "mixed complex life problem"
    assert decision.response_mode == "Give me advice"


def test_interview_failure_routes_to_cognitive_challenge_with_career_or_self_esteem_topic():
    decision = route("I feel like a failure because I failed one interview.")
    assert decision.intent == "cognitive challenge"
    assert decision.topic in {"self_esteem", "career"}


def test_leave_company_routes_to_workplace_decision_support():
    decision = route("Should I leave my company?")
    assert decision.intent == "decision support"
    assert decision.topic == "workplace"


def test_cognitive_science_data_science_routes_to_education_decision_support():
    decision = route("Should I study Cognitive Science or Data Science?")
    assert decision.intent == "decision support"
    assert decision.topic == "education"


def test_grief_therapy_question_routes_to_grief_decision_support():
    decision = route("I miss my grandfather every day. Should I see a therapist?")
    assert decision.intent == "decision support"
    assert decision.topic == "grief"


def test_workplace_leave_question_does_not_route_to_crisis():
    decision = route(
        "My boss only promotes women. I have been stuck for 5 years. "
        "I want to leave the company. Should I do it?"
    )
    assert decision.intent == "decision support"
    assert decision.topic == "workplace"


def test_new_cognitive_distortion_examples_route_to_challenge():
    examples = (
        "I failed two interviews. I guess I am not smart enough.",
        "Everyone my age is ahead of me.",
        "I am 24 and my life is already ruined.",
    )
    for text in examples:
        decision = route(text)
        assert decision.intent == "cognitive challenge"


def test_relationship_decision_routes_to_relationship_topic():
    decision = route(
        "My girlfriend and I keep fighting. I still love her. "
        "I am exhausted. Should I stay or leave?"
    )
    assert decision.intent == "decision support"
    assert decision.topic == "relationship"


def test_quit_job_start_business_routes_to_business_topic():
    decision = route("I want to quit my job and start a business tomorrow.")
    assert decision.intent == "decision support"
    assert decision.topic == "business"


def test_workplace_discrimination_statement_gets_practical_workplace_route():
    decision = route("I think my manager discriminates against me.")
    assert decision.intent == "practical advice"
    assert decision.topic == "workplace"


def test_requested_mixed_life_example_routes_to_mixed_complex():
    decision = route(
        "My grandfather died last year. I am waiting for Germany admissions. "
        "I feel behind compared to my friends. Part of me wants to start a "
        "business. I am worried I am making the wrong decision with my life."
    )
    assert decision.intent == "mixed complex life problem"


def test_psychology_data_science_tradeoff_routes_to_education_decision():
    decision = route("I love psychology but Data Science pays more.")
    assert decision.intent == "decision support"
    assert decision.topic == "education"


def test_playful_questions_route_to_casual_chat():
    for text in ("Tell me a terrible joke.", "What Pokemon would make the best therapist?"):
        decision = route(text)
        assert decision.intent == "casual conversation"
