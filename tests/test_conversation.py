from core.clinical_signals import detect_clinical_signals
from core.conversation import ConversationState, continue_conversation


def test_detects_multiple_cross_cutting_domains():
    signals = detect_clinical_signals(
        "I feel hopeless, cannot focus, and have been sleeping too much."
    )
    domains = {signal.domain for signal in signals}
    assert "depressed mood" in domains
    assert "concentration and memory" in domains
    assert "sleep" in domains


def test_negated_unusual_experience_is_not_flagged():
    signals = detect_clinical_signals("I am not hearing voices or seeing things.")
    domains = {signal.domain for signal in signals}
    assert "unusual perceptions or beliefs" not in domains


def test_first_response_asks_for_duration_when_unknown():
    reply = continue_conversation(
        "I feel anxious about everything.",
        support_mode="Help me understand my feelings",
    )
    assert reply.state.pending_domain == "duration"
    assert "how long" in reply.response.lower()


def test_existing_duration_moves_to_functioning():
    reply = continue_conversation(
        "I have felt anxious every day for three weeks.",
        support_mode="Help me understand my feelings",
    )
    assert reply.state.pending_domain == "functioning"
    assert reply.state.support_urgency == "professional"


def test_conversation_accumulates_context_across_turns():
    state = ConversationState()
    first = continue_conversation("I feel low and have lost interest.", state)
    second = continue_conversation("It has been happening for a month.", first.state)
    third = continue_conversation("It is affecting work and I stay in bed.", second.state)
    assert "depressed mood" in third.state.accumulated_domains
    assert third.state.duration_known
    assert third.state.functioning_known
    assert third.state.support_urgency == "professional"


def test_direct_crisis_language_interrupts_normal_questions():
    reply = continue_conversation("I want to end my life.")
    assert reply.state.safety_followup
    assert reply.recommendation_type == "urgent safety support"
    assert "immediate danger" in reply.response.lower()


def test_crisis_resources_follow_the_selected_country():
    reply = continue_conversation(
        "I feel like dying.",
        country_code="US",
    )
    assert "988" in reply.response
    assert "911" in reply.response
    assert "14416" not in reply.response


def test_negative_safety_answer_is_not_treated_as_emergency():
    state = ConversationState(safety_followup=True, pending_domain="safety")
    reply = continue_conversation("No, I don't have a plan.", state)
    assert reply.recommendation_type == "urgent professional support"
    assert reply.state.support_urgency == "urgent"
    assert "emergency help now" not in reply.response.lower()


def test_uncertain_safety_answer_remains_urgent():
    state = ConversationState(safety_followup=True, pending_domain="safety")
    reply = continue_conversation("I don't know.", state)
    assert reply.recommendation_type == "urgent safety support"
    assert "treat this as urgent" in reply.response.lower()


def test_mixed_no_plan_but_might_act_stays_emergency_level():
    state = ConversationState(safety_followup=True, pending_domain="safety")
    reply = continue_conversation("No plan, but I might act on it.", state)
    assert reply.recommendation_type == "emergency help now"
    assert "112" in reply.response


def test_just_listen_validates_grief_without_advice():
    reply = continue_conversation(
        "My grandfather passed away months ago but I still cry about it.",
        support_mode="Just listen",
    )
    lowered = reply.response.lower()
    assert "grief does not follow a fixed timetable" in lowered
    assert "what do you miss most" in lowered
    assert "try " not in lowered
    assert "a gentle place to start" not in lowered
    assert reply.recommendation_type == "routine"


def test_advice_mode_validates_before_practical_guidance():
    reply = continue_conversation(
        "My grandfather passed away months ago but I still cry about it.",
        support_mode="Give me advice",
    )
    lowered = reply.response.lower()
    validation_index = lowered.index("it makes sense")
    guidance_index = lowered.index("a gentle place to start")
    reasoning_index = lowered.index("this approach is meant")
    question_index = lowered.index("when does the grief")
    assert validation_index < guidance_index < reasoning_index < question_index


def test_challenge_mode_does_not_lead_with_debate():
    reply = continue_conversation(
        "Everyone thinks I am a failure.",
        support_mode="Challenge my thinking",
    )
    lowered = reply.response.lower()
    assert lowered.index("that thought sounds painful") < lowered.index(
        "one possibility to check"
    )


def test_grief_challenge_targets_self_judgment_not_the_feeling():
    reply = continue_conversation(
        "My grandfather passed away months ago but I still cry about it.",
        support_mode="Challenge my thinking",
    )
    lowered = reply.response.lower()
    assert "what are you telling yourself it means" in lowered
    assert "the feeling itself does not need to be argued away" in lowered


def test_safety_overrides_just_listen_mode():
    reply = continue_conversation(
        "I feel like dying.",
        support_mode="Just listen",
    )
    assert "112" in reply.response
    assert "14416" in reply.response


def test_king_statement_gets_casual_not_therapeutic_response():
    reply = continue_conversation("I am the king")
    assert reply.route.intent == "casual conversation"
    assert "king of what" in reply.response.lower()
    assert "what feels most present" not in reply.response.lower()


def test_university_business_question_compares_named_options():
    reply = continue_conversation(
        "Should I go to university or just choose my business?"
    )
    lowered = reply.response.lower()
    assert reply.route.intent == "decision support"
    assert "university may offer" in lowered
    assert "business may offer" in lowered
    assert "is there a specific choice" not in lowered


def test_procrastination_question_receives_practical_steps():
    reply = continue_conversation("How can I stop procrastinating?")
    assert reply.route.intent == "practical advice"
    assert "five-minute starting action" in reply.response


def test_explicit_advice_request_gives_advice_before_clarifying():
    reply = continue_conversation(
        "I don't know what to do. Suggest me something."
    )
    lowered = reply.response.lower()

    assert reply.route.intent == "practical advice"
    assert reply.route.response_mode == "Give me advice"
    assert lowered.index("a practical starting plan") < lowered.index(
        "what problem feels most urgent"
    )
    assert "what feels most present" not in lowered
    assert "effect on your daily life matters" not in lowered
    assert lowered.count("?") <= 1


def test_emotional_advice_request_does_not_continue_reflection_mode():
    reply = continue_conversation(
        "I feel completely lost and overwhelmed. What should I do?"
    )
    lowered = reply.response.lower()

    assert reply.route.intent == "practical advice"
    assert "a practical starting plan" in lowered
    assert "what part of this is hardest" not in lowered
    assert lowered.count("?") <= 1


def test_explicit_venting_request_receives_no_advice():
    reply = continue_conversation(
        "I don't need advice. I just want to complain."
    )
    assert reply.route.intent == "venting"
    assert "i will not turn it into advice" in reply.response.lower()


def test_factual_route_does_not_fake_current_information_without_web():
    reply = continue_conversation(
        "What are the visa requirements for Germany in 2026?"
    )
    assert reply.route.knowledge_route == "internet"
    assert "i don't want to guess" in reply.response.lower()
    assert "api key" not in reply.response.lower()
    assert "backend" not in reply.response.lower()


def test_research_failure_does_not_claim_to_have_read_papers():
    reply = continue_conversation(
        "What does research say about cognitive reframing?"
    )
    lowered = reply.response.lower()
    assert reply.route.intent == "research paper question"
    assert "not performed a live literature search" in lowered
    assert "widely cited foundational works" in lowered
    assert "source" in lowered
    assert "api" not in lowered


def test_laptop_recommendation_requires_current_product_search():
    reply = continue_conversation("Which laptop should I buy for data science?")
    assert reply.route.intent == "current factual search"
    assert reply.route.knowledge_route == "internet"


def test_health_recommendation_uses_advice_mode_and_authoritative_internet():
    reply = continue_conversation("What treatment is recommended for insomnia?")
    assert reply.route.intent == "practical advice"
    assert reply.route.response_mode == "Give me advice"
    assert reply.route.knowledge_route == "internet"


def test_cat_grief_uses_empathy_without_live_search():
    reply = continue_conversation("My cat died today.")
    assert reply.route.intent == "grief"
    assert reply.route.knowledge_route != "internet"
    assert "grief does not follow a fixed timetable" in reply.response.lower()


def test_explicit_research_suggestion_overrides_ambiguous_next_message():
    reply = continue_conversation(
        "Does it actually work?",
        knowledge_override="research papers",
    )
    assert reply.route.intent == "research paper question"
    assert reply.route.knowledge_route == "research papers"


def test_emotional_followups_do_not_repeat_stock_phrase():
    state = ConversationState()
    first = continue_conversation("I feel low and have lost interest.", state)
    second = continue_conversation("It has been happening for a month.", first.state)
    third = continue_conversation("It affects my work.", second.state)
    assert "that adds something important" not in second.response.lower()
    assert second.response.split("\n\n", 1)[0] != third.response.split("\n\n", 1)[0]


def test_local_decision_support_uses_approved_memory_without_openai():
    reply = continue_conversation(
        "Should I choose university or business?",
        approved_memory=("I value financial independence.",),
    )
    assert "context you asked me to remember" in reply.response.lower()
    assert "i value financial independence" in reply.response.lower()


def test_indirect_self_harm_wording_activates_safety():
    for text in (
        "I feel unsafe with myself.",
        "I don't want to be here anymore.",
        "I might do something bad to myself tonight.",
        "Nobody would care if I disappeared.",
    ):
        reply = continue_conversation(text)
        assert reply.route.intent == "crisis / safety"


def test_product_decision_fallback_compares_without_inventing_products():
    reply = continue_conversation("Which headphones should I buy under 70000?")
    lowered = reply.response.lower()
    assert reply.route.knowledge_route == "internet"
    assert "compare the options" in lowered
    assert "will not invent" in lowered


def test_mixed_complex_response_separates_prioritizes_and_recommends():
    reply = continue_conversation(
        "I feel overwhelmed about my career, university admissions, family pressure, "
        "money, and moving abroad. I don't know what to do about my future."
    )
    lowered = reply.response.lower()
    assert reply.route.intent == "mixed complex life problem"
    assert "separate it first" in lowered
    assert "priority" in lowered
    assert "my recommendation" in lowered
    assert reply.response.count("?") <= 1


def test_interview_failure_gets_overgeneralization_reframe_and_next_step():
    reply = continue_conversation("I feel like a failure because I failed one interview.")
    lowered = reply.response.lower()
    assert reply.route.intent == "cognitive challenge"
    assert "overgeneralization" in lowered
    assert "does not make you a failure" in lowered
    assert "next step" in lowered
    assert reply.response.count("?") <= 1


def test_leave_company_gets_workplace_guidance_not_option_request():
    reply = continue_conversation("Should I leave my company?")
    lowered = reply.response.lower()
    assert reply.route.intent == "decision support"
    assert reply.route.topic == "workplace"
    assert "do not recommend quitting impulsively" in lowered
    assert "update your resume" in lowered
    assert "tell me the options" not in lowered


def test_unfair_boss_quit_tomorrow_discourages_impulsive_quitting():
    reply = continue_conversation("My boss is unfair. Should I quit tomorrow?")
    lowered = reply.response.lower()
    assert reply.route.intent == "decision support"
    assert "do not recommend quitting impulsively" in lowered
    assert "document" in lowered
    assert "job market" in lowered or "resume" in lowered


def test_cognitive_science_data_science_compares_fields():
    reply = continue_conversation("Should I study Cognitive Science or Data Science?")
    lowered = reply.response.lower()
    assert reply.route.intent == "decision support"
    assert reply.route.topic == "education"
    assert "cognitive science" in lowered
    assert "data science" in lowered
    assert "job-oriented" in lowered


def test_grief_therapist_decision_validates_and_explains_when_therapy_helps():
    reply = continue_conversation(
        "I miss my grandfather every day. Should I see a therapist?"
    )
    lowered = reply.response.lower()
    assert reply.route.intent == "decision support"
    assert reply.route.topic == "grief"
    assert "missing your grandfather" in lowered
    assert "therapist could be a good idea" in lowered
    assert "sleep" in lowered
    assert "tell me the options" not in lowered
    assert reply.response.count("?") <= 1


def test_workplace_discrimination_decision_not_crisis_and_not_impulsive():
    reply = continue_conversation(
        "My boss only promotes women. I have been stuck for 5 years. "
        "I want to leave the company. Should I do it?"
    )
    lowered = reply.response.lower()
    assert reply.route.intent == "decision support"
    assert reply.route.topic == "workplace"
    assert reply.route.intent != "crisis / safety"
    assert "document" in lowered
    assert "promotion criteria" in lowered
    assert "still employed" in lowered


def test_requested_mixed_intent_example_prioritizes_tracks():
    reply = continue_conversation(
        "I miss my grandfather. I am waiting for Germany admissions. "
        "I feel behind in life. Part of me wants to start a business. "
        "What should I do?"
    )
    lowered = reply.response.lower()
    assert reply.route.intent == "mixed complex life problem"
    assert "separate the issues" in lowered
    assert "main track" in lowered
    assert "side experiment" in lowered
    assert "recommendation" in lowered
    assert reply.response.count("?") <= 1


def test_phase_false_positive_examples_do_not_trigger_crisis():
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
        reply = continue_conversation(text)
        assert reply.route.intent != "crisis / safety"


def test_psychology_data_science_tradeoff_gets_decision_support():
    reply = continue_conversation("I love psychology but Data Science pays more.")
    lowered = reply.response.lower()
    assert reply.route.intent == "decision support"
    assert reply.route.topic == "education"
    assert "psychology" in lowered
    assert "data science" in lowered
    assert "income security" in lowered or "earning potential" in lowered


def test_failed_two_interviews_gets_distortion_challenge():
    reply = continue_conversation(
        "I failed two interviews. I guess I am not smart enough."
    )
    lowered = reply.response.lower()
    assert reply.route.intent == "cognitive challenge"
    assert "overgeneralization" in lowered
    assert "not automatically mean" in lowered
    assert "next step" in lowered


def test_social_comparison_gets_social_comparison_challenge():
    reply = continue_conversation("Everyone my age is ahead of me.")
    lowered = reply.response.lower()
    assert reply.route.intent == "cognitive challenge"
    assert "social comparison" in lowered
    assert "next step" in lowered


def test_life_ruined_gets_catastrophizing_challenge():
    reply = continue_conversation("I am 24 and my life is already ruined.")
    lowered = reply.response.lower()
    assert reply.route.intent == "cognitive challenge"
    assert "catastrophizing" in lowered
    assert "next step" in lowered


def test_relationship_decision_uses_relationship_strategy():
    reply = continue_conversation(
        "My girlfriend and I keep fighting. I still love her. "
        "I am exhausted. Should I stay or leave?"
    )
    lowered = reply.response.lower()
    assert reply.route.intent == "decision support"
    assert reply.route.topic == "relationship"
    assert "communication" in lowered or "communicate" in lowered
    assert "boundaries" in lowered
    assert "repair" in lowered
    assert "tell me the options" not in lowered


def test_business_decision_discourages_impulsive_quitting():
    reply = continue_conversation("I want to quit my job and start a business tomorrow.")
    lowered = reply.response.lower()
    assert reply.route.intent == "decision support"
    assert reply.route.topic == "business"
    assert "runway" in lowered
    assert "validation" in lowered
    assert "do not quit impulsively" in lowered


def test_workplace_discrimination_statement_gives_documentation_guidance():
    reply = continue_conversation("I think my manager discriminates against me.")
    lowered = reply.response.lower()
    assert reply.route.intent == "practical advice"
    assert reply.route.topic == "workplace"
    assert "document" in lowered
    assert "hr" in lowered
    assert "qualified workplace" in lowered or "legal adviser" in lowered


def test_phase_mixed_complex_example_prioritizes_without_crisis():
    reply = continue_conversation(
        "My grandfather died last year. I am waiting for Germany admissions. "
        "I feel behind compared to my friends. Part of me wants to start a "
        "business. I am worried I am making the wrong decision with my life."
    )
    lowered = reply.response.lower()
    assert reply.route.intent == "mixed complex life problem"
    assert reply.route.intent != "crisis / safety"
    assert "separate" in lowered
    assert "priority" in lowered
    assert "recommendation" in lowered
    assert reply.response.count("?") <= 1


def test_casual_chat_works_without_psychology_mode():
    for text in ("Tell me a terrible joke.", "What Pokemon would make the best therapist?"):
        reply = continue_conversation(text)
        lowered = reply.response.lower()
        assert reply.route.intent == "casual conversation"
        assert "what feels most present" not in lowered
        assert "daily life" not in lowered


def test_clinical_overlap_note_is_non_diagnostic():
    reply = continue_conversation(
        "I feel hopeless, lost interest in everything, cannot focus, and cannot get out of bed."
    )
    lowered = reply.response.lower()
    assert "overlap" in lowered
    assert "clinician" in lowered
    assert "not mean any condition is present" in lowered
    assert "you have depression" not in lowered
    assert "you have adhd" not in lowered
    assert "you have bipolar" not in lowered
    assert reply.symptom_profile["depressive_symptoms"] >= 3
    assert reply.possible_clinical_overlaps


def test_symptom_profile_accumulates_across_turns_without_diagnosis():
    state = ConversationState()
    first = continue_conversation("I feel worried and on edge.", state)
    second = continue_conversation("I also cannot sleep and cannot focus.", first.state)
    assert second.state.symptom_profile["anxiety_symptoms"] >= 2
    assert second.state.symptom_profile["attention_regulation_symptoms"] >= 1
    assert second.state.possible_clinical_overlaps
    assert "you have" not in second.response.lower()


def test_conversation_state_tracks_grief_confusion_overwhelm_progression():
    state = ConversationState()
    first = continue_conversation("My grandfather died and I am sad.", state)
    second = continue_conversation("I don't know what to do.", first.state)
    third = continue_conversation("I feel suffocated.", second.state)

    assert "grief" in third.state.active_themes
    assert "grandfather" in third.state.active_relationships
    assert "grief" in third.state.active_emotions
    assert "confusion" in third.state.active_emotions
    assert "overwhelm" in third.state.active_emotions
    assert third.state.narrative_progression[-3:] == [
        "grief",
        "confusion",
        "overwhelm",
    ]
    assert "grief" in third.state.conversation_summary
    assert "uncertainty about what to do next" in third.state.unresolved_concerns
    assert "feeling suffocated or overwhelmed" in third.state.unresolved_concerns


def test_followup_response_uses_prior_grief_context_instead_of_restarting():
    state = ConversationState()
    first = continue_conversation("My grandfather died and I am sad.", state)
    second = continue_conversation("I don't know what to do.", first.state)
    third = continue_conversation("I feel suffocated.", second.state)
    lowered = third.response.lower()

    assert "grandfather" in lowered or "grief" in lowered
    assert "thread" in lowered or "connected" in lowered
    assert "what happened" not in lowered
    assert "what came next" not in lowered


def test_intervention_request_gives_non_diagnostic_support_options():
    reply = continue_conversation("Suggest me some therapies for grief.")
    lowered = reply.response.lower()

    assert reply.route.intent == "intervention_request"
    assert reply.route.topic == "grief"
    assert "grief counseling" in lowered
    assert "support groups" in lowered
    assert "cbt" in lowered
    assert "not a diagnosis" in lowered
    assert "you have" not in lowered


def test_humanization_reflects_before_question_for_low_self_worth():
    reply = continue_conversation("I feel empty, tired, and worthless.")
    lowered = reply.response.lower()

    assert lowered.startswith("what you described sounds exhausting")
    assert lowered.index("exhausting") < lowered.index("?")
    assert "there is no need to turn this into a problem" not in lowered
    assert reply.possible_explanations
    assert reply.reasoning_style == "reflective"


def test_adhd_question_explains_alternatives_without_diagnosis():
    reply = continue_conversation("I can't focus. Do I have ADHD?")
    lowered = reply.response.lower()

    assert reply.route.intent == "health / wellness information"
    assert "adhd-like attention regulation" in lowered
    assert "burnout" in lowered
    assert "anxiety" in lowered
    assert "depression-related low energy" in lowered
    assert "poor sleep" in lowered
    assert "since childhood" in lowered
    assert "this is not a diagnosis" in lowered
    assert "you have adhd" not in lowered
    assert len(reply.possible_explanations) >= 3


def test_response_style_rotates_across_turns():
    state = ConversationState()
    first = continue_conversation("I feel anxious about tomorrow.", state)
    second = continue_conversation("How can I stop procrastinating?", first.state)
    third = continue_conversation("Suggest me some therapies for grief.", second.state)

    assert len(third.state.last_5_styles) >= 3
    assert len(set(third.state.last_5_styles)) >= 3
    assert third.state.reasoning_style == third.reasoning_style


def test_identity_exploration_goes_deeper_than_surface_question():
    reply = continue_conversation("I don't know who I am anymore.")
    lowered = reply.response.lower()

    assert reply.route.intent == "identity_exploration"
    assert "old identity" in lowered
    assert "values" in lowered
    assert "approval" in lowered
    assert "who am i" in lowered


def test_v13_safety_precision_does_not_trigger_for_non_crisis_contexts():
    examples = (
        "My grandfather died yesterday and I feel like disappearing.",
        "I don't want to be here anymore in this relationship.",
        "I can't go on with this career uncertainty.",
        "I don't know who I am anymore.",
        "I can't focus. Do I have ADHD?",
        "What is the point of life if everyone dies eventually?",
    )
    for text in examples:
        reply = continue_conversation(text)
        assert reply.route.intent != "crisis / safety", text


def test_v13_safety_still_triggers_for_actual_self_harm_intent():
    examples = (
        "I am going to kill myself tonight.",
        "I have a suicide plan.",
        "I want to end my life.",
        "I might hurt myself now.",
    )
    for text in examples:
        reply = continue_conversation(text)
        assert reply.route.intent == "crisis / safety", text


def test_v13_identity_variants_route_to_identity_exploration():
    examples = (
        "I don't recognize myself anymore.",
        "Who am I anymore?",
        "I don't know what my real self is.",
    )
    for text in examples:
        reply = continue_conversation(text)
        assert reply.route.intent == "identity_exploration", text


def test_v13_narrative_memory_answers_stress_driver_question():
    state = ConversationState()
    first = continue_conversation("My grandfather died last year.", state)
    second = continue_conversation(
        "I am waiting for Germany admissions and I feel behind compared to friends.",
        first.state,
    )
    third = continue_conversation(
        "I love psychology but Data Science pays more and my family expects stability.",
        second.state,
    )
    final = continue_conversation(
        "What do you think is actually driving most of my stress?",
        third.state,
    )
    lowered = final.response.lower()

    assert final.route.intent == "narrative_memory"
    assert "loss of grandfather" in final.state.major_losses
    assert "falling behind others" in final.state.recurring_fears
    assert "financial security" in final.state.recurring_values
    assert "uncertainty about the future" in lowered or "future" in lowered
    assert "emotional weight" in lowered


def test_v13_ethical_reasoning_avoids_generic_decision_template():
    reply = continue_conversation(
        "My company is doing something unethical but legal. What should I do?"
    )
    lowered = reply.response.lower()

    assert reply.route.intent == "ethical_dilemma"
    assert "harm" in lowered
    assert "responsibility" in lowered or "autonomy" in lowered
    assert "tell me the options" not in lowered
