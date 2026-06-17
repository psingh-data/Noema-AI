from ai.tavily import should_use_tavily
from core.conversation import continue_conversation
from core.language_ontology import match_language_ontology


def test_im_cooked_routes_to_overwhelm_without_internet():
    reply = continue_conversation("I'm cooked")
    match = reply.language_ontology_match

    assert match.matched
    assert match.category == "pure_emotional_validation"
    assert match.canonical_emotion == "overwhelm"
    assert match.register == "gen_z_slang"
    assert reply.route.intent == "emotional reflection"
    assert reply.route.knowledge_route == "conversation context"
    assert not should_use_tavily(reply.route)
    assert "overload" in reply.response.lower()


def test_let_me_yap_routes_to_venting_without_advice_or_internet():
    reply = continue_conversation("let me yap")
    lowered = reply.response.lower()

    assert reply.language_ontology_match.category == "venting"
    assert reply.route.intent == "venting"
    assert reply.route.knowledge_route == "conversation context"
    assert not should_use_tavily(reply.route)
    assert "will not turn this into advice" in lowered


def test_no_one_got_me_fr_routes_to_loneliness_without_internet():
    reply = continue_conversation("no one got me fr")

    assert reply.language_ontology_match.category == "loneliness"
    assert reply.language_ontology_match.register == "gen_z_slang"
    assert reply.route.intent == "emotional reflection"
    assert reply.route.topic == "loneliness"
    assert reply.route.knowledge_route == "conversation context"
    assert not should_use_tavily(reply.route)
    assert "not really seeing you" in reply.response.lower()


def test_i_lost_my_plot_routes_to_identity_reflection_without_internet():
    reply = continue_conversation("I lost my plot")

    assert reply.language_ontology_match.category == "identity_reflection"
    assert reply.route.intent == "identity_exploration"
    assert reply.route.knowledge_route == "conversation context"
    assert not should_use_tavily(reply.route)
    assert "old version" in reply.response.lower()


def test_relationship_draining_fr_routes_to_relationship_feelings_without_internet():
    reply = continue_conversation("this relationship is draining fr")
    lowered = reply.response.lower()

    assert reply.language_ontology_match.category == "relationship_feelings"
    assert reply.route.topic == "relationship"
    assert reply.route.knowledge_route == "conversation context"
    assert not should_use_tavily(reply.route)
    assert "relationship" in lowered
    assert "draining" in lowered


def test_starter_rapport_yo_is_casual_not_clinical():
    reply = continue_conversation("yo")
    lowered = reply.response.lower()

    assert reply.language_ontology_match.category == "casual_conversation"
    assert reply.route.intent == "casual conversation"
    assert reply.route.knowledge_route == "conversation context"
    assert not should_use_tavily(reply.route)
    assert "what feels most present" not in lowered
    assert "daily life" not in lowered


def test_evidence_based_therapies_for_grief_uses_internet_and_internal_context():
    reply = continue_conversation("suggest evidence-based therapies for grief")

    assert reply.route.intent == "intervention_request"
    assert reply.route.knowledge_route == "research papers"
    assert should_use_tavily(reply.route)
    assert reply.language_ontology_match.internet_needed


def test_grandfather_died_yesterday_is_grief_disclosure_without_internet():
    reply = continue_conversation("my grandfather died yesterday")
    lowered = reply.response.lower()

    assert reply.language_ontology_match.category == "grief_disclosure"
    assert reply.route.intent == "grief"
    assert reply.route.knowledge_route == "conversation context"
    assert not should_use_tavily(reply.route)
    assert "grief does not follow a fixed timetable" in lowered


def test_semantic_matcher_understands_requested_examples():
    expected = {
        "I'm cooked": "pure_emotional_validation",
        "let me yap": "venting",
        "no one got me fr": "loneliness",
        "I lost my plot": "identity_reflection",
        "this relationship is draining fr": "relationship_feelings",
        "yo": "casual_conversation",
        "my grandfather died yesterday": "grief_disclosure",
    }
    for text, category in expected.items():
        assert match_language_ontology(text).category == category
