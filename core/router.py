"""Route each message to an interaction style and knowledge source."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.language_ontology import match_language_ontology, route_for_language_category


@dataclass(frozen=True)
class RouteDecision:
    intent: str
    response_mode: str
    knowledge_route: str
    confidence: float
    reason: str
    topic: str = "general"


EXPLICIT_VENTING = (
    "don't need advice",
    "do not need advice",
    "no advice",
    "just listen",
    "let me vent",
    "need to vent",
    "want to vent",
    "just want to complain",
    "just let me complain",
    "let me complain",
)

RESEARCH_MARKERS = (
    "research say",
    "studies say",
    "evidence say",
    "evidence for",
    "any studies",
    "latest papers",
    "foundational papers",
    "recent studies",
    "scientific evidence",
    "scientific basis",
    "does cognitive reframing work",
    "research paper",
    "research papers",
    "peer reviewed",
    "peer-reviewed",
    "literature review",
    "meta-analysis",
    "systematic review",
    "pubmed",
    "arxiv",
)

INTERVENTION_MARKERS = (
    "suggest therapies",
    "suggest me therapies",
    "what therapy helps",
    "what therapies help",
    "what kind of therapy",
    "therapy helps",
    "therapy for",
    "therapies for",
    "interventions",
    "evidence-based techniques",
    "evidence based techniques",
    "coping methods",
    "coping strategies for",
    "what can help with",
    "what can i do when grief",
    "how to go through these emotions",
    "go through these emotions",
    "coping methods for",
    "coping methods",
    "treatment options",
    "research-backed support",
    "research backed support",
)

FAILED_INTERVENTION_MARKERS = (
    "tried breathing",
    "tried grounding",
    "breathing but it did not help",
    "breathing but it didn't help",
    "grounding exercise but it had no effect",
    "did not help",
    "didn't help",
    "had no effect",
)

IDENTITY_EXPLORATION_MARKERS = (
    "someone i don't recognize",
    "someone i do not recognize",
    "don't recognize myself",
    "do not recognize myself",
    "who i am anymore",
    "who am i anymore",
    "parts of me are real",
    "my real self",
    "not myself anymore",
    "became someone",
    "identity",
    "wasting my life",
    "waste my life",
    "wasted my life",
)

ACHIEVEMENT_SELF_WORTH_MARKERS = (
    "built around achievement",
    "only feel valuable",
    "productive",
    "if i fail",
    "performance as proof",
    "worth depends",
)

EXISTENTIAL_MARKERS = (
    "free will",
    "point of life",
    "why does anything matter",
    "does anything matter",
    "why care about",
    "everything ends",
    "everyone dies",
    "why should i try",
    "meaning of life",
)

ETHICAL_MARKERS = (
    "painful truth",
    "comforting lie",
    "unethical but legal",
    "ethical",
    "morally",
    "cheated in an exam",
    "should i report",
    "report him",
    "report her",
    "report wrongdoing",
    "wrongdoing",
)

STRUCTURED_PROBLEM_MARKERS = (
    "organize my life",
    "everything feels scattered",
    "productivity system",
    "adhd-like symptoms",
    "adhd like symptoms",
    "design a productivity",
    "build a system",
)

USER_FRUSTRATION_MARKERS = (
    "this is not helping",
    "not helping",
    "same thing again",
    "already said that",
    "repeating",
)

CONTINUITY_REPAIR_MARKERS = (
    "i tried that",
    "that did not help",
    "that didn't help",
    "previous suggestion",
)

FACTUAL_MARKERS = (
    "visa requirement",
    "visa rules",
    "student visa",
    "requirements for",
    "official requirement",
    "application deadline",
    "admission deadline",
    "admission information",
    "deadline for",
    "what's the deadline",
    "when is the deadline",
    "how much does",
    "salary in",
    "salary for",
    "average salary",
    "salary",
    "job market",
    "market trend",
    "cost of",
    "price of",
    "latest news",
    "news",
    "ai news",
    "news about",
    "today's news",
    "todays news",
    "what's new in",
    "whats new in",
    "current news",
    "current event",
    "what happened",
    "latest tool",
    "best tool",
    "tool for",
    "software tool",
    "software",
    "software update",
    "software version",
    "framework version",
    "framework",
    "library version",
    "library",
    "latest version",
    "product update",
    "service update",
    "new regulation",
    "current regulation",
    "regulation for",
    "regulations for",
    "regulation",
    "regulations",
    "law in 2025",
    "law in 2026",
    "near me",
    "common symptoms",
    "what are the symptoms",
    "what is the deadline",
    "which laptop",
    "best laptop",
    "recommend a laptop",
    "which computer",
    "best computer",
    "which phone",
    "best phone",
    "which product",
    "best product",
    "should i buy",
    "worth buying",
    "product recommendation",
    "product comparison",
    "compare products",
    "compare laptops",
    "compare laptop",
    "laptop vs",
    "phone vs",
    "health guideline",
    "treatment guideline",
)

DECISION_MARKERS = (
    "should i",
    "which should i",
    "help me decide",
    "help me choose",
    "confused between",
    "choose between",
    "decide between",
    "torn between",
    "whether i should",
    "is it the right move",
    "is this the right move",
    "wrong decision",
    "making the wrong decision",
    "should i do it",
    "if you were me",
    "would you",
)

ADVICE_MARKERS = (
    "what should i do",
    "suggest something",
    "suggest me something",
    "suggest a practical plan",
    "give me advice",
    "help me decide",
    "recommend",
    "any advice",
    "any ideas",
    "what to do",
    "wrong decision",
    "making the wrong decision",
    "should i do it",
    "how can i",
    "how do i",
    "tips for",
    "help me stop",
    "help me start",
    "need steps",
    "be practical",
)

CAREER_EDUCATION_MARKERS = (
    "career",
    "job interview",
    "resume",
    "cv ",
    "degree",
    "course",
    "university",
    "college",
    "major in",
    "study ",
    "learning path",
    "become a",
    "become an",
)

HEALTH_WELLNESS_MARKERS = (
    "health",
    "wellness",
    "symptom",
    "diagnosis",
    "treatment",
    "medication",
    "medicine",
    "side effect",
    "nutrition",
    "diet",
    "exercise",
    "sleep problem",
    "insomnia",
    "burnout",
)

CASUAL_MARKERS = (
    "i am the king",
    "i'm the king",
    "guess what",
    "just kidding",
    "tell me a joke",
    "terrible joke",
    "bad joke",
    "make me laugh",
    "pokemon",
    "pokémon",
    "lol",
    "haha",
    "what's up",
    "whats up",
)

EMOTIONAL_UNCERTAINTY_MARKERS = (
    "i feel weird",
    "feel strange",
    "don't know what i need",
    "do not know what i need",
    "i don't know how i feel",
    "i do not know how i feel",
)

EMOTIONAL_DISCLOSURE_MARKERS = (
    "i feel ",
    "i'm feeling ",
    "im feeling ",
    "overwhelming me",
    "makes me feel",
    "stresses me",
    "upsets me",
    "scares me",
    "hopeless about",
    "lonely",
)

GRIEF_LIKE_MARKERS = (
    "miss my",
    "missing my",
    "memory of my",
    "reminds me of",
    "broke down",
    "still cry",
)

FUTURE_UNCERTAINTY_MARKERS = (
    "future",
    "what if",
    "uncertain",
    "don't know",
    "do not know",
    "stuck",
    "behind",
    "feel behind",
    "worried about",
    "afraid that",
    "next year",
    "admission",
    "career path",
    "move abroad",
    "settle",
)

MAJOR_TOPIC_MARKERS = {
    "career": ("career", "job", "company", "boss", "work"),
    "education": ("university", "college", "study", "degree", "admission", "exam", "education"),
    "relationship": (
        "relationship",
        "girlfriend",
        "boyfriend",
        "partner",
        "family",
        "parents",
        "sister",
        "brother",
    ),
    "health": ("health", "doctor", "sleep", "burnout", "symptom", "treatment"),
    "money": ("money", "salary", "income", "financial", "rent", "debt", "invest", "lakh"),
    "location": ("move abroad", "visa", "country", "city", "relocate", "germany"),
    "identity": ("purpose", "future", "life", "identity", "not myself", "behind"),
    "business": ("business", "startup", "brand", "customers"),
    "grief": GRIEF_LIKE_MARKERS
    + ("grandfather", "grandmother", "grieving", "died", "death", "passed away", "loss"),
}

TOPIC_PRIORITY = (
    "grief",
    "workplace",
    "education",
    "business",
    "career",
    "health",
    "finance",
    "family",
    "relationship",
    "self_esteem",
    "motivation",
)

TOPIC_MARKERS = {
    "grief": GRIEF_LIKE_MARKERS + (
        "grandfather",
        "grandmother",
        "grief",
        "died",
        "death",
        "passed away",
        "loss",
    ),
    "workplace": (
        "boss",
        "company",
        "promotion",
        "promotes",
        "promoted",
        "workplace",
        "hr",
        "manager",
        "unfair",
        "discriminat",
        "discriminates",
        "discrimination",
        "harassment",
        "stuck for",
    ),
    "education": (
        "cognitive science",
        "data science",
        "psychology",
        "university",
        "college",
        "study",
        "education",
        "degree",
        "admission",
        "exam",
    ),
    "career": (
        "career",
        "job",
        "interview",
        "resume",
        "cv",
        "salary",
        "role",
        "profession",
    ),
    "business": ("business", "startup", "brand", "customers", "income", "product"),
    "health": ("health", "therapist", "therapy", "doctor", "sleep", "symptom"),
    "finance": ("money", "finance", "financial", "debt", "rent", "budget"),
    "family": ("family", "parents", "sister", "brother", "mother", "father"),
    "relationship": (
        "relationship",
        "girlfriend",
        "boyfriend",
        "partner",
        "breakup",
        "breaking up",
        "coming back",
    ),
    "self_esteem": (
        "failure",
        "failed",
        "not good enough",
        "not smart enough",
        "worthless",
        "hate myself",
        "insecure",
    ),
    "motivation": ("motivation", "procrastinat", "discipline", "can't start"),
}


SUGGESTIONS = {
    "grief": ("Listen more", "Give advice"),
    "anxiety / stress": ("Listen more", "Give advice"),
    "overwhelm": ("Listen more", "Give advice"),
    "emotional reflection": (
        "Listen more",
        "Give advice",
        "Challenge this thought",
    ),
    "venting": ("Listen more", "Give advice"),
    "decision support": ("Help me decide", "Challenge this thought"),
    "practical advice": ("Give advice", "Find research evidence"),
    "career / education": ("Give advice", "Help me decide", "Search current facts"),
    "health / wellness information": ("Search current facts", "Find research evidence"),
    "cognitive challenge": ("Challenge this thought", "Listen more"),
    "current factual search": ("Search current facts",),
    "research paper question": ("Find research evidence",),
    "intervention_request": ("Find research evidence", "Give advice"),
    "general knowledge": ("Give advice", "Search current facts"),
    "mixed complex life problem": ("Give advice", "Help me decide"),
    "identity_exploration": ("Listen more", "Challenge this thought"),
    "achievement_self_worth": ("Challenge this thought", "Give advice"),
    "existential_question": ("Listen more", "Give advice"),
    "ethical_dilemma": ("Give advice", "Challenge this thought"),
    "structured_problem_solving": ("Give advice",),
    "failed_intervention_repair": ("Give advice", "Listen more"),
    "user_frustration_repair": ("Listen more", "Give advice"),
    "conversation_continuity": ("Listen more", "Give advice"),
}


SUGGESTION_MODES = {
    "Listen more": "Just listen",
    "Give advice": "Give me advice",
    "Help me decide": "Help me make a decision",
    "Challenge this thought": "Challenge my thinking",
}

SUGGESTION_KNOWLEDGE = {
    "Search current facts": "internet",
    "Find research evidence": "research papers",
}


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def explicit_advice_requested(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return _has_any(normalized, ADVICE_MARKERS)


def explicit_decision_requested(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return _has_any(normalized, DECISION_MARKERS)


def _named_decision_requested(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return explicit_decision_requested(normalized) and (
        " or " in normalized
        or "between" in normalized
        or "right move" in normalized
        or "should i do it" in normalized
        or "if you were me" in normalized
        or "would you" in normalized
    )


def _major_topics(text: str) -> set[str]:
    return {
        topic
        for topic, markers in MAJOR_TOPIC_MARKERS.items()
        if _has_any(text, markers)
    }


def detect_problem_topics(text: str) -> tuple[str, ...]:
    normalized = " ".join(text.lower().split())
    topics = [
        topic
        for topic in TOPIC_PRIORITY
        if _has_any(normalized, TOPIC_MARKERS[topic])
    ]
    return tuple(dict.fromkeys(topics)) or ("general",)


def detect_problem_topic(text: str) -> str:
    return detect_problem_topics(text)[0]


def _has_grief_signal(text: str, category: str, emotion: str) -> bool:
    return category == "grief" or emotion == "grief" or _has_any(text, GRIEF_LIKE_MARKERS)


def _is_mixed_complex_life_problem(
    text: str,
    *,
    emotional_content: bool,
) -> bool:
    return (
        len(_major_topics(text)) >= 3
        and emotional_content
        and _has_any(text, FUTURE_UNCERTAINTY_MARKERS)
        and explicit_advice_requested(text)
    )


def _looks_like_question(text: str) -> bool:
    return text.rstrip().endswith("?") or bool(
        re.match(r"^(what|when|where|who|why|how|is|are|can|does|do)\b", text)
    )


def _is_emotional_statement(
    text: str,
    *,
    category: str,
    emotion: str,
    clinical_domains: tuple[str, ...],
) -> bool:
    if _looks_like_question(text):
        return False
    return (
        category
        in {
            "grief",
            "anxiety",
            "overwhelm",
            "relationship",
            "self-esteem",
            "loneliness",
        }
        or emotion != "neutral"
        or bool(clinical_domains)
        or _has_any(text, EMOTIONAL_UNCERTAINTY_MARKERS)
        or _has_any(text, EMOTIONAL_DISCLOSURE_MARKERS)
    )


def suggestions_for_intent(intent: str) -> tuple[str, ...]:
    return SUGGESTIONS.get(intent, ())


def mode_for_suggestion(label: str) -> str | None:
    return SUGGESTION_MODES.get(label)


def knowledge_for_suggestion(label: str) -> str | None:
    return SUGGESTION_KNOWLEDGE.get(label)


def route_message(
    text: str,
    *,
    preferred_mode: str,
    category: str,
    emotion: str,
    has_bias: bool,
    clinical_domains: tuple[str, ...],
    prior_intent: str = "open conversation",
    has_active_context: bool = False,
) -> RouteDecision:
    normalized = " ".join(text.lower().split())
    topic = detect_problem_topic(normalized)
    language_match = match_language_ontology(text)
    emotional_statement = _is_emotional_statement(
        normalized,
        category=category,
        emotion=emotion,
        clinical_domains=clinical_domains,
    )
    emotional_content = (
        emotional_statement
        or _has_any(normalized, EMOTIONAL_DISCLOSURE_MARKERS)
        or category
        in {
            "grief",
            "anxiety",
            "overwhelm",
            "relationship",
            "self-esteem",
            "loneliness",
        }
        or emotion != "neutral"
    )

    if _has_any(normalized, EXPLICIT_VENTING):
        return RouteDecision(
            "venting",
            "Just listen",
            "conversation context",
            0.98,
            "The user explicitly asked not to receive advice.",
            topic,
        )

    if _has_any(normalized, RESEARCH_MARKERS):
        return RouteDecision(
            "research paper question",
            "Research Assistant",
            "research papers",
            0.96,
            "The message explicitly asks what research or evidence says.",
            topic,
        )

    if _has_any(normalized, INTERVENTION_MARKERS):
        return RouteDecision(
            "intervention_request",
            "Research Assistant",
            "research papers",
            0.92,
            "The user asks for therapy, intervention, or evidence-informed support options.",
            topic,
        )

    language_route = (
        route_for_language_category(language_match.category)
        if (
            language_match.matched
            and not language_match.internet_needed
            and not explicit_advice_requested(normalized)
            and not explicit_decision_requested(normalized)
            and preferred_mode
            not in {
                "Give me advice",
                "Help me make a decision",
                "Challenge my thinking",
            }
        )
        else None
    )
    if language_route is not None:
        intent, response_mode, language_topic = language_route
        return RouteDecision(
            intent,
            response_mode,
            "conversation context",
            max(0.82, min(language_match.confidence, 0.99)),
            (
                "The offline Noema language ontology matched "
                f"{language_match.category} without needing internet."
            ),
            topic if topic != "general" else language_topic,
        )

    if _has_any(normalized, USER_FRUSTRATION_MARKERS):
        return RouteDecision(
            "user_frustration_repair",
            "Repair conversation",
            "conversation context",
            0.9,
            "The user is saying the response pattern is not helping.",
            topic,
        )

    if _has_any(normalized, CONTINUITY_REPAIR_MARKERS):
        return RouteDecision(
            "conversation_continuity",
            "Conversation continuity",
            "conversation context",
            0.88,
            "The user refers to a prior suggestion or earlier thread.",
            topic,
        )

    if _has_any(normalized, FAILED_INTERVENTION_MARKERS):
        return RouteDecision(
            "failed_intervention_repair",
            "Give me advice",
            "conversation context",
            0.9,
            "The user says a coping intervention did not work and needs a different approach.",
            topic,
        )

    if _has_any(normalized, IDENTITY_EXPLORATION_MARKERS):
        return RouteDecision(
            "identity_exploration",
            "Help me understand my feelings",
            "conversation context",
            0.87,
            "The user is exploring identity or loss of self-continuity.",
            topic,
        )

    if _has_any(normalized, ACHIEVEMENT_SELF_WORTH_MARKERS):
        return RouteDecision(
            "achievement_self_worth",
            "Challenge my thinking",
            "conversation context",
            0.88,
            "The message links worth or identity to achievement and performance.",
            topic if topic != "general" else "self_esteem",
        )

    if _has_any(normalized, EXISTENTIAL_MARKERS):
        return RouteDecision(
            "existential_question",
            "Reflective conversation",
            "conversation context",
            0.86,
            "The message asks a philosophical or existential question with emotional weight.",
            topic,
        )

    if _has_any(normalized, ETHICAL_MARKERS):
        return RouteDecision(
            "ethical_dilemma",
            "Structured reflection",
            "conversation context",
            0.86,
            "The message asks for practical reasoning about an ethical dilemma.",
            topic,
        )

    if _has_any(normalized, STRUCTURED_PROBLEM_MARKERS):
        return RouteDecision(
            "structured_problem_solving",
            "Give me advice",
            "conversation context",
            0.89,
            "The user asks for an organized system or practical problem-solving structure.",
            topic,
        )

    if _has_any(normalized, CASUAL_MARKERS):
        return RouteDecision(
            "casual conversation",
            "Friend",
            "conversation context",
            0.9,
            "The message is casual, playful, or asks for light conversation.",
            topic,
        )

    if _is_mixed_complex_life_problem(
        normalized,
        emotional_content=emotional_content,
    ):
        return RouteDecision(
            "mixed complex life problem",
            "Give me advice",
            "conversation context",
            0.94,
            "The message combines several life areas, emotion, future uncertainty, and an advice request.",
            topic,
        )

    if topic == "education" and _has_any(
        normalized,
        ("pays more", "pay more", "salary", "love psychology", "data science pays"),
    ):
        return RouteDecision(
            "decision support",
            "Help me make a decision",
            "conversation context",
            0.88,
            "The user is comparing education/career fit against pay or outcomes.",
            topic,
        )

    if topic == "workplace" and not explicit_decision_requested(normalized) and _has_any(
        normalized,
        ("discriminat", "harassment", "unfair", "manager", "promotion"),
    ):
        return RouteDecision(
            "practical advice",
            "Give me advice",
            "conversation context",
            0.9,
            "The user describes a workplace concern that benefits from practical guidance.",
            topic,
        )

    if topic == "relationship" and _looks_like_question(normalized) and _has_any(
        normalized,
        ("healthy", "stay", "leave", "break up", "breaking up", "coming back"),
    ):
        return RouteDecision(
            "decision support",
            "Help me make a decision",
            "conversation context",
            0.9,
            "The user asks whether a relationship pattern is healthy or worth continuing.",
            "relationship",
        )

    if _has_any(normalized, ("two job offers", "one pays more", "one feels meaningful")):
        return RouteDecision(
            "decision support",
            "Help me make a decision",
            "conversation context",
            0.88,
            "The user is weighing career options with different values and practical tradeoffs.",
            "career",
        )

    if topic == "business" and _has_any(
        normalized,
        ("quit my job", "leave my job", "start a business", "start my business", "tomorrow"),
    ):
        return RouteDecision(
            "decision support",
            "Help me make a decision",
            "conversation context",
            0.92,
            "The user is describing a business commitment decision with risk.",
            topic,
        )

    if has_bias and not _named_decision_requested(normalized):
        return RouteDecision(
            "cognitive challenge",
            "Challenge my thinking",
            "conversation context",
            0.9,
            "A thinking pattern is central enough to address before ordinary advice.",
            topic if topic != "general" else "self_esteem",
        )

    if _named_decision_requested(normalized):
        return RouteDecision(
            "decision support",
            "Help me make a decision",
            "conversation context",
            0.97,
            "The user named options or asked whether a move is right.",
            topic,
        )

    if explicit_advice_requested(normalized):
        current_information_needed = (
            _has_any(normalized, FACTUAL_MARKERS)
            or _has_any(normalized, HEALTH_WELLNESS_MARKERS)
            or _has_any(normalized, CAREER_EDUCATION_MARKERS)
        )
        return RouteDecision(
            "practical advice",
            "Give me advice",
            "internet" if current_information_needed else "conversation context",
            0.98,
            (
                "The user explicitly requested advice that depends on current facts."
                if current_information_needed
                else "The user explicitly requested practical advice."
            ),
            topic,
        )

    if _has_any(normalized, FACTUAL_MARKERS) and not emotional_statement:
        return RouteDecision(
            "current factual search",
            "Research Assistant",
            "internet",
            0.93,
            "The answer may depend on current external facts.",
            topic,
        )

    if _has_any(normalized, ("do i have adhd", "am i adhd", "could this be adhd")):
        return RouteDecision(
            "health / wellness information",
            "Health Educator",
            "conversation context",
            0.9,
            "The user asks for symptom education about attention difficulties, not a diagnosis.",
            "health",
        )

    if explicit_decision_requested(normalized):
        return RouteDecision(
            "decision support",
            "Help me make a decision",
            "conversation context",
            0.96,
            "The user directly named a choice or asked for a decision.",
            topic,
        )

    if _has_any(normalized, HEALTH_WELLNESS_MARKERS) and _looks_like_question(
        normalized
    ):
        return RouteDecision(
            "health / wellness information",
            "Health Educator",
            "internet",
            0.91,
            "Health information benefits from current, authoritative sources.",
            "health",
        )

    if _has_any(normalized, CAREER_EDUCATION_MARKERS) and not emotional_statement:
        return RouteDecision(
            "career / education",
            "Coach",
            "internet",
            0.89,
            "Career and education information may depend on current opportunities.",
            topic if topic != "general" else "career",
        )

    if _has_grief_signal(normalized, category, emotion):
        return RouteDecision(
            "grief",
            preferred_mode,
            "conversation context",
            0.94,
            "The message describes bereavement or loss.",
            "grief",
        )

    if (
        category == "anxiety"
        or emotion == "anxiety"
        or "anxiety and fear" in clinical_domains
    ):
        return RouteDecision(
            "anxiety / stress",
            preferred_mode,
            "private clinical reference" if clinical_domains else "conversation context",
            0.9,
            "The message primarily describes anxiety, worry, fear, or stress.",
            topic,
        )

    if category == "overwhelm" or emotion == "overwhelm":
        return RouteDecision(
            "overwhelm",
            preferred_mode,
            "conversation context",
            0.9,
            "The message primarily describes overload or difficulty coping.",
            topic,
        )

    is_emotional = (
        category in {"relationship", "self-esteem", "loneliness"}
        or emotion != "neutral"
        or bool(clinical_domains)
        or _has_any(normalized, EMOTIONAL_UNCERTAINTY_MARKERS)
    )
    if is_emotional:
        return RouteDecision(
            "emotional reflection",
            preferred_mode,
            "private clinical reference" if clinical_domains else "conversation context",
            0.85,
            "The message primarily describes an emotional experience.",
            topic,
        )

    if has_active_context and prior_intent in {
        "grief",
        "anxiety / stress",
        "overwhelm",
        "emotional reflection",
    }:
        return RouteDecision(
            prior_intent,
            preferred_mode,
            "conversation context",
            0.8,
            "The message is a follow-up within an active emotional conversation.",
            topic,
        )

    if _has_any(normalized, CASUAL_MARKERS) or (
        not _looks_like_question(normalized) and len(normalized.split()) <= 8
    ):
        return RouteDecision(
            "casual conversation",
            "Friend",
            "conversation context",
            0.82,
            "The message is short and does not clearly request emotional support or facts.",
            topic,
        )

    if _looks_like_question(normalized):
        return RouteDecision(
            "general knowledge",
            "General Assistant",
            "internal knowledge",
            0.76,
            "The message asks a stable general-knowledge question.",
            topic,
        )

    return RouteDecision(
        "general conversation",
        "Conversation",
        "conversation context",
        0.55,
        "No strong intent signal was present, so the user's preference is used.",
        topic,
    )
