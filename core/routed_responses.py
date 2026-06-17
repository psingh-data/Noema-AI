"""Local fallback responses for non-clinical conversation routes."""

from __future__ import annotations

import re

from core.pipeline import ReflectionResult
from core.foundational_research import foundational_response
from core.response_strategy import strategy_response
from core.router import RouteDecision


DECISION_REQUEST_MARKERS = (
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
    "if you were me",
    "would you",
)

TOPIC_LABELS = {
    "career": ("career", "job", "company", "boss", "work", "business"),
    "education": ("university", "college", "study", "degree", "admission", "exam"),
    "relationships": (
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
    "money": ("money", "salary", "income", "financial", "rent", "debt"),
    "location": ("move abroad", "visa", "country", "city", "relocate"),
    "identity/future": ("purpose", "future", "life", "identity", "not myself"),
    "grief/loss": ("miss my", "missing my", "memory of my", "died", "passed away"),
}


def _explicit_decision_requested(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(marker in normalized for marker in DECISION_REQUEST_MARKERS)


def _decision_options(text: str) -> tuple[str, str] | None:
    normalized = " ".join(text.strip().split())
    match = re.search(
        r"(?:should i|whether i should)\s+(.+?)\s+or\s+(.+?)[?.]?$",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"confused between\s+(.+?)\s+and\s+(.+?)[?.]?$",
            normalized,
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    left = _clean_option(match.group(1))
    right = _clean_option(match.group(2))
    return left, right


def _clean_option(value: str) -> str:
    value = re.sub(
        r"\b(what should i do|can you help|please be practical|i don't know|i do not know)\b.*$",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    )
    return value.strip(" ?.!") or "this option"


def _detected_topics(text: str) -> list[str]:
    normalized = text.lower()
    return [
        label
        for label, markers in TOPIC_LABELS.items()
        if any(marker in normalized for marker in markers)
    ]


def _mixed_complex_response(text: str) -> str:
    topics = _detected_topics(text)
    topic_text = ", ".join(topics[:5]) if topics else "several life areas"
    return (
        f"You are not asking about one small problem; this sounds like {topic_text} "
        "are getting tangled together while you are trying to make a future-facing "
        "choice.\n\n"
        "**Separate it first:**\n"
        "1. Emotional load: what is making this feel urgent or heavy.\n"
        "2. Practical choices: the decisions that actually need action.\n"
        "3. Unknowns: facts you still need before committing.\n\n"
        "**Priority:** do not try to solve every area at once. Stabilize the next "
        "week, choose the one decision with the nearest consequence, and delay "
        "anything that does not need an answer yet.\n\n"
        "**My recommendation:** make a two-column plan today: one small action that "
        "protects your wellbeing, and one action that gathers information for the "
        "main decision. That gives you movement without forcing a rushed life choice.\n\n"
        "Which one decision has the nearest real deadline?"
    )


def _decision_response(text: str) -> str:
    normalized = text.lower()
    if "university" in normalized and "business" in normalized:
        return (
            "This sounds like a genuine fork in the road: university offers a more "
            "structured path, while the business offers earlier independence and "
            "real-world learning.\n\n"
            "The choice becomes clearer when you compare four things: whether the "
            "business already has customers or income, what you want university for "
            "(learning, credentials, network, or security), how much financial runway "
            "you have, and which path you would regret not testing.\n\n"
            "**University may offer:** structure, credentials, networking, and lower "
            "short-term career uncertainty in some fields.\n\n"
            "**Business may offer:** independence, faster learning through experience, "
            "higher upside, and higher uncertainty.\n\n"
            "What stage is the business at today: an idea, early customers, or "
            "consistent income?"
        )

    options = _decision_options(text)
    if options:
        left, right = options
        return (
            f"You are weighing **{left}** against **{right}**. Rather than pretending "
            "there is a universally correct answer, compare them on the same criteria: "
            "values, likely benefits, costs, reversibility, and worst realistic risk.\n\n"
            "Which difference between the two options matters most to you right now?"
        )

    return (
        "Here is a simple decision frame: compare the likely benefit, cost, risk, "
        "reversibility, and fit with your long-term values. If the topic is still "
        "unclear, start with the lowest-risk step that gives you more information "
        "before you commit.\n\n"
        "What is the main constraint: time, money, energy, or risk?"
    )


def _advice_response(text: str, analysis: ReflectionResult) -> str:
    normalized = text.lower()
    emotion = analysis.emotion.emotion
    if emotion == "anxiety":
        validation = "It sounds like uncertainty is making it hard to choose a next move."
    elif emotion == "overwhelm":
        validation = "You sound overloaded, so the next step needs to be small and manageable."
    elif emotion in {"sadness", "loneliness", "grief"}:
        validation = "That sounds genuinely difficult, and you do not need a perfect answer today."
    else:
        validation = "It makes sense to want something practical rather than another round of questions."

    if "procrastinat" in normalized:
        return (
            f"{validation}\n\n"
            "**Try this today:**\n"
            "1. Define a five-minute starting action, not the whole task.\n"
            "2. Remove one source of friction before you begin.\n"
            "3. Work for ten minutes, then decide whether to continue.\n\n"
            "**Why this helps:** Procrastination is often an attempt to avoid "
            "discomfort, uncertainty, or fear of doing badly. A tiny start lowers "
            "the emotional cost and creates momentum.\n\n"
            "Which task are you avoiding most right now?"
        )

    return (
        f"{validation}\n\n"
        "**A practical starting plan:**\n"
        "1. Write the immediate problem in one sentence.\n"
        "2. List two realistic options, even if neither is perfect.\n"
        "3. Choose the smallest reversible action you can take today.\n"
        "4. If you are too overwhelmed to act alone, contact one trusted person and "
        "tell them exactly what kind of help you need.\n\n"
        "**Why this helps:** A reversible next step reduces pressure. You gain useful "
        "information without pretending you must solve the entire situation at once.\n\n"
        "What problem feels most urgent today?"
    )


def _continuity_response(text: str, route: RouteDecision, conversation_state: object | None) -> str | None:
    if conversation_state is None:
        return None
    active_themes = getattr(conversation_state, "active_themes", {})
    active_relationships = getattr(conversation_state, "active_relationships", set())
    progression = getattr(conversation_state, "narrative_progression", [])
    normalized = " ".join(text.lower().split())
    grief_active = "grief" in active_themes
    relationship_text = (
        sorted(active_relationships)[0]
        if active_relationships
        else "the person you lost"
    )

    if grief_active and route.intent == "practical advice" and (
        "what to do" in normalized or "don't know" in normalized or "do not know" in normalized
    ):
        return (
            f"Given what you shared about your {relationship_text}, this does not "
            "sound like a random lack of direction. It sounds like grief has moved "
            "into confusion, and your mind is asking for something steady to hold.\n\n"
            "**For the next few hours, keep it small:** drink water, sit somewhere "
            "quiet, message one person you trust, and do one grounding action such "
            "as naming five things you can see. Do not try to solve your whole life "
            "while grief is this close to the surface.\n\n"
            "My recommendation: choose support before strategy today. Once your body "
            "settles even a little, we can separate emotional pain from practical "
            "decisions.\n\n"
            "Are you alone right now, or is there someone nearby you can contact?"
        )

    if grief_active and route.intent in {"emotional reflection", "overwhelm", "anxiety / stress"} and (
        "suffocat" in normalized or "can't breathe" in normalized or "overwhelm" in progression
    ):
        return (
            f"This feels connected to what you said about your {relationship_text}. "
            "The thread seems to be grief, then not knowing what to do, and now a "
            "suffocating kind of overwhelm.\n\n"
            "For this moment, the goal is not to analyze everything. Try to lower "
            "the pressure in your body first: sit upright, loosen your shoulders, "
            "take a slower breath out than in, and name one thing in the room that "
            "feels solid.\n\n"
            "The next useful question is more specific now: is this suffocation "
            "mostly in your body, your thoughts, or the feeling of being trapped "
            "by the loss?"
        )
    return None


def _challenge_response(analysis: ReflectionResult) -> str:
    if analysis.biases:
        bias = analysis.biases[0]
        return (
            "That thought sounds painful, especially if it has started to feel like a "
            "fact rather than a fear.\n\n"
            f"One possibility to check is **{bias.name}**. {bias.explanation} "
            f"{bias.reframe}\n\n"
            "What direct evidence supports the thought, and what evidence might it be "
            "leaving out?"
        )
    return (
        "I can challenge the thought without dismissing the feeling underneath it. "
        "What is the exact belief you want us to examine?"
    )


def _casual_response(text: str, turn_count: int) -> str:
    normalized = text.lower()
    if "joke" in normalized:
        options = (
            "Here is a terrible one: I tried to organize my thoughts, but they formed a union.",
            "Terrible joke incoming: my calendar broke up with me because I kept taking it for granted.",
            "Fine, but you asked for terrible: I told my laptop I needed space, and now it will not stop opening tabs.",
        )
        return options[turn_count % len(options)]
    if "pokemon" in normalized or "pokémon" in normalized:
        return (
            "Chansey has the bedside manner, Slowking has the reflective questions, "
            "and Psyduck has lived experience with stress. Best therapist? Probably "
            "Chansey. Most relatable therapist? Psyduck, unfortunately."
        )
    if "king" in normalized:
        options = (
            "That is a bold statement. King of what exactly?",
            "Royal confidence today, apparently. What makes you say that?",
            "All right, Your Majesty. Is this confidence, roleplay, or are you testing me?",
        )
        return options[turn_count % len(options)]
    options = (
        "Okay, what happened?",
        "I get what you mean. What came next?",
        "Interesting. What made that come to mind?",
        "All right, I am curious. Go on.",
    )
    return options[turn_count % len(options)]


def routed_local_response(
    *,
    text: str,
    route: RouteDecision,
    analysis: ReflectionResult,
    turn_count: int,
    approved_memory: tuple[str, ...] = (),
    conversation_state: object | None = None,
) -> str | None:
    memory_note = ""
    if approved_memory and route.intent in {
        "decision support",
        "practical advice",
        "career / education",
        "mixed complex life problem",
    }:
        memory_note = (
            "\n\n**Context you asked me to remember:** "
            + " | ".join(approved_memory[-2:])
        )
    strategic = strategy_response(text=text, route=route, analysis=analysis)
    if strategic is not None:
        return strategic + memory_note
    continuity = _continuity_response(text, route, conversation_state)
    if continuity is not None:
        return continuity + memory_note
    if route.intent == "casual conversation":
        return _casual_response(text, turn_count)
    if route.intent == "decision support":
        return _decision_response(text) + memory_note
    if route.intent == "mixed complex life problem":
        return _mixed_complex_response(text) + memory_note
    if route.intent == "practical advice":
        return _advice_response(text, analysis) + memory_note
    if route.intent == "venting":
        return (
            "Go ahead. You do not need to make this neat, balanced, or productive, "
            "and I will not turn it into advice.\n\n"
            "What is the part you most need to get off your chest?"
        )
    if route.intent == "cognitive challenge":
        return _challenge_response(analysis)
    if route.intent == "current factual search":
        if _explicit_decision_requested(text):
            return (
                "Current prices, availability, and product details need live "
                "verification, so I will not invent a recommendation.\n\n"
                "To compare the options, use the same criteria for each: your budget, "
                "the features you will actually use, reliability, support, total "
                "cost, and the risk of buying now versus waiting. Shortlist two or "
                "three choices from verified current sources, then prefer the option "
                "that meets your real needs without paying for features you will not "
                "use."
            )
        return (
            "I'd want to verify that with current sources before answering. Right now "
            "I don't have live web access available, so I can't confirm the latest "
            "details and I don't want to guess.\n\n"
            "I can still explain the general process and point you toward the official "
            "sources that should be checked."
        )
    if route.intent == "research paper question":
        fallback = foundational_response(text)
        if fallback:
            return fallback
        return (
            "I'd want to check the research literature before presenting a conclusion. "
            "Right now I can't retrieve and verify papers, so I shouldn't pretend that "
            "a general answer is a research synthesis.\n\n"
            "I can still help define the question and explain what kinds of studies or "
            "evidence would be most useful to look for."
        )
    if route.intent == "career / education":
        return (
            "I couldn't verify current career or education information from live "
            "sources, so I won't guess about programs, opportunities, requirements, "
            "or market conditions.\n\n"
            "A stable way to think about the choice is to compare the skills you want "
            "to build, credentials required, cost and time, realistic job pathways, "
            "and how well the path fits your motivation and values."
        ) + memory_note
    if route.intent == "health / wellness information":
        return (
            "I would want to check current, authoritative health sources before "
            "giving you factual guidance. I do not have a verified live result right "
            "now, so I will not guess. A qualified clinician can assess symptoms or "
            "treatment decisions properly."
        )
    if route.intent == "general knowledge":
        return (
            "I can help with that, but the enhanced general-answer model is not "
            "available in this session. I do not want to manufacture an answer from "
            "a narrow local rule set."
        )
    if route.intent == "general conversation":
        return (
            "I am following. Give me a bit more context and I will help you work "
            "through it."
        )
    return None
