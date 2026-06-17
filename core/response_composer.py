"""Natural response composition for offline ontology-backed routes."""

from __future__ import annotations

from core.language_ontology import LanguageOntologyMatch


def compose_ontology_response(
    text: str,
    match: LanguageOntologyMatch,
    *,
    turn_count: int = 0,
) -> str | None:
    """Compose a short, human response from an offline language match."""
    if not match.matched or match.internet_needed:
        return None

    category = match.category
    if category == "casual_conversation":
        lowered = text.lower()
        if any(word in lowered for word in ("hi", "hello", "hey", "yo", "sup", "wassup")):
            options = (
                "Hey, glad you're here. How are you arriving today: light, heavy, restless, or just curious?",
                "Yo, I am here. Want to talk through something, ask a random question, or just hang for a minute?",
                "Hey, I am with you. We can keep it light, think through something, or just chat.",
            )
            return options[turn_count % len(options)]
        if "bored" in lowered:
            return (
                "Bored in the restless way, the tired way, or the just-give-me-something-interesting way?\n\n"
                "We can keep it light, do a random question, or turn the boredom into a tiny useful plan."
            )

    if category == "venting":
        return (
            "Go on, you can yap. I will not turn this into advice unless you ask for it.\n\n"
            "Say it messy if you need to; what is the part that is annoying you most?"
        )

    if category == "loneliness":
        return (
            "That sounds lonely in the deeper sense, not just being physically alone, "
            "but feeling like people are not really seeing you.\n\n"
            "I will not rush to fix that with a generic social tip. For right now, I "
            "am hearing that you want to feel understood and chosen, not like an "
            "afterthought.\n\n"
            "Is this more about one specific person not getting you, or a broader "
            "feeling that nobody does?"
        )

    if category == "identity_reflection":
        return (
            "Losing the plot of yourself can feel unsettling, like an old identity "
            "or old version of you does not fit anymore but the new one has not "
            "become clear yet.\n\n"
            "That is not something to solve with a quick label. I would look at "
            "three threads: what feels fake, what still feels genuinely yours, and "
            "what you keep doing mainly for approval. The big Who am I question "
            "usually becomes easier when it is broken into values, roles, and "
            "choices.\n\n"
            "Which part feels most off right now: your goals, your values, or how you "
            "act around people?"
        )

    if category == "relationship_feelings":
        return (
            "That sounds draining. When a relationship keeps taking emotional energy, "
            "it can become hard to tell whether the main problem is love, conflict, "
            "fear, guilt, or just exhaustion.\n\n"
            "I would not force a decision from one tired sentence. Start by noticing "
            "the pattern: do you feel safe to speak honestly, do conflicts repair, "
            "and do both people make effort?\n\n"
            "What is draining you most: the fights, the uncertainty, or feeling like "
            "you are carrying it alone?"
        )

    if category == "grief_disclosure":
        recent = any(word in text.lower() for word in ("yesterday", "today", "last night"))
        if recent:
            return (
                "I am sorry. If this happened so recently, your mind and body may "
                "still be in shock, even if part of you is trying to act normal.\n\n"
                "Grief does not follow a fixed timetable. For today, the goal is "
                "simple: do not force yourself to make grief neat. Eat something small if you "
                "can, drink water, and stay near someone safe or reachable.\n\n"
                "Do you have someone with you or someone you can message right now?"
            )
        return (
            "It makes sense that this still hurts. Grief can keep showing up as "
            "missing, remembering, guilt, numbness, or sudden waves of sadness.\n\n"
            "Grief does not follow a fixed timetable. I would not treat crying as "
            "proof that you are failing to move on. It may simply mean the bond "
            "still matters.\n\n"
            "What do you miss most about them right now?"
        )

    if category == "pure_emotional_validation":
        emotion = match.canonical_emotion.replace("_", " ") or "this"
        if "cooked" in text.lower() or emotion == "overwhelm":
            return (
                "That sounds like overload, the kind where your system is tired "
                "before you even start explaining what happened.\n\n"
                "I will not make this clinical. If you are cooked right now, the "
                "first move is lowering the pressure: one drink of water, one slower "
                "exhale, and one sentence naming what is too much.\n\n"
                "What pushed you into that state today?"
            )
        return (
            "What you described sounds exhausting. Feeling empty, tired, and "
            "worthless at the same time can leave you stuck between wanting help "
            "and not having the energy to reach for it.\n\n"
            "Before trying to fix it, I want to understand the shape of it: is this "
            "more sadness, numbness, shame, or plain exhaustion?"
        )

    return None
