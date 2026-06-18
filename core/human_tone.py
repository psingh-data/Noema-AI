"""Light human-tone polish for Noema responses.

This layer adds emotional texture without changing routing, diagnosis safety,
or factual claims. It should make Noema sound more like a thoughtful guide and
less like a template.
"""

from __future__ import annotations

from core.router import RouteDecision


SKIP_INTENTS = {
    "crisis / safety",
    "current factual search",
    "research paper question",
    "general knowledge",
}


def _insert_after_first_paragraph(response: str, bridge: str) -> str:
    parts = response.split("\n\n", 1)
    if len(parts) == 1:
        return f"{response}\n\n{bridge}"
    return f"{parts[0]}\n\n{bridge}\n\n{parts[1]}"


def _bridge_options(route: RouteDecision, user_text: str) -> tuple[str, ...]:
    normalized = " ".join(user_text.lower().split())

    if route.intent == "mixed complex life problem":
        return (
            "No wonder your mind feels noisy here. Each choice seems to touch a different version of your future.",
            "This has emotional weight because it is not just logistics; it is identity, security, love, and timing all pulling at once.",
            "I would not flatten this into a simple checklist. The pressure makes sense because several important parts of life are moving at the same time.",
        )

    if route.intent == "health / wellness information" and (
        "adhd" in normalized or "focus" in normalized or "sleep" in normalized
    ):
        return (
            "This can feel personal, like your mind is failing you, but the timeline matters more than self-blame.",
            "The hard part is that focus problems can feel like a character flaw when they may be a signal from stress, sleep, grief, or overload.",
            "I would hold this gently: attention trouble is real, but it needs context before it becomes a label.",
        )

    if route.topic == "relationship" or route.intent == "relationship":
        return (
            "There is tenderness in this too: you are trying to be honest without being careless with someone else's heart.",
            "This is emotionally messy because care, guilt, relief, and fear can all be true at once.",
            "I would treat these feelings as information, not as a verdict you have to obey instantly.",
        )

    if route.intent == "existential_question":
        return (
            "There is a real ache under that question, because it asks what is worth carrying when nothing is permanent.",
            "That question is not cold philosophy; it touches fear, freedom, death, and the wish for life to feel worth the effort.",
            "I would answer this slowly, because meaning questions usually need honesty more than slogans.",
        )

    if route.topic == "grief" or route.intent == "grief":
        return (
            "There is love inside that pain, which is why it can feel raw instead of neatly processed.",
            "Grief can make ordinary moments feel strange, because the world keeps moving while your inner world is still catching up.",
            "I want to keep this gentle: the goal is not to erase the bond, but to help the pain feel less lonely.",
        )

    if route.intent in {"identity_exploration", "achievement_self_worth"}:
        return (
            "That can feel lonely, like the old map of yourself stopped working before a new one appeared.",
            "Identity questions often hurt because they are not just about plans; they are about who you can trust yourself to become.",
            "I would not rush to label this. Something in you may be asking for a truer shape.",
        )

    if route.intent in {"decision support", "practical advice", "structured_problem_solving"}:
        return (
            "I want to keep this practical without flattening the feeling underneath it.",
            "The useful answer should give you movement, but still respect why this feels heavy.",
            "This needs both steadiness and action: enough care to not rush, enough structure to not stay stuck.",
        )

    if route.intent in {"emotional reflection", "anxiety / stress", "overwhelm"}:
        return (
            "I hear the emotional weight in this; it is not just a problem to solve, it is something you are living through.",
            "Before turning this into advice, it helps to name the human part: this sounds tiring to carry.",
            "There is a lot happening under the surface here, so I want the response to feel steady, not mechanical.",
        )

    return ()


def humanize_response(
    response: str,
    *,
    user_text: str,
    route: RouteDecision,
    turn_count: int,
) -> str:
    """Add a short emotionally aware bridge to eligible responses."""
    if not response or route.intent in SKIP_INTENTS:
        return response
    if "If you feel unsafe" in response or "crisis" in response.lower():
        return response

    options = _bridge_options(route, user_text)
    if not options:
        return response

    bridge = options[turn_count % len(options)]
    lowered = response.lower()
    if bridge.lower() in lowered:
        return response
    return _insert_after_first_paragraph(response, bridge)
