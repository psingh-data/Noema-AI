"""Compose human-readable reflection responses from rule-based analysis."""

from __future__ import annotations

from core.bias_detector import BiasResult


VALIDATIONS = {
    "grief": "I'm sorry you're carrying this loss. It makes sense that this feels painful.",
    "anxiety": "That sounds unsettling. When uncertainty builds up, it can take over a lot of mental space.",
    "overwhelm": "That sounds like a lot to hold at once. You do not need to solve all of it right now.",
    "sadness": "It sounds like things feel heavy right now. That feeling deserves some care.",
    "anger": "Your frustration makes sense. Something important to you may feel blocked or crossed.",
    "loneliness": "Feeling alone with this can be deeply tiring. I'm glad you put it into words here.",
    "confusion": "It makes sense to feel torn when the situation is not giving you a clear answer.",
    "hope": "There is a sense of possibility in what you shared, even if the path is still taking shape.",
    "motivation": "It sounds like part of you is ready to move forward and wants a clear place to begin.",
    "neutral": "I hear you. Let us work with what you have actually said.",
}

RECOMMENDATIONS = {
    "grief": "Give yourself permission to remember the person or loss without requiring yourself to feel better immediately.",
    "anxiety": "Separate what you can influence today from what remains uncertain.",
    "overwhelm": "Pause and choose only the smallest task that needs your attention next.",
    "career": "Write down the decision, the options you control, and the next piece of information you need.",
    "study": "Choose one 15-minute study task with a clear stopping point.",
    "relationship": "Describe what happened, what you felt, and what you need before deciding what to say.",
    "self-esteem": "Describe the specific event without turning it into a judgment about your entire worth.",
    "decision making": "Compare your options using the values that matter most to you, not only the fear each option creates.",
    "motivation": "Lower the starting threshold until the first action feels almost too small to avoid.",
    "loneliness": "Consider one low-pressure way to make contact with someone safe.",
    "general reflection": "Notice which part of this situation is asking most strongly for your attention.",
}

QUESTIONS = {
    "grief": "What do you most wish someone understood about this loss?",
    "anxiety": "Which part is a present problem, and which part is a feared possibility?",
    "overwhelm": "What is the one thing that truly needs attention first?",
    "career": "What would a good-enough next step look like, rather than a perfect decision?",
    "study": "What is making it hardest to begin: uncertainty, energy, or fear of the result?",
    "relationship": "What need or boundary feels most important beneath the immediate conflict?",
    "self-esteem": "Would you judge someone you care about by the same standard?",
    "decision making": "Which option fits the person you want to become?",
    "motivation": "What could you do for five minutes without needing to feel motivated first?",
    "loneliness": "Who feels safest to contact, even briefly?",
    "general reflection": "What feels most important about this to you?",
}

SMALL_STEPS = {
    "grief": "Take three slow breaths and write one memory you want to keep.",
    "anxiety": "Write down one action within your control and do only that.",
    "overwhelm": "Put both feet on the floor and name five things you can see.",
    "career": "Write one question whose answer would make the decision clearer.",
    "study": "Open the material and work for five minutes.",
    "relationship": 'Write one honest sentence beginning with "I feel..." without sending it yet.',
    "self-esteem": "Replace one global self-judgment with a factual description of what happened.",
    "decision making": "List the two most important values involved.",
    "motivation": "Set a five-minute timer and begin the smallest part.",
    "loneliness": "Send one simple check-in message to a trusted person.",
    "general reflection": 'Complete this sentence: "What I need most right now is..."',
}


def generate_response(
    emotion: str,
    category: str,
    intensity: str,
    tone: str,
    biases: list[BiasResult],
) -> str:
    """Build a response with validation before analysis or recommendations."""
    validation = VALIDATIONS.get(emotion, VALIDATIONS["neutral"])
    recommendation = RECOMMENDATIONS.get(category, RECOMMENDATIONS["general reflection"])
    question = QUESTIONS.get(category, QUESTIONS["general reflection"])
    small_step = SMALL_STEPS.get(category, SMALL_STEPS["general reflection"])

    if intensity == "high" or tone == "short and grounding":
        return (
            f"{validation}\n\n"
            "Let us focus on stabilization before analysis.\n\n"
            f"**For this moment:** {recommendation}\n\n"
            f"**One small step:** {small_step}\n\n"
            "If you feel unsafe, cannot manage basic needs, or this continues, "
            "contact a licensed mental health professional or doctor promptly."
        )

    sections = [
        validation,
        (
            f"This may connect with **{category}**, and it sounds **{intensity}** "
            "in intensity. That is only a tentative reading of the words available."
        ),
    ]
    if biases:
        top_bias = biases[0]
        sections.append(
            f"One possibility to gently check: there may be some "
            f"{top_bias.name} here. {top_bias.reframe}"
        )

    sections.extend(
        (
            f"**A helpful direction:** {recommendation}",
            f"**A question to continue with:** {question}",
            f"**One small step:** {small_step}",
            (
                "**When to involve a professional:** Consider speaking with a licensed "
                "mental health professional or doctor if this has lasted around two "
                "weeks, keeps returning, is worsening, or is interfering with sleep, "
                "work or study, relationships, or self-care."
            ),
        )
    )
    return "\n\n".join(sections)
