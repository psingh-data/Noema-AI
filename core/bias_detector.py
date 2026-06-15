"""Tentative detection of common thinking patterns."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BiasResult:
    name: str
    confidence: float
    explanation: str
    reframe: str


BIAS_RULES = {
    "all-or-nothing thinking": {
        "phrases": (
            "i always fail",
            "i never succeed",
            "completely useless",
            "total failure",
            "i always mess everything up",
            "always mess everything up",
            "i am a failure because",
            "i feel like a failure because",
            "failed one interview",
            "failed once so",
            "not smart enough",
            "i am not smart enough",
            "i'm not smart enough",
        ),
        "explanation": "The situation may be getting reduced to only success or failure.",
        "reframe": "A difficult result can be real without defining your whole ability or future.",
    },
    "catastrophizing": {
        "phrases": (
            "everything is ruined",
            "it will be a disaster",
            "my life is over",
            "my life is already ruined",
            "life is already ruined",
            "worst thing",
            "my whole future is ruined",
            "whole future is ruined",
            "if this fails",
            "i will never",
        ),
        "explanation": "Your mind may be jumping from a difficult possibility to the worst outcome.",
        "reframe": "The outcome is uncertain, and there may be several less severe possibilities.",
    },
    "mind reading": {
        "phrases": (
            "they think i'm",
            "they think i am",
            "everyone thinks",
            "everyone hates me",
            "nobody likes me",
            "nobody cares about me",
            "she must think",
            "he must think",
        ),
        "explanation": "You may be treating an assumption about another person's thoughts as a fact.",
        "reframe": "Their thoughts are not known yet; direct evidence or a conversation may clarify them.",
    },
    "social comparison": {
        "phrases": (
            "everyone my age is ahead",
            "everyone my age is ahead of me",
            "everyone is ahead of me",
            "behind compared to my friends",
            "behind in life",
            "others are ahead of me",
        ),
        "explanation": "You may be measuring your whole life against other people's visible timelines.",
        "reframe": "Other people's progress can be real without proving that your path is failing.",
    },
    "labeling": {
        "phrases": (
            "i am a failure",
            "i'm a failure",
            "i am stupid",
            "i'm stupid",
            "i am worthless",
            "i'm worthless",
        ),
        "explanation": "A painful result may be turning into a label about your whole self.",
        "reframe": "A label is not evidence; the specific situation can be understood and improved.",
    },
    "fortune telling": {
        "phrases": ("i know it will fail", "nothing will work", "i'll definitely fail", "will never get better"),
        "explanation": "A feared prediction may be feeling more certain than the evidence supports.",
        "reframe": "You cannot know the outcome yet, but you can influence the next step.",
    },
    "emotional reasoning": {
        "phrases": ("i feel like a failure so", "because i feel", "it feels true"),
        "explanation": "A strong feeling may be acting as proof that a conclusion is true.",
        "reframe": "The feeling deserves attention, while the conclusion can still be checked against evidence.",
    },
    "sunk cost thinking": {
        "phrases": ("already spent so much", "come this far", "too late to quit", "wasted years"),
        "explanation": "Past investment may be outweighing what would help you from this point forward.",
        "reframe": "Past effort matters, but the best next choice can be judged by future costs and benefits.",
    },
}


def detect_biases(text: str) -> list[BiasResult]:
    normalized = " ".join(text.lower().split())
    results: list[BiasResult] = []

    for name, rule in BIAS_RULES.items():
        matches = sum(phrase in normalized for phrase in rule["phrases"])
        if matches:
            results.append(
                BiasResult(
                    name=name,
                    confidence=min(0.9, 0.55 + matches * 0.15),
                    explanation=rule["explanation"],
                    reframe=rule["reframe"],
                )
            )

    return sorted(results, key=lambda result: result.confidence, reverse=True)[:2]
