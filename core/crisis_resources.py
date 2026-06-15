"""Verified, locally available crisis-resource responses."""

from __future__ import annotations

import json
from pathlib import Path


RESOURCE_PATH = (
    Path(__file__).resolve().parent.parent / "resources" / "crisis_resources.json"
)


def load_resources() -> dict:
    return json.loads(RESOURCE_PATH.read_text(encoding="utf-8"))


def crisis_response(
    *,
    country_code: str = "IN",
    immediate: bool,
) -> str:
    resources = load_resources()
    country = resources["countries"].get(
        country_code.upper(),
        resources["countries"][resources["default_country"]],
    )
    emergency = country["emergency"]
    crisis = country["crisis"][0]

    opening = (
        "I'm really glad you told me. I want to take what you said seriously and "
        "focus on keeping you safe right now."
    )
    immediate_step = (
        f"\n\nIf you might act on these thoughts, cannot keep yourself safe, or "
        f"have already harmed yourself, call **{emergency['number']}** now or go "
        "to the nearest emergency department."
    )
    human_step = (
        "\n\nPlease contact someone you trust and say clearly: \"I am not feeling "
        "safe and need you to stay with me.\" Move away from medicines, weapons, "
        "or anything else you could use to hurt yourself."
    )
    helpline = (
        f"\n\nYou can also contact **{crisis['label']}** at "
        f"**{crisis['contact']}** ({crisis['availability']})."
    )
    question = (
        "\n\nAre you in immediate danger right now, or do you have a plan, access "
        "to the means, or an intention to act soon? Please answer yes, no, or unsure."
    )

    return opening + immediate_step + human_step + helpline + question
