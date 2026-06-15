"""Small, transparent fallback for widely cited foundational literature."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoundationalReference:
    title: str
    authors: str
    year: int
    finding: str
    why_it_matters: str
    url: str


COGNITIVE_SCIENCE = (
    FoundationalReference(
        title="Judgment under Uncertainty: Heuristics and Biases",
        authors="Amos Tversky and Daniel Kahneman",
        year=1974,
        finding=(
            "Shows how judgments under uncertainty often rely on mental shortcuts "
            "such as representativeness, availability, and anchoring."
        ),
        why_it_matters=(
            "It helped establish the modern study of predictable judgment errors."
        ),
        url="https://doi.org/10.1126/science.185.4157.1124",
    ),
    FoundationalReference(
        title="The Extended Mind",
        authors="Andy Clark and David Chalmers",
        year=1998,
        finding=(
            "Argues that tools and external resources can sometimes function as "
            "parts of a person's cognitive system."
        ),
        why_it_matters=(
            "It is a major reference point in debates about where cognition ends."
        ),
        url="https://doi.org/10.1093/analys/58.1.7",
    ),
)

COGNITIVE_REFRAMING = (
    FoundationalReference(
        title="The Empirical Status of Cognitive-Behavioral Therapy",
        authors="Andrew C. Butler, Jason E. Chapman, Evan M. Forman, and Aaron T. Beck",
        year=2006,
        finding=(
            "Reviews meta-analyses of cognitive-behavioral therapy across multiple "
            "conditions and reports substantial evidence of benefit."
        ),
        why_it_matters=(
            "Cognitive reframing is one component of CBT, so this supports the wider "
            "approach without proving that every reframe is helpful."
        ),
        url="https://doi.org/10.1016/j.cpr.2005.07.003",
    ),
    FoundationalReference(
        title="The Efficacy of Cognitive Behavioral Therapy: A Review of Meta-analyses",
        authors="Stefan G. Hofmann, Anu Asnaani, Imke J. J. Vonk, Alice T. Sawyer, and Angela Fang",
        year=2012,
        finding=(
            "Reviews 106 meta-analyses and finds a strong overall evidence base for "
            "CBT, with effectiveness varying by problem and population."
        ),
        why_it_matters=(
            "It places cognitive techniques within the broader evidence for CBT and "
            "also makes the limits of a one-size-fits-all claim clear."
        ),
        url="https://doi.org/10.1007/s10608-012-9476-1",
    ),
)

PROCRASTINATION = (
    FoundationalReference(
        title="The Nature of Procrastination: A Meta-Analytic and Theoretical Review",
        authors="Piers Steel",
        year=2007,
        finding=(
            "Synthesizes research linking procrastination to task aversiveness, "
            "delay, impulsiveness, self-regulation, and expectancy."
        ),
        why_it_matters=(
            "It supports treating procrastination as a self-regulation problem, not "
            "simply a character flaw."
        ),
        url="https://doi.org/10.1037/0033-2909.133.1.65",
    ),
)


def foundational_references(text: str) -> tuple[FoundationalReference, ...]:
    normalized = text.lower()
    if "refram" in normalized or "cognitive behavioral" in normalized or "cbt" in normalized:
        return COGNITIVE_REFRAMING
    if "procrastinat" in normalized:
        return PROCRASTINATION
    if "cognitive science" in normalized or "cognition" in normalized:
        return COGNITIVE_SCIENCE
    return ()


def foundational_response(text: str) -> str | None:
    references = foundational_references(text)
    if not references:
        return None

    sections = [
        (
            "I have not performed a live literature search here, but these are "
            "widely cited foundational works. They are a starting point, not a "
            "claim about the newest or complete evidence."
        )
    ]
    for reference in references:
        sections.append(
            f"**{reference.title}**  \n"
            f"{reference.authors} ({reference.year})  \n"
            f"**Finding:** {reference.finding}  \n"
            f"**Why it matters:** {reference.why_it_matters}  \n"
            f"[Source]({reference.url})"
        )
    return "\n\n".join(sections)
