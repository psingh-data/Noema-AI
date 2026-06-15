"""Context-aware response strategies using topic + user goal."""

from __future__ import annotations

import re

from core.bias_detector import BiasResult
from core.pipeline import ReflectionResult
from core.router import RouteDecision, detect_problem_topics


def _question_count(text: str) -> int:
    return text.count("?")


def _clean_option(value: str) -> str:
    value = re.sub(
        r"\b(what should i do|should i do it|can you help|please be practical|i don't know|i do not know)\b.*$",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    )
    return value.strip(" ?.!") or "this option"


def _decision_options(text: str) -> tuple[str, str] | None:
    normalized = " ".join(text.strip().split())
    patterns = (
        r"(?:should i|whether i should)\s+(.+?)\s+or\s+(.+?)[?.]?$",
        r"confused between\s+(.+?)\s+and\s+(.+?)[?.]?$",
        r"choose between\s+(.+?)\s+and\s+(.+?)[?.]?$",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return _clean_option(match.group(1)), _clean_option(match.group(2))
    return None


def _contains(text: str, *markers: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _grief_decision(text: str) -> str:
    if _contains(text, "therapist", "therapy", "counselor", "counsellor"):
        return (
            "Missing your grandfather every day can be a very real part of grief, "
            "not a sign that you are doing something wrong.\n\n"
            "Seeing a therapist could be a good idea if the grief feels unbearable, "
            "keeps disrupting sleep, work, study, relationships, appetite, or daily "
            "functioning, or if you feel stuck with guilt, regret, or numbness. You "
            "do not need to be in crisis to deserve support.\n\n"
            "My recommendation: if this has been weighing on you daily, try one "
            "initial session and treat it as support, not as a label. You can decide "
            "after that whether the therapist feels useful.\n\n"
            "Is the grief affecting your sleep, work/study, or daily routine?"
        )
    return (
        "This decision is happening inside grief, so the first step is not to treat "
        "your feelings as an obstacle. Name what the loss is changing, then choose "
        "the option that protects your functioning and support system while you heal.\n\n"
        "What part of daily life is grief affecting most?"
    )


def _education_decision(text: str) -> str:
    options = _decision_options(text)
    lowered = text.lower()
    if "university" in lowered and "business" in lowered:
        return (
            "This sounds like a genuine fork in the road: university offers a more "
            "structured path, while the business offers earlier independence and "
            "real-world learning. Compare them by structure, risk, learning, income "
            "readiness, and long-term fit.\n\n"
            "**University may offer:** structure, credentials, networking, and lower "
            "short-term career uncertainty in some fields.\n\n"
            "**Business may offer:** independence, faster learning through experience, "
            "higher upside, and higher uncertainty.\n\n"
            "My recommendation: keep university as the safer main track unless the "
            "business already has customers or income; test the business in a small, "
            "measurable way before making it your only path.\n\n"
            "What stage is the business at today?"
        )
    if "cognitive science" in lowered and "data science" in lowered:
        return (
            "You are choosing between Cognitive Science and Data Science, and they "
            "lead to different strengths.\n\n"
            "**Cognitive Science** fits better if you are curious about mind, brain, "
            "behavior, language, psychology, human-computer interaction, or research. "
            "It can be powerful, but the path is often less linear and may need "
            "graduate study, research experience, or a clear applied niche.\n\n"
            "**Data Science** is usually more directly job-oriented: statistics, "
            "machine learning, coding, analytics, and business problems. It can offer "
            "clearer early career options, but it may feel narrower if your main "
            "interest is people, cognition, and meaning.\n\n"
            "My conditional recommendation: choose Data Science if employability and "
            "technical career clarity matter most right now; choose Cognitive Science "
            "if you are willing to build a niche around research, UX, AI cognition, "
            "or psychology-informed technology.\n\n"
            "Which matters more for your next 2 years: job clarity or intellectual fit?"
        )
    if "psychology" in lowered and "data science" in lowered:
        return (
            "This is a real tradeoff: psychology sounds closer to your interest, "
            "while Data Science may feel safer financially.\n\n"
            "**Psychology** can fit if you care about people, behavior, mental health, "
            "research, or human-centered work, but the career path may require more "
            "specialization, credentials, or graduate study.\n\n"
            "**Data Science** is usually more directly tied to technical jobs and "
            "higher early earning potential, but it may feel empty if you choose it "
            "only for pay and dislike the daily work.\n\n"
            "My recommendation: do not treat this as passion versus money only. Look "
            "for an overlap: psychology plus data, behavioral science, UX research, "
            "mental health analytics, cognitive science, or AI/product research.\n\n"
            "Would you rather optimize first for income security or for subject fit?"
        )
    if options:
        left, right = options
        return (
            f"You are choosing between {left} and {right}. Compare them through "
            "four lenses: career outcomes, skills you will build, how much structure "
            "you need, and which path you can stay curious about when it becomes hard.\n\n"
            "My recommendation: pick the option that gives you both a realistic next "
            "career step and enough interest to keep improving.\n\n"
            "Which option has the clearer next opportunity?"
        )
    return ""


def _workplace_decision(text: str) -> str:
    return (
        "That sounds frustrating, especially if you have been stuck for years and "
        "promotion decisions feel unfair.\n\n"
        "I do not recommend quitting impulsively tomorrow unless your safety or "
        "health is at immediate risk. A stronger plan is: document promotion history, "
        "feedback, responsibilities, dates, and any specific unfair comments or "
        "patterns; ask for written promotion criteria; update your resume; and start "
        "testing the job market while you are still employed.\n\n"
        "My recommendation: prepare an exit path before resigning. If discrimination "
        "may be involved, be careful with wording, keep records, and consider HR, a "
        "trusted senior person, or qualified legal/workplace advice before making a "
        "high-risk move.\n\n"
        "Do you have written promotion criteria or only verbal feedback?"
    )


def _workplace_concern(text: str) -> str:
    return (
        "That is a serious concern, and it is worth handling carefully rather than "
        "jumping straight to a conclusion.\n\n"
        "Start by separating what you know from what you suspect. Write down dates, "
        "specific incidents, decisions, witnesses, promotion criteria, feedback, "
        "and any patterns you have noticed. Keep the wording factual: what happened, "
        "who was present, and what the impact was.\n\n"
        "My recommendation: document first, compare the pattern against written "
        "policies or criteria, then decide whether to speak with HR, a trusted senior "
        "person, or a qualified workplace/legal adviser. At the same time, quietly "
        "keep your resume and options updated so you are not trapped if the situation "
        "does not improve.\n\n"
        "Do you have specific incidents written down yet?"
    )


def _business_decision(text: str) -> str:
    return (
        "The business impulse may be exciting, but quitting tomorrow would be a "
        "high-risk move unless you already have strong validation and enough runway.\n\n"
        "Check five things before committing: cash runway, current income, customer "
        "validation, your minimum living costs, and how reversible the decision is. "
        "An idea is not the same as a tested business; paid demand matters more than "
        "motivation alone.\n\n"
        "My recommendation: do not quit impulsively. Keep the job for now, test the "
        "business with a small offer, get real customer feedback or sales, and set a "
        "clear threshold for leaving, such as savings runway plus repeated revenue.\n\n"
        "Do you already have paying customers or is it still an idea?"
    )


def _relationship_decision(text: str) -> str:
    return (
        "That sounds emotionally conflicted: you still care about her, but the "
        "fighting is exhausting you. Both parts matter.\n\n"
        "Before deciding to stay or leave, look at the pattern: can both of you "
        "communicate without escalating, take responsibility, repair after conflict, "
        "respect boundaries, and change repeated behaviors? Love matters, but it "
        "cannot carry the whole relationship if the pattern keeps harming you.\n\n"
        "My recommendation: do not decide from one exhausted moment. Have one calm, "
        "specific conversation about the pattern, set one or two boundaries, and "
        "watch whether both of you make real repair attempts. If the same fights keep "
        "repeating with no accountability or respect, leaving becomes a more serious "
        "option to consider.\n\n"
        "Have you both tried to repair the pattern, or are you the only one trying?"
    )


def _career_decision(text: str) -> str:
    options = _decision_options(text)
    if options:
        left, right = options
        return (
            f"You are weighing {left} against {right}. Compare stability, growth, "
            "salary, learning, energy cost, reversibility, and long-term fit.\n\n"
            "My recommendation: do low-risk exploration first. Talk to people in the "
            "new path, update your resume, test applications, and run a small trial "
            "before making a hard switch.\n\n"
            "Which option gives you better growth without creating unnecessary risk?"
        )
    if _contains(text, "leave my company", "quit", "change company"):
        return _workplace_decision(text)
    return ""


def _self_esteem_challenge(text: str, biases: list[BiasResult]) -> str:
    bias_name = biases[0].name if biases else "overgeneralization"
    if _contains(text, "failed one interview", "one interview", "failed two interviews", "not smart enough"):
        return (
            "Failing interviews does not make you a failure and does not "
            "automatically mean you are not smart enough. "
            "This may be overgeneralization: a few outcomes are being treated as proof "
            "about your entire ability.\n\n"
            "A more balanced thought is: these interviews did not go well, but "
            "interviews measure preparation, communication, timing, technical skill, "
            "and fit, not intelligence alone. Your next step is to identify what "
            "specifically did not go well and improve that area.\n\n"
            "Which part of the interview felt weakest?"
        )
    if _contains(text, "everyone my age is ahead", "everyone is ahead", "behind compared"):
        return (
            "That comparison can feel painful, especially when everyone else's life "
            "looks more organized from the outside.\n\n"
            "This may be social comparison: measuring your whole life against other "
            "people's visible progress while missing the private uncertainty in their "
            "lives. A more balanced view is that being behind on one timeline does "
            "not mean your life is failing.\n\n"
            "Your next step is to define your own next milestone instead of using "
            "everyone else's pace as the measuring stick.\n\n"
            "What milestone would actually matter for you this month?"
        )
    if _contains(text, "life is already ruined", "my life is ruined", "my life is over"):
        return (
            "That sounds terrifying to believe, even for a moment.\n\n"
            "This may be catastrophizing: your mind is jumping from a painful present "
            "to a final verdict about your whole life. A more balanced view is that "
            "your situation may be serious, but it is not the same as your entire "
            "future being decided.\n\n"
            "Your next step is to name the one area that is most urgent and choose "
            "one reversible action there.\n\n"
            "Which area feels most urgent right now?"
        )
    return (
        "That thought sounds painful, especially if it has started to feel like a "
        "fact rather than a fear.\n\n"
        f"One possibility to check is {bias_name}: a painful moment is being turned into a "
        "larger conclusion about who you are.\n\n"
        "A steadier reframe is: this result matters, but it is not your whole "
        "identity or future. Your practical next step is to separate the event, the "
        "lesson, and the next action.\n\n"
        "What is the specific event this thought is based on?"
    )


def _anxiety_advice() -> str:
    return (
        "That anxiety makes sense, especially if the situation feels uncertain.\n\n"
        "For now, list what is controllable, what needs information, and what must "
        "wait. Then choose one action you can complete today in under 30 minutes.\n\n"
        "My recommendation: reduce the problem to one next action instead of trying "
        "to solve the whole future at once.\n\n"
        "What is the one part you can act on today?"
    )


def _mixed_complex(text: str) -> str:
    topics = detect_problem_topics(text)
    visible_topics = [topic for topic in topics if topic != "general"]
    topic_text = ", ".join(visible_topics[:4]) or "several life areas"
    return (
        f"You are carrying {topic_text} at the same time, so it makes sense that "
        "one simple answer would feel too small.\n\n"
        "**Separate it first:** separate the issues into buckets.\n"
        "1. Emotional load: grief, pressure, fear, or comparison.\n"
        "2. Main track: the path that protects your future most right now.\n"
        "3. Side experiment: the idea you can test without risking everything.\n\n"
        "**Priority:** do not force one huge life decision today. Keep the most "
        "important future track moving, then test the uncertain idea in a small way.\n\n"
        "**My recommendation:** do one main-track action today and one 30-minute "
        "experiment. For example, continue admissions/career preparation as the main "
        "track, and validate the business idea as a side test.\n\n"
        "What is the most time-sensitive track right now?"
    )


def strategy_response(
    *,
    text: str,
    route: RouteDecision,
    analysis: ReflectionResult,
) -> str | None:
    topic = route.topic
    intent = route.intent

    if intent == "mixed complex life problem":
        return _mixed_complex(text)
    if intent == "decision support" and topic == "grief":
        return _grief_decision(text)
    if intent == "decision support" and topic == "education":
        return _education_decision(text) or None
    if intent == "decision support" and topic == "workplace":
        return _workplace_decision(text)
    if intent == "decision support" and topic == "business":
        return _business_decision(text)
    if intent == "decision support" and topic == "relationship":
        return _relationship_decision(text)
    if intent == "decision support" and topic == "career":
        return _career_decision(text) or None
    if intent == "cognitive challenge" and topic in {"self_esteem", "career", "education"}:
        return _self_esteem_challenge(text, analysis.biases)
    if intent == "practical advice" and route.topic in {"anxiety", "general"} and analysis.emotion.emotion == "anxiety":
        return _anxiety_advice()
    if intent == "practical advice" and topic == "workplace":
        return _workplace_concern(text)
    return None


def has_topic_strategy(route: RouteDecision) -> bool:
    return (
        route.intent,
        route.topic,
    ) in {
        ("decision support", "grief"),
        ("decision support", "education"),
        ("decision support", "workplace"),
        ("decision support", "business"),
        ("decision support", "relationship"),
        ("decision support", "career"),
        ("cognitive challenge", "self_esteem"),
        ("cognitive challenge", "career"),
        ("cognitive challenge", "education"),
    } or route.intent == "mixed complex life problem" or (
        route.intent == "practical advice" and route.topic == "workplace"
    )
