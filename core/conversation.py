"""Adaptive, psychologically informed multi-turn conversation."""

from __future__ import annotations
from dataclasses import dataclass, field

from core.clinical_signals import ClinicalSignal, detect_clinical_signals
from core.clinical_reasoning import (
    PossibleExplanation,
    explanation_dicts,
    possible_explanations,
    select_response_style,
    update_style_history,
)
from core.language_ontology import (
    LanguageOntologyMatch,
    empty_language_ontology_match,
    match_language_ontology,
)
from core.pipeline import ReflectionResult, process_reflection
from core.routed_responses import routed_local_response
from core.router import RouteDecision, route_message
from core.safety import assess_safety, generate_crisis_response
from core.symptom_profile import (
    ClinicalOverlap,
    build_symptom_profile,
    empty_symptom_profile,
    merge_symptom_profiles,
    overlap_dicts,
    overlap_summary,
    possible_clinical_overlaps,
)


SUPPORT_MODES = (
    "Just listen",
    "Help me understand my feelings",
    "Give me advice",
    "Help me make a decision",
    "Challenge my thinking",
)


@dataclass
class ConversationState:
    turn_count: int = 0
    pending_domain: str | None = None
    covered_domains: set[str] = field(default_factory=set)
    accumulated_domains: dict[str, int] = field(default_factory=dict)
    support_urgency: str = "routine"
    safety_followup: bool = False
    safety_concern_disclosed: bool = False
    duration_known: bool = False
    functioning_known: bool = False
    support_known: bool = False
    goal_known: bool = False
    support_mode: str = "Just listen"
    current_intent: str = "open conversation"
    knowledge_route: str = "conversation context"
    last_emotional_category: str | None = None
    active_themes: dict[str, int] = field(default_factory=dict)
    active_emotions: dict[str, int] = field(default_factory=dict)
    active_relationships: set[str] = field(default_factory=set)
    unresolved_concerns: list[str] = field(default_factory=list)
    current_goals: list[str] = field(default_factory=list)
    conversation_summary: str = ""
    narrative_progression: list[str] = field(default_factory=list)
    symptom_profile: dict[str, int] = field(default_factory=empty_symptom_profile)
    possible_clinical_overlaps: list[dict[str, str | int]] = field(default_factory=list)
    possible_explanations: list[dict[str, str]] = field(default_factory=list)
    last_5_styles: list[str] = field(default_factory=list)
    reasoning_style: str = "reflective"
    interventions_tried: list[str] = field(default_factory=list)
    interventions_failed: list[str] = field(default_factory=list)
    major_losses: list[str] = field(default_factory=list)
    recurring_fears: list[str] = field(default_factory=list)
    recurring_conflicts: list[str] = field(default_factory=list)
    recurring_values: list[str] = field(default_factory=list)
    major_goals: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConversationReply:
    response: str
    analysis: ReflectionResult
    state: ConversationState
    clinical_domains: tuple[str, ...]
    recommendation_type: str
    route: RouteDecision
    symptom_profile: dict[str, int] = field(default_factory=empty_symptom_profile)
    possible_clinical_overlaps: tuple[ClinicalOverlap, ...] = ()
    possible_explanations: tuple[PossibleExplanation, ...] = ()
    reasoning_style: str = "reflective"
    language_ontology_match: LanguageOntologyMatch = field(
        default_factory=empty_language_ontology_match
    )


DOMAIN_QUESTIONS = {
    "depressed mood": (
        "When you say things feel low, have you also noticed losing interest or "
        "pleasure in things that would normally matter to you?"
    ),
    "anxiety and fear": (
        "When the anxiety rises, what happens in your body and what do you find "
        "yourself avoiding or repeatedly checking?"
    ),
    "anger and irritability": (
        "What tends to happen just before the anger rises, and has it led to actions "
        "or arguments that worry you afterward?"
    ),
    "elevated mood or activation": (
        "Have there been periods when you needed much less sleep but still felt "
        "unusually energized, fast-thinking, confident, or more impulsive than usual?"
    ),
    "sleep": (
        "How much are you sleeping compared with your usual pattern, and do you feel "
        "tired afterward or unusually energized despite little sleep?"
    ),
    "physical symptoms": (
        "Have these physical symptoms been medically checked, and do they appear "
        "mainly during stress or also at other times?"
    ),
    "concentration and memory": (
        "Is the difficulty with concentration or memory new, and how much is it "
        "interfering with work, study, conversations, or decisions?"
    ),
    "repetitive thoughts or behaviors": (
        "Do those thoughts feel unwanted and intrusive, and are there actions you "
        "feel driven to repeat to reduce the anxiety?"
    ),
    "detachment or unreality": (
        "When that disconnected or unreal feeling happens, how long does it last, "
        "and are you still able to tell where you are and keep yourself safe?"
    ),
    "unusual perceptions or beliefs": (
        "Are those experiences happening now, and are they telling you to do "
        "anything or making it hard to tell what is real?"
    ),
    "substance use": (
        "Has your alcohol or substance use changed recently, and does it seem to "
        "improve these feelings briefly but worsen them later?"
    ),
    "relationships and sense of self": (
        "What happened in the relationship, and what did it lead you to believe "
        "about yourself?"
    ),
    "daily functioning": (
        "Which part of daily life has become hardest: work or study, relationships, "
        "sleep, eating, hygiene, or getting basic tasks done?"
    ),
}

GENERAL_QUESTIONS = {
    "duration": (
        "How long has this been happening, and has it been continuous or does it "
        "come in episodes?"
    ),
    "functioning": (
        "How much is this affecting your sleep, work or study, relationships, and "
        "ability to take care of yourself?"
    ),
    "safety": (
        "I want to ask this directly because your safety matters: have you had "
        "thoughts of harming yourself, ending your life, or not wanting to be alive?"
    ),
    "supports": (
        "Who knows you are going through this, and who could you contact if today "
        "became significantly harder?"
    ),
    "goal": (
        "What would be most useful from this conversation right now: understanding "
        "the pattern, coping with the next few hours, making a practical plan, or "
        "working out how to seek professional help?"
    ),
}

YES_TERMS = (
    "yes",
    "yeah",
    "i do",
    "i have",
    "sometimes",
    "probably",
    "a little",
    "i might",
    "might act",
    "have a plan",
    "have access",
    "intend to",
    "not safe",
)
NO_TERMS = ("no", "never", "not at all", "i don't", "i do not")
UNSURE_TERMS = ("unsure", "not sure", "don't know", "do not know", "maybe")
NEGATED_RISK_TERMS = (
    "don't have a plan",
    "do not have a plan",
    "no plan",
    "don't have access",
    "do not have access",
    "no access",
    "no intention",
    "won't act",
    "will not act",
)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    normalized = " ".join(text.lower().split())
    return any(term in normalized for term in terms)


def _remember_count(container: dict[str, int], value: str) -> None:
    if not value or value == "general":
        return
    container[value] = container.get(value, 0) + 1


def _remember_once(items: list[str], value: str, limit: int = 6) -> None:
    if value and value not in items:
        items.append(value)
    if len(items) > limit:
        del items[:-limit]


def _detect_relationships(text: str) -> set[str]:
    normalized = " ".join(text.lower().split())
    relationships = {
        relation
        for relation in (
            "grandfather",
            "grandmother",
            "mother",
            "father",
            "sister",
            "brother",
            "girlfriend",
            "boyfriend",
            "partner",
            "friend",
        )
        if relation in normalized
    }
    if "grandpa" in normalized:
        relationships.add("grandfather")
    if "grandma" in normalized:
        relationships.add("grandmother")
    return relationships


def _emotion_label(text: str, analysis: ReflectionResult) -> str | None:
    normalized = " ".join(text.lower().split())
    if "suffocat" in normalized or "can't breathe" in normalized:
        return "overwhelm"
    if "don't know" in normalized or "do not know" in normalized or "confused" in normalized:
        return "confusion"
    if analysis.emotion.emotion not in {"neutral", "distress"}:
        return analysis.emotion.emotion
    return None


def _theme_label(text: str, analysis: ReflectionResult, route: RouteDecision | None) -> str | None:
    normalized = " ".join(text.lower().split())
    if route and route.topic != "general":
        return route.topic
    if analysis.category.category not in {"general reflection", "decision making"}:
        return analysis.category.category
    if "grandfather" in normalized or "passed away" in normalized or "died" in normalized:
        return "grief"
    if "what to do" in normalized or "next" in normalized:
        return "next step"
    return None


def _concern_label(text: str, emotion: str | None, theme: str | None) -> str | None:
    normalized = " ".join(text.lower().split())
    relationships = _detect_relationships(normalized)
    if theme == "grief" and relationships:
        return f"grief connected to {sorted(relationships)[0]}"
    if "what to do" in normalized or "don't know" in normalized or "do not know" in normalized:
        return "uncertainty about what to do next"
    if "suffocat" in normalized or "can't breathe" in normalized:
        return "feeling suffocated or overwhelmed"
    if emotion in {"anxiety", "overwhelm", "sadness", "grief", "confusion"}:
        return f"{emotion} needs support"
    return None


def _goal_label(route: RouteDecision, text: str) -> str | None:
    normalized = " ".join(text.lower().split())
    if route.intent == "practical advice" or "what to do" in normalized:
        return "find a practical next step"
    if route.intent == "decision support":
        return "make a decision"
    if route.intent in {"grief", "emotional reflection", "overwhelm", "anxiety / stress"}:
        return "understand and steady the feeling"
    if route.intent == "venting":
        return "be heard without advice"
    return None


def _summarize_state(state: ConversationState) -> str:
    themes = sorted(state.active_themes, key=state.active_themes.get, reverse=True)[:3]
    emotions = state.narrative_progression[-4:] or list(state.active_emotions)[:3]
    relationships = sorted(state.active_relationships)[:3]
    parts: list[str] = []
    if themes:
        parts.append("themes: " + ", ".join(themes))
    if emotions:
        parts.append("emotional progression: " + " -> ".join(emotions))
    if relationships:
        parts.append("relationships: " + ", ".join(relationships))
    if state.unresolved_concerns:
        parts.append("open concern: " + state.unresolved_concerns[-1])
    return "Conversation state - " + "; ".join(parts) if parts else ""


def _stress_driver_response(state: ConversationState) -> str:
    drivers: list[str] = []
    if state.major_losses:
        drivers.append(
            "the emotional weight of "
            + ", ".join(state.major_losses[:2])
        )
    if state.recurring_fears:
        drivers.append(
            "recurring fear around "
            + ", ".join(state.recurring_fears[:2])
        )
    if state.recurring_conflicts:
        drivers.append(
            "conflicts between "
            + ", ".join(state.recurring_conflicts[:2])
        )
    if state.recurring_values and state.major_goals:
        drivers.append(
            "the pressure to protect values like "
            + ", ".join(state.recurring_values[:2])
            + " while moving toward "
            + ", ".join(state.major_goals[:2])
        )
    if not drivers:
        return (
            "From what you have shared so far, I do not have enough repeated context "
            "to name one main driver confidently. I would look for the pattern that "
            "keeps returning across different topics: fear, loss, conflict, values, "
            "or a goal that feels uncertain."
        )
    return (
        "Looking across what you have shared, I do not think your stress is coming "
        "from one small problem. The strongest pattern seems to be "
        + "; ".join(drivers[:3])
        + ".\n\n"
        "My read: the stress is probably being driven by uncertainty about the "
        "future while you are also carrying emotional weight from the past. That "
        "combination can make every practical decision feel bigger than it really is.\n\n"
        "The next useful move is to separate grief or emotional load from the actual "
        "decision in front of you, then choose one concrete step that protects the "
        "future without demanding total certainty today."
    )


def _asks_for_stress_driver(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return (
        "driving most of my stress" in normalized
        or "what is driving my stress" in normalized
        or "why am i so stressed" in normalized
        or "main reason i am stressed" in normalized
        or "actually driving" in normalized
    )


def conversation_state_snapshot(state: ConversationState) -> dict[str, object]:
    return {
        "active_themes": sorted(
            state.active_themes,
            key=state.active_themes.get,
            reverse=True,
        ),
        "active_emotions": sorted(
            state.active_emotions,
            key=state.active_emotions.get,
            reverse=True,
        ),
        "active_relationships": sorted(state.active_relationships),
        "unresolved_concerns": list(state.unresolved_concerns),
        "current_goals": list(state.current_goals),
        "conversation_summary": state.conversation_summary,
        "narrative_progression": list(state.narrative_progression),
        "symptom_profile": dict(state.symptom_profile),
        "possible_clinical_overlaps": list(state.possible_clinical_overlaps),
        "possible_explanations": list(state.possible_explanations),
        "conversation_stage": (
            state.narrative_progression[-1]
            if state.narrative_progression
            else state.current_intent
        ),
        "reasoning_style": state.reasoning_style,
        "last_5_styles": list(state.last_5_styles),
        "interventions_tried": list(state.interventions_tried),
        "interventions_failed": list(state.interventions_failed),
        "major_losses": list(state.major_losses),
        "recurring_fears": list(state.recurring_fears),
        "recurring_conflicts": list(state.recurring_conflicts),
        "recurring_values": list(state.recurring_values),
        "major_goals": list(state.major_goals),
    }


def _update_symptom_layer(
    state: ConversationState,
    current_profile: dict[str, int],
    current_overlaps: tuple[ClinicalOverlap, ...],
) -> None:
    state.symptom_profile = merge_symptom_profiles(
        state.symptom_profile,
        current_profile,
    )
    state.possible_clinical_overlaps = overlap_dicts(
        possible_clinical_overlaps(state.symptom_profile)
        or current_overlaps
    )


def _update_reasoning_layer(
    state: ConversationState,
    explanations: tuple[PossibleExplanation, ...],
    style: str,
) -> None:
    state.possible_explanations = explanation_dicts(explanations)
    state.reasoning_style = style
    update_style_history(state.last_5_styles, style)


def _update_intervention_tracking(text: str, state: ConversationState) -> None:
    normalized = " ".join(text.lower().split())
    intervention_markers = {
        "breathing": ("breathing", "breathe", "breath"),
        "grounding": ("grounding", "five things", "5 things"),
        "journaling": ("journal", "write it down", "writing"),
        "therapy": ("therapy", "therapist", "counseling", "counselling"),
        "support group": ("support group",),
    }
    failed = any(
        marker in normalized
        for marker in (
            "did not help",
            "didn't help",
            "not helping",
            "had no effect",
            "doesn't work",
            "does not work",
        )
    )
    for label, markers in intervention_markers.items():
        if any(marker in normalized for marker in markers):
            _remember_once(state.interventions_tried, label, limit=8)
            if failed:
                _remember_once(state.interventions_failed, label, limit=8)


def _update_narrative_memory(text: str, state: ConversationState) -> None:
    normalized = " ".join(text.lower().split())
    for loss_marker, label in (
        ("grandfather", "loss of grandfather"),
        ("grandmother", "loss of grandmother"),
        ("cat died", "loss of cat"),
        ("dog died", "loss of dog"),
        ("passed away", "bereavement"),
        ("died", "bereavement"),
    ):
        if loss_marker in normalized:
            _remember_once(state.major_losses, label, limit=6)

    fear_patterns = (
        ("wrong decision", "making the wrong life decision"),
        ("feel behind", "falling behind others"),
        ("behind compared", "falling behind others"),
        ("future", "uncertainty about the future"),
        ("admission", "uncertainty about admissions"),
        ("not smart enough", "not being capable enough"),
        ("life is ruined", "future being ruined"),
        ("can't focus", "attention not cooperating"),
        ("cannot focus", "attention not cooperating"),
    )
    for marker, label in fear_patterns:
        if marker in normalized:
            _remember_once(state.recurring_fears, label, limit=8)

    conflict_patterns = (
        ("family pressure", "family expectations"),
        ("family expects", "family expectations"),
        ("girlfriend", "relationship strain"),
        ("boyfriend", "relationship strain"),
        ("partner", "relationship strain"),
        ("boss", "workplace unfairness"),
        ("manager", "workplace unfairness"),
        ("company", "career/workplace tension"),
        ("business", "business versus safer path"),
        ("university", "education or admissions decision"),
        ("data science", "interest versus employability"),
        ("psychology", "interest versus employability"),
    )
    for marker, label in conflict_patterns:
        if marker in normalized:
            _remember_once(state.recurring_conflicts, label, limit=8)

    value_patterns = (
        ("meaningful", "meaning"),
        ("psychology", "understanding people"),
        ("cognitive science", "mind and cognition"),
        ("business", "independence"),
        ("income", "financial security"),
        ("salary", "financial security"),
        ("pays more", "financial security"),
        ("pay more", "financial security"),
        ("family", "family responsibility"),
        ("truth", "honesty"),
        ("fairness", "fairness"),
        ("freedom", "freedom"),
    )
    for marker, label in value_patterns:
        if marker in normalized:
            _remember_once(state.recurring_values, label, limit=8)

    goal_patterns = (
        ("germany", "move or study in Germany"),
        ("admission", "secure admissions"),
        ("university", "choose an education path"),
        ("data science", "build a data/career path"),
        ("business", "test a business path"),
        ("career", "build a stable career"),
        ("focus", "regain focus"),
        ("adhd", "understand attention difficulties"),
    )
    for marker, label in goal_patterns:
        if marker in normalized:
            _remember_once(state.major_goals, label, limit=8)


def _affirmative_risk(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    for phrase in NEGATED_RISK_TERMS:
        normalized = normalized.replace(phrase, " ")
    return _contains_any(normalized, YES_TERMS)


def _persistent(text: str) -> bool:
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in (
            "week",
            "month",
            "year",
            "for a long time",
            "every day",
            "most days",
            "keeps coming back",
        )
    )


def _functioning_disrupted(text: str) -> bool:
    return bool(detect_clinical_signals(text)) and any(
        marker in text.lower()
        for marker in (
            "affecting",
            "interfering",
            "can't work",
            "cannot work",
            "can't study",
            "cannot study",
            "not eating",
            "not showering",
            "can't get out of bed",
            "cannot get out of bed",
        )
    )


def _urgent_domain(signals: list[ClinicalSignal]) -> bool:
    names = {signal.domain for signal in signals}
    return bool(
        names
        & {
            "unusual perceptions or beliefs",
            "detachment or unreality",
            "elevated mood or activation",
        }
    )


def _update_state(
    text: str,
    analysis: ReflectionResult,
    signals: list[ClinicalSignal],
    state: ConversationState,
    route: RouteDecision | None = None,
) -> None:
    state.turn_count += 1
    if state.pending_domain:
        state.covered_domains.add(state.pending_domain)

    for signal in signals:
        state.accumulated_domains[signal.domain] = max(
            signal.score,
            state.accumulated_domains.get(signal.domain, 0),
        )

    if _persistent(text):
        state.duration_known = True
        if (
            analysis.category.category != "grief"
            and state.last_emotional_category != "grief"
        ):
            state.support_urgency = "professional"
    if _functioning_disrupted(text):
        state.functioning_known = True
        state.support_urgency = "professional"
    if analysis.emotion.intensity == "high" and state.support_urgency == "routine":
        state.support_urgency = "professional"
    if _urgent_domain(signals):
        state.support_urgency = "urgent"

    theme = _theme_label(text, analysis, route)
    emotion = _emotion_label(text, analysis)
    if theme:
        _remember_count(state.active_themes, theme)
    if emotion:
        _remember_count(state.active_emotions, emotion)
        if not state.narrative_progression or state.narrative_progression[-1] != emotion:
            state.narrative_progression.append(emotion)
            if len(state.narrative_progression) > 8:
                del state.narrative_progression[:-8]
    for relationship in _detect_relationships(text):
        state.active_relationships.add(relationship)

    concern = _concern_label(text, emotion, theme)
    if concern:
        _remember_once(state.unresolved_concerns, concern)
    if route:
        goal = _goal_label(route, text)
        if goal:
            _remember_once(state.current_goals, goal, limit=4)
    state.conversation_summary = _summarize_state(state)


def _choose_question(
    signals: list[ClinicalSignal],
    state: ConversationState,
) -> tuple[str, str]:
    if not state.duration_known:
        return "duration", GENERAL_QUESTIONS["duration"]
    if not state.functioning_known:
        return "functioning", GENERAL_QUESTIONS["functioning"]

    for signal in signals:
        if signal.domain not in state.covered_domains:
            return signal.domain, DOMAIN_QUESTIONS[signal.domain]

    accumulated = sorted(
        state.accumulated_domains.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    for domain, _score in accumulated:
        if domain not in state.covered_domains:
            return domain, DOMAIN_QUESTIONS[domain]

    if "safety" not in state.covered_domains:
        return "safety", GENERAL_QUESTIONS["safety"]
    if not state.support_known:
        return "supports", GENERAL_QUESTIONS["supports"]
    if not state.goal_known:
        return "goal", GENERAL_QUESTIONS["goal"]
    return "goal", GENERAL_QUESTIONS["goal"]


def _validation(
    analysis: ReflectionResult,
    state: ConversationState,
    text: str = "",
) -> str:
    emotion = analysis.emotion.emotion
    normalized = " ".join(text.lower().split())
    active_grief = "grief" in state.active_themes
    progression = state.narrative_progression[-4:]
    category = (
        state.last_emotional_category
        if analysis.category.category == "general reflection"
        and state.last_emotional_category
        else analysis.category.category
    )

    if all(marker in normalized for marker in ("empty", "tired", "worthless")):
        return (
            "What you described sounds exhausting. Feeling empty, tired, and "
            "worthless at the same time can leave a person stuck between wanting "
            "help and not having the energy to reach for it."
        )
    if "can't focus" in normalized or "cannot focus" in normalized:
        return (
            "That can be genuinely frustrating: wanting your mind to cooperate, "
            "then watching it slip away no matter how much you try to force it."
        )
    if active_grief and "overwhelm" in progression:
        return (
            "This seems connected to the grief you started with. The thread has "
            "moved from loss into uncertainty, and now into a suffocating kind of "
            "overwhelm."
        )
    if active_grief and "confusion" in progression and category != "grief":
        return (
            "I am keeping the earlier grief context in view here. Not knowing what "
            "to do can be part of how loss unsettles the next step."
        )
    if category == "grief":
        return (
            "It makes sense that this loss can still hurt.\n\n"
            "Grief does not follow a fixed timetable, and crying can be one way that "
            "love, memories, and absence continue to show up in our lives."
        )
    if emotion == "anxiety":
        return (
            "That sounds exhausting to carry. When worry keeps taking up space, it "
            "can be hard to feel settled even when you are trying to keep going."
        )
    if emotion in {"sadness", "loneliness"}:
        return (
            "What you are describing sounds painful and lonely. You do not have to "
            "make it sound smaller or more manageable here."
        )
    if emotion == "anger":
        return (
            "It sounds as though something important has been hurt, crossed, or left "
            "unheard. The anger deserves to be understood before it is judged."
        )
    if emotion == "overwhelm":
        return (
            "You seem to be carrying more than feels manageable right now. We can "
            "slow this down instead of trying to solve everything at once."
        )
    if state.turn_count == 1:
        return (
            "Thank you for putting this into words. I want to understand what this "
            "has been like for you before moving toward explanations or advice."
        )
    followups = (
        "I hear you. This has clearly been weighing on you for a while.",
        "That sounds hard to keep dealing with day after day.",
        "I can see why that part would be difficult.",
        "Okay, I have a clearer picture now.",
    )
    return followups[(state.turn_count - 2) % len(followups)]


def _professional_guidance(state: ConversationState) -> str:
    if state.support_urgency == "urgent":
        return (
            "Because part of what you described can sometimes need prompt assessment, "
            "please consider contacting a licensed mental health professional, doctor, "
            "or urgent mental health service today. If you feel unable to stay safe or "
            "cannot tell what is real, seek emergency help now."
        )
    if state.support_urgency == "professional":
        return (
            "Given the duration, intensity, or effect on daily life, arranging an "
            "appointment with a licensed mental health professional or doctor would "
            "be a sensible next step. This conversation can help you organize what "
            "to tell them, but it should not replace an assessment."
        )
    return (
        "If this begins worsening or disrupting sleep, work or study, relationships, "
        "or basic self-care, a licensed mental health professional can help assess it "
        "more fully."
    )


def _with_clinical_overlap_note(
    response: str,
    route: RouteDecision,
    overlaps: tuple[ClinicalOverlap, ...],
) -> str:
    if not overlaps or route.intent in {
        "crisis / safety",
        "casual conversation",
        "current factual search",
        "research paper question",
        "general knowledge",
    }:
        return response
    if max(overlap.score for overlap in overlaps) < 2:
        return response
    note = overlap_summary(overlaps)
    if not note or note.lower() in response.lower():
        return response
    return f"{response}\n\n{note}"


def _mode_question(
    mode: str,
    clinical_question: str,
    analysis: ReflectionResult,
    state: ConversationState,
) -> str:
    category = (
        state.last_emotional_category
        if analysis.category.category == "general reflection"
        and state.last_emotional_category
        else analysis.category.category
    )
    if mode == "Just listen":
        if category == "grief":
            if "feeling suffocated or overwhelmed" in state.unresolved_concerns:
                return (
                    "When you say suffocated, is it mostly in your body, your "
                    "thoughts, or the feeling of being trapped by the loss?"
                )
            if "grief connected to grandfather" in state.unresolved_concerns and state.turn_count > 1:
                return (
                    "Right now, does the grief feel more like missing him, feeling "
                    "lost about what comes next, or pressure building in your body?"
                )
            return (
                "What do you miss most right now: the person, the time you had "
                "together, a particular memory, or something else?"
            )
        return (
            "When this shows up, does it feel more like heaviness in your body, "
            "harsh thoughts about yourself, or a loss of motivation?"
        )
    if (
        mode == "Help me understand my feelings"
        and category == "grief"
    ):
        return (
            "When the tears come, what do they seem to carry most: missing them, "
            "memories, unfinished feelings, or something else?"
        )
    if mode == "Help me make a decision":
        if analysis.category.category != "decision making":
            return (
                "Is there a specific choice connected to this that you want help "
                "sorting through?"
            )
        return (
            "What matters most to you in this choice, and what are you most afraid "
            "each option might cost?"
        )
    if mode == "Challenge my thinking":
        if category == "grief":
            return (
                "What are you telling yourself it means that you still cry about "
                "this loss?"
            )
        return (
            "What feels like the strongest evidence for this thought, and what might "
            "the thought be leaving out?"
        )
    if mode == "Give me advice" and category == "grief":
        return "When does the grief usually hit you hardest?"
    return clinical_question


def _exploration(
    mode: str,
    question: str,
    signals: list[ClinicalSignal],
) -> str:
    if mode == "Just listen":
        return (
            "I will stay with the experience before trying to fix it. "
            f"{question}"
        )

    domain_names = [signal.domain for signal in signals[:2]]
    if domain_names:
        joined = " and ".join(domain_names)
        context = (
            f"A couple of things may be overlapping here, including {joined}."
        )
    else:
        context = "The details around when this happens can help make sense of it."
    return f"{context} {question}"


def _mode_guidance(
    mode: str,
    analysis: ReflectionResult,
    state: ConversationState,
) -> str:
    category = (
        state.last_emotional_category
        if analysis.category.category == "general reflection"
        and state.last_emotional_category
        else analysis.category.category
    )

    if mode in {"Just listen", "Help me understand my feelings"}:
        return ""
    if mode == "Give me advice":
        if category == "grief":
            return (
                "A gentle place to start is to give the grief a small, intentional "
                "space: write down one memory, speak to someone who knew them, or "
                "create a simple ritual. The goal is not to stop caring or force the "
                "tears away, but to make the grief less lonely."
            )
        if analysis.emotion.emotion == "anxiety":
            return (
                "For now, choose one part that is within your control, write the next "
                "small action, and give it ten focused minutes. Pair that with slower "
                "breathing or a short walk so your body receives a cue that it can "
                "come down from alert."
            )
        return (
            "For now, choose one small action that reduces pressure rather than trying "
            "to fix the whole situation. Keep it specific enough to do today, and ask "
            "one trusted person for the kind of support you actually need."
        )
    if mode == "Help me make a decision":
        if category != "decision making":
            return (
                "Once the choice is clear, we can compare the options by what each "
                "one gives you, what it costs, and which personal value it serves. "
                "Noema should not invent a decision that you have not actually named."
            )
        return (
            "Try separating the decision into three columns: what each option gives "
            "you, what it costs, and which personal value it serves. A good decision "
            "is not always the option with no discomfort; it is often the one whose "
            "costs you can accept for reasons that matter to you."
        )
    if mode == "Challenge my thinking":
        if category == "grief":
            return (
                "The feeling itself does not need to be argued away. The part worth "
                "challenging may be the judgment attached to it, such as believing "
                "that crying means you are weak, stuck, or grieving incorrectly."
            )
        return (
            "A gentle challenge is to treat the thought as one interpretation rather "
            "than a verdict. Look for absolute words such as always, never, everyone, "
            "or failure, then rewrite the thought so it includes both the painful "
            "evidence and any facts it currently excludes."
        )
    return ""


def _compose_human_response(
    *,
    text: str,
    mode: str,
    analysis: ReflectionResult,
    signals: list[ClinicalSignal],
    state: ConversationState,
    question: str,
) -> str:
    if mode == "Give me advice":
        guidance = _mode_guidance(mode, analysis, state)
        sections = [_validation(analysis, state, text)]
        if guidance:
            sections.append(guidance)
        sections.append(
            "This approach is meant to lower the immediate pressure and give you "
            "something concrete to test, rather than waiting for complete certainty."
        )
        sections.append(question)
        if state.support_urgency in {"professional", "urgent"}:
            sections.append(_professional_guidance(state))
        return "\n\n".join(section for section in sections if section)

    sections = [
        _validation(analysis, state, text),
        _exploration(mode, question, signals),
    ]
    guidance = _mode_guidance(mode, analysis, state)
    if guidance:
        sections.append(guidance)
    if state.support_urgency in {"professional", "urgent"}:
        sections.append(_professional_guidance(state))
    return "\n\n".join(section for section in sections if section)


def _safety_reply(
    text: str,
    analysis: ReflectionResult,
    state: ConversationState,
    country_code: str,
    symptom_profile: dict[str, int],
    clinical_overlaps: tuple[ClinicalOverlap, ...],
) -> ConversationReply | None:
    safety_route = RouteDecision(
        intent="crisis / safety",
        response_mode="Safety",
        knowledge_route="local crisis resources",
        confidence=1.0,
        reason="Urgent safety language always overrides other routes.",
    )
    safety = assess_safety(text)
    if safety.is_crisis:
        state.safety_concern_disclosed = True
        state.safety_followup = True
        state.support_urgency = "urgent"
        state.pending_domain = "safety_immediacy"
        return ConversationReply(
            response=generate_crisis_response(safety.level, country_code),
            analysis=analysis,
            state=state,
            clinical_domains=("suicidal thoughts or self-harm",),
            recommendation_type="urgent safety support",
            route=safety_route,
            symptom_profile=symptom_profile,
            possible_clinical_overlaps=clinical_overlaps,
        )

    if state.safety_followup:
        if _contains_any(text, UNSURE_TERMS):
            state.support_urgency = "urgent"
            return ConversationReply(
                response=(
                    "Because you are unsure, please treat this as urgent. Contact a "
                    "trusted person and a crisis service now, and move away from anything "
                    "you could use to harm yourself. Are you able to make that contact now?"
                ),
                analysis=analysis,
                state=state,
                clinical_domains=("uncertain immediate safety",),
                recommendation_type="urgent safety support",
                route=safety_route,
                symptom_profile=symptom_profile,
                possible_clinical_overlaps=clinical_overlaps,
            )
        if _affirmative_risk(text):
            state.support_urgency = "emergency"
            state.pending_domain = "safety_immediacy"
            return ConversationReply(
                response=(
                    f"{generate_crisis_response('immediate', country_code)}\n\n"
                    "Please act on one of those options now and stay with another "
                    "person if possible."
                ),
                analysis=analysis,
                state=state,
                clinical_domains=("immediate safety concern",),
                recommendation_type="emergency help now",
                route=safety_route,
                symptom_profile=symptom_profile,
                possible_clinical_overlaps=clinical_overlaps,
            )
        if _contains_any(text, NO_TERMS):
            state.safety_followup = False
            state.covered_domains.add("safety")
            state.pending_domain = "supports"
            state.support_urgency = "urgent"
            return ConversationReply(
                response=(
                    "Thank you for answering directly. Even without immediate intent, "
                    "thoughts of death or self-harm deserve prompt professional support. "
                    "Please tell someone you trust today and arrange contact with a "
                    "licensed mental health professional.\n\n"
                    f"{GENERAL_QUESTIONS['supports']}"
                ),
                analysis=analysis,
                state=state,
                clinical_domains=("suicidal thoughts or self-harm",),
                recommendation_type="urgent professional support",
                route=safety_route,
                symptom_profile=symptom_profile,
                possible_clinical_overlaps=clinical_overlaps,
            )

    return None


def continue_conversation(
    text: str,
    state: ConversationState | None = None,
    country_code: str = "IN",
    support_mode: str = "Just listen",
    knowledge_override: str | None = None,
    approved_memory: tuple[str, ...] = (),
) -> ConversationReply:
    state = state or ConversationState()
    state.support_mode = (
        support_mode if support_mode in SUPPORT_MODES else "Just listen"
    )
    analysis = process_reflection(text)
    language_match = match_language_ontology(text)
    signals = detect_clinical_signals(text)
    current_symptom_profile = build_symptom_profile(text)
    current_clinical_overlaps = possible_clinical_overlaps(current_symptom_profile)

    safety_reply = _safety_reply(
        text,
        analysis,
        state,
        country_code,
        current_symptom_profile,
        current_clinical_overlaps,
    )
    if safety_reply:
        _update_symptom_layer(state, current_symptom_profile, current_clinical_overlaps)
        _update_narrative_memory(text, state)
        _update_state(text, analysis, signals, state, safety_reply.route)
        state.current_intent = safety_reply.route.intent
        state.knowledge_route = safety_reply.route.knowledge_route
        return safety_reply

    route = route_message(
        text,
        preferred_mode=state.support_mode,
        category=analysis.category.category,
        emotion=analysis.emotion.emotion,
        has_bias=bool(analysis.biases),
        clinical_domains=tuple(signal.domain for signal in signals),
        prior_intent=state.current_intent,
        has_active_context=bool(
            state.accumulated_domains or state.pending_domain
            or state.active_themes or state.narrative_progression
        ),
    )
    if knowledge_override == "internet":
        route = RouteDecision(
            intent="current factual search",
            response_mode="Research Assistant",
            knowledge_route="internet",
            confidence=1.0,
            reason="The user explicitly requested a current-facts search.",
        )
    elif knowledge_override == "research papers":
        route = RouteDecision(
            intent="research paper question",
            response_mode="Research Assistant",
            knowledge_route="research papers",
            confidence=1.0,
            reason="The user explicitly requested research evidence.",
        )
    if _asks_for_stress_driver(text):
        _update_narrative_memory(text, state)
        route = RouteDecision(
            intent="narrative_memory",
            response_mode="Insight",
            knowledge_route="conversation context",
            confidence=0.93,
            reason="The user asks for an accumulated-context explanation of their stress pattern.",
            topic="general",
        )
    current_explanations = possible_explanations(text, current_symptom_profile)
    current_style = select_response_style(
        intent=route.intent,
        emotion=analysis.emotion.emotion,
        last_5_styles=state.last_5_styles,
    )
    _update_reasoning_layer(state, current_explanations, current_style)
    _update_intervention_tracking(text, state)
    _update_narrative_memory(text, state)
    state.current_intent = route.intent
    state.knowledge_route = route.knowledge_route
    if (
        route.intent
        in {"grief", "anxiety / stress", "overwhelm", "emotional reflection"}
        and analysis.category.category != "general reflection"
    ):
        state.last_emotional_category = analysis.category.category

    if state.pending_domain == "supports":
        state.support_known = True
    if state.pending_domain == "goal":
        state.goal_known = True
    if state.pending_domain == "duration":
        state.duration_known = True
    if state.pending_domain == "functioning":
        state.functioning_known = True

    turn_index = state.turn_count
    _update_symptom_layer(state, current_symptom_profile, current_clinical_overlaps)
    _update_state(text, analysis, signals, state, route)

    routed_response = routed_local_response(
        text=text,
        route=route,
        analysis=analysis,
        turn_count=turn_index,
        approved_memory=approved_memory,
        conversation_state=state,
    )
    if routed_response is not None:
        routed_response = _with_clinical_overlap_note(
            routed_response,
            route,
            current_clinical_overlaps,
        )
        return ConversationReply(
            response=routed_response,
            analysis=analysis,
            state=state,
            clinical_domains=tuple(signal.domain for signal in signals),
            recommendation_type=state.support_urgency,
            route=route,
            symptom_profile=current_symptom_profile,
            possible_clinical_overlaps=current_clinical_overlaps,
            possible_explanations=current_explanations,
            reasoning_style=current_style,
            language_ontology_match=language_match,
        )

    if route.intent == "narrative_memory":
        return ConversationReply(
            response=_stress_driver_response(state),
            analysis=analysis,
            state=state,
            clinical_domains=tuple(signal.domain for signal in signals),
            recommendation_type=state.support_urgency,
            route=route,
            symptom_profile=current_symptom_profile,
            possible_clinical_overlaps=current_clinical_overlaps,
            possible_explanations=current_explanations,
            reasoning_style=current_style,
            language_ontology_match=language_match,
        )

    question_domain, question = _choose_question(signals, state)
    state.pending_domain = question_domain

    if question_domain == "safety":
        state.safety_followup = True

    response = _compose_human_response(
        text=text,
        mode=state.support_mode,
        analysis=analysis,
        signals=signals,
        state=state,
        question=_mode_question(state.support_mode, question, analysis, state),
    )
    response = _with_clinical_overlap_note(
        response,
        route,
        current_clinical_overlaps,
    )
    return ConversationReply(
        response=response,
        analysis=analysis,
        state=state,
        clinical_domains=tuple(
            sorted(
                state.accumulated_domains,
                key=state.accumulated_domains.get,
                reverse=True,
            )
        ),
        recommendation_type=state.support_urgency,
        route=route,
        symptom_profile=current_symptom_profile,
        possible_clinical_overlaps=current_clinical_overlaps,
        possible_explanations=current_explanations,
        reasoning_style=current_style,
        language_ontology_match=language_match,
    )
