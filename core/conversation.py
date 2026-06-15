"""Adaptive, psychologically informed multi-turn conversation."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.clinical_signals import ClinicalSignal, detect_clinical_signals
from core.pipeline import ReflectionResult, process_reflection
from core.routed_responses import routed_local_response
from core.router import RouteDecision, route_message
from core.safety import assess_safety, generate_crisis_response


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


@dataclass(frozen=True)
class ConversationReply:
    response: str
    analysis: ReflectionResult
    state: ConversationState
    clinical_domains: tuple[str, ...]
    recommendation_type: str
    route: RouteDecision


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
) -> str:
    emotion = analysis.emotion.emotion
    category = (
        state.last_emotional_category
        if analysis.category.category == "general reflection"
        and state.last_emotional_category
        else analysis.category.category
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
            return (
                "What do you miss most right now: the person, the time you had "
                "together, a particular memory, or something else?"
            )
        return "What part of this is hardest right now?"
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
            "There is no need to turn this into a problem to solve right away. "
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
    mode: str,
    analysis: ReflectionResult,
    signals: list[ClinicalSignal],
    state: ConversationState,
    question: str,
) -> str:
    if mode == "Give me advice":
        guidance = _mode_guidance(mode, analysis, state)
        sections = [_validation(analysis, state)]
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
        _validation(analysis, state),
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
    signals = detect_clinical_signals(text)

    safety_reply = _safety_reply(text, analysis, state, country_code)
    if safety_reply:
        state.turn_count += 1
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
    state.current_intent = route.intent
    state.knowledge_route = route.knowledge_route
    if (
        route.intent
        in {"grief", "anxiety / stress", "overwhelm", "emotional reflection"}
        and analysis.category.category != "general reflection"
    ):
        state.last_emotional_category = analysis.category.category

    routed_response = routed_local_response(
        text=text,
        route=route,
        analysis=analysis,
        turn_count=state.turn_count,
        approved_memory=approved_memory,
    )
    if routed_response is not None:
        if signals:
            _update_state(text, analysis, signals, state)
        else:
            state.turn_count += 1
        return ConversationReply(
            response=routed_response,
            analysis=analysis,
            state=state,
            clinical_domains=tuple(signal.domain for signal in signals),
            recommendation_type=state.support_urgency,
            route=route,
        )

    if state.pending_domain == "supports":
        state.support_known = True
    if state.pending_domain == "goal":
        state.goal_known = True
    if state.pending_domain == "duration":
        state.duration_known = True
    if state.pending_domain == "functioning":
        state.functioning_known = True

    _update_state(text, analysis, signals, state)

    question_domain, question = _choose_question(signals, state)
    state.pending_domain = question_domain

    if question_domain == "safety":
        state.safety_followup = True

    response = _compose_human_response(
        mode=state.support_mode,
        analysis=analysis,
        signals=signals,
        state=state,
        question=_mode_question(state.support_mode, question, analysis, state),
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
    )
