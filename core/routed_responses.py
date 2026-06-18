"""Local fallback responses for non-clinical conversation routes."""

from __future__ import annotations

import re

from core.pipeline import ReflectionResult
from core.foundational_research import foundational_response
from core.language_ontology import match_language_ontology
from core.response_composer import compose_ontology_response
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


def _intervention_response(text: str) -> str:
    normalized = text.lower()
    if "grief" in normalized or "loss" in normalized or "grandfather" in normalized:
        focus = "grief"
        options = (
            "grief counseling or bereavement-focused therapy",
            "support groups with other people who have experienced loss",
            "CBT-style work for guilt, avoidance, sleep disruption, or stuck routines",
            "ACT or meaning-focused work for carrying love and loss without forcing closure",
            "gentle journaling or memory rituals when the grief feels hard to name",
        )
    else:
        focus = "this"
        options = (
            "CBT-style skills for thoughts, habits, and avoidance loops",
            "ACT-style work for values, acceptance, and action during emotional pain",
            "problem-solving therapy for turning vague overwhelm into next steps",
            "supportive counseling when you mainly need a steady person to process with",
            "professional assessment if symptoms are intense, persistent, or disrupting daily life",
        )

    return (
        "If this is affecting daily life, it makes sense to look for support that is "
        "specific, not just comforting words.\n\n"
        "**Direct answer:** several options may help, depending on what is happening "
        "for you and how intense or persistent it is.\n\n"
        f"For {focus}, common support options people discuss with clinicians include:\n"
        f"1. {options[0]}.\n"
        f"2. {options[1]}.\n"
        f"3. {options[2]}.\n"
        f"4. {options[3]}.\n"
        f"5. {options[4]}.\n\n"
        "**Plain-language explanation:** each option helps in a different way. Some "
        "focus on understanding the loss, some on changing painful thought loops, "
        "some on rebuilding daily functioning, and some on feeling less alone.\n\n"
        "This is not a diagnosis or a treatment plan. My practical recommendation is "
        "to choose the lowest-pressure first step: one consultation with a qualified "
        "mental health professional, or one structured support option, then judge "
        "whether it feels useful.\n\n"
        "What kind of support feels most realistic right now: private therapy, a "
        "support group, or self-guided practice?"
    )


def _symptom_education_response(text: str) -> str:
    normalized = text.lower()
    if "adhd" in normalized or "focus" in normalized or "concentrat" in normalized:
        return (
            "Not being able to focus can be genuinely upsetting, especially when it "
            "starts feeling like a flaw in you rather than a signal that something "
            "needs attention.\n\n"
            "Several explanations can look similar from the inside:\n"
            "1. ADHD-like attention regulation, especially if it has been present "
            "since childhood and happens across school, work, home, and relationships.\n"
            "2. Burnout or chronic stress, where your system is overused and attention "
            "starts collapsing.\n"
            "3. Anxiety, where worry uses up the mental space needed for focus.\n"
            "4. Depression-related low energy, where tiredness, emptiness, or low "
            "motivation make concentration harder.\n"
            "5. Poor sleep, overwhelm, perfectionism, or too much task pressure.\n\n"
            "That is why ADHD assessment is more complex than asking whether you can "
            "focus today. A clinician usually looks at how long this has been present, "
            "whether it started early in life, whether it appears in multiple settings, "
            "and whether sleep, anxiety, depression, substances, stress, or medical "
            "issues could explain it better.\n\n"
            "This is not a diagnosis. If the problem is persistent or disrupting study, "
            "work, relationships, or self-care, a professional assessment would be the "
            "right next step.\n\n"
            "Have these attention difficulties been present since childhood, or are "
            "they relatively recent?"
        )
    return (
        "I can explain possible symptom overlaps, but I should not diagnose you from "
        "a chat message. Similar experiences can come from stress, sleep, grief, "
        "anxiety, mood changes, medical issues, substances, or a diagnosable condition.\n\n"
        "The useful next step is to notice duration, intensity, daily-life impact, "
        "and whether the pattern appears in more than one setting. A qualified "
        "professional can assess that properly."
    )


def _continuity_response(text: str, route: RouteDecision, conversation_state: object | None) -> str | None:
    if conversation_state is None:
        return None
    active_themes = getattr(conversation_state, "active_themes", {})
    active_relationships = getattr(conversation_state, "active_relationships", set())
    relationship_signals = getattr(conversation_state, "relationship_signals", [])
    progression = getattr(conversation_state, "narrative_progression", [])
    normalized = " ".join(text.lower().split())
    grief_active = "grief" in active_themes
    active_thread = getattr(conversation_state, "active_thread", "general_thread")
    relationship_text = (
        sorted(active_relationships)[0]
        if active_relationships
        else "the person you lost"
    )

    relationship_active = active_thread == "relationship_thread" or route.topic == "relationship"
    if relationship_active:
        if (
            route.intent == "decision support"
            and getattr(conversation_state, "turn_count", 0) <= 1
            and not any(marker in normalized for marker in ("relief", "relieved", "guilt", "guilty", "what does that say"))
        ):
            return None
        if (
            "what does that say" in normalized
            or "decision" in normalized
            or route.intent == "decision support"
        ) and any(
            signal in relationship_signals
            for signal in (
                "relief when imagining leaving",
                "guilt about hurting partner",
                "relationship uncertainty about love",
                "commitment pressure",
            )
        ):
            return (
                "Taken together, the relief and the guilt are giving you different "
                "kinds of information.\n\n"
                "Relief may be telling you something about your needs: a part of you "
                "imagines leaving and feels less trapped, less pressured, or more like "
                "yourself. Guilt may be telling you something about your values: you "
                "do care about her pain, and you do not want to be careless with "
                "someone who loves you.\n\n"
                "Neither emotion automatically decides the relationship. The real "
                "question is whether staying is being driven by love, willingness, and "
                "a genuine future together, or by fear, responsibility, habit, and "
                "guilt.\n\n"
                "My recommendation: do not stay only to avoid hurting her, and do not "
                "leave impulsively just because relief appeared. Be honest with "
                "yourself first, then have a careful conversation rather than letting "
                "silence make the decision for both of you."
            )
        if "relief" in normalized or "relieved" in normalized:
            return (
                "That relief is important information. It does not automatically mean "
                "you should leave, but it does mean the idea of leaving is not only "
                "painful for you; some part of you experiences it as release.\n\n"
                "In relationship decisions, relief often points toward a need that has "
                "been compressed for too long: space, honesty, autonomy, less pressure, "
                "or an end to pretending. I would take that seriously without treating "
                "it as the whole answer.\n\n"
                "The useful next step is to ask: what exactly feels relieving about "
                "leaving: freedom, less guilt, less pressure, or not having to perform "
                "feelings you are unsure you have?"
            )
        if "guilt" in normalized or "guilty" in normalized or "hurting" in normalized:
            return (
                "That guilt makes sense if she loves you and you do not want to hurt "
                "her. But guilt is not the same thing as love, and it is not a stable "
                "reason to stay.\n\n"
                "Guilt may be showing your values: you want to be kind, honest, and "
                "responsible with another person's heart. The danger is when guilt "
                "turns into avoidance, because avoiding the truth can hurt someone "
                "more slowly.\n\n"
                "A kinder decision is not always the one that prevents pain today. It "
                "is the one that is honest, respectful, and does not keep both people "
                "inside a relationship that one person no longer chooses."
            )
        if "commitment" in normalized or (
            "unsure" in normalized and "commitment pressure" in relationship_signals
        ):
            return (
                "That sounds like commitment pressure: she may be asking for a clearer "
                "future, while you are not sure your feelings can honestly meet that.\n\n"
                "The important part is not to confuse kindness with commitment. You can "
                "care about her and still need to be honest that your certainty is not "
                "where hers is. A relationship can become painful when one person is "
                "asking for reassurance and the other is giving it mainly to avoid "
                "guilt.\n\n"
                "The next useful step is to separate two questions: do you want a future "
                "with her, and are you only afraid of hurting her if the answer is no?"
            )
        if (
            "don't think i love" in normalized
            or "do not think i love" in normalized
            or "not sure i love" in normalized
            or "exhausted" in normalized
            or "draining" in normalized
        ):
            return (
                "That sounds like relationship uncertainty, not a simple lack of "
                "emotion. There may be care, guilt, history, and pressure all mixed "
                "together, while the romantic part of you feels unclear or tired.\n\n"
                "When someone loves you but you are unsure you love them back, the "
                "hardest part is that kindness and honesty can feel like they are "
                "pulling in opposite directions. The relationship may feel draining "
                "because you are carrying both her expectations and your own doubt.\n\n"
                "For now, the key question is not 'how do I avoid hurting her?' It is "
                "'what is true enough that avoiding it would become unfair to both of us?'"
            )

    if active_thread == "adhd_thread" and route.intent == "conversation_continuity":
        if any(marker in normalized for marker in ("started last", "last year", "recent", "only started")):
            return (
                "That detail matters. If the focus difficulty only started last year, "
                "that makes ADHD less straightforward as an explanation, because ADHD "
                "patterns usually begin earlier in life and show up across more than "
                "one setting.\n\n"
                "Recent attention problems can also come from burnout, anxiety, grief, "
                "depression-like low energy, poor sleep, stress, or a big life transition. "
                "That does not rule ADHD out, but it means the timeline needs careful "
                "checking instead of jumping to a label.\n\n"
                "My practical next step: track when it started, what else changed around "
                "that time, sleep quality, stress level, and where it happens most. If it "
                "is disrupting study, work, or daily life, a professional assessment can "
                "sort the possibilities safely.\n\n"
                "What changed in your life around the time this started?"
            )
        return (
            "I am treating that as part of the attention/ADHD question we were already "
            "discussing, not as a new topic.\n\n"
            "The key is pattern: when it began, where it happens, what makes it worse, "
            "and whether sleep, anxiety, burnout, grief, or stress could explain it."
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

    if grief_active and route.topic == "grief" and any(
        marker in normalized for marker in ("unbearable", "too much", "can't take")
    ):
        return (
            f"This sounds connected to the grief around your {relationship_text}. "
            "When loss is this fresh or this heavy, the feeling can become less like "
            "sadness and more like your whole system saying, 'this is too much.'\n\n"
            "For the next few minutes, do not ask yourself to make meaning from it. "
            "Reduce the load: sit near someone safe if you can, drink water, and put "
            "one sentence to the feeling, even if the sentence is just 'I miss them "
            "and I cannot hold all of this right now.'\n\n"
            "If the feeling starts turning into thoughts of harming yourself or being "
            "unable to stay safe, treat that as urgent and contact emergency or crisis "
            "support immediately."
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


def _failed_intervention_response() -> str:
    return (
        "Thank you for saying that directly. If breathing or grounding did not help, "
        "repeating the same tool harder would probably feel frustrating, so let us "
        "switch layers.\n\n"
        "Sometimes the emotion is not mainly a breathing problem; it needs expression, "
        "support, movement, or meaning. Try one different route: say the feeling out "
        "loud without fixing it, write one uncensored paragraph, take a walk or shower "
        "to release body tension, or ask someone for quiet company instead of advice.\n\n"
        "My recommendation: stop forcing the tool that failed. Pick one different "
        "category of support for the next 10 minutes, then judge whether your body "
        "softens even slightly.\n\n"
        "Which route feels least annoying right now: expression, movement, or company?"
    )


def _identity_exploration_response() -> str:
    return (
        "That can feel frightening, like you are watching yourself from the outside. "
        "But it can also mean an old identity is no longer fitting the life you are "
        "actually living.\n\n"
        "Instead of trying to answer “Who am I?” as one giant question, break it into "
        "smaller pieces: what still feels genuinely yours, what you do mainly for "
        "approval, what values keep returning even when goals change, and what kind "
        "of person you respect even before they succeed.\n\n"
        "My recommendation: do not force a complete identity today. Start by naming "
        "what feels false, what still feels alive, and one behavior that would make "
        "you feel more like yourself this week.\n\n"
        "What part of your current life feels most unlike you?"
    )


def _achievement_self_worth_response() -> str:
    return (
        "If your worth is tied to achievement, every setback can feel like a threat "
        "to your whole self. That is exhausting because performance starts acting "
        "like proof of whether you deserve peace.\n\n"
        "Achievement is not the enemy. The problem is when it becomes the only place "
        "you are allowed to feel valuable. A healthier structure separates three "
        "things: worth, growth, and results. Worth is not earned daily. Growth is "
        "your responsibility. Results are influenced by effort, timing, opportunity, "
        "luck, and environment.\n\n"
        "Try this today: write one sentence for each: I am proud of achieving __. I "
        "am still worthy when __ fails. I want to become someone who values __ too."
    )


def _existential_response() -> str:
    return (
        "That question is heavy because it is not only philosophical; it touches "
        "fear, meaning, mortality, and the strange fact that we have to live without "
        "perfect certainty.\n\n"
        "One answer is that life does not need to last forever to matter. A song ends, "
        "a conversation ends, and people we love are not here forever, but temporary "
        "things can still be meaningful. Sometimes impermanence is what makes "
        "attention, love, and choice matter more.\n\n"
        "A practical way to hold this is: if life is temporary, what deserves more "
        "of your attention while you are here? You do not need a complete philosophy "
        "today. You need one honest direction that makes being alive feel less empty."
    )


def _ethical_response(text: str = "") -> str:
    lowered = text.lower()
    if "report" in lowered or "cheat" in lowered:
        return (
            "This is an ethical dilemma, not just a pros-and-cons choice. The tension "
            "is fairness, loyalty, consequences, and responsibility.\n\n"
            "If your friend cheated, the fair thing is to take it seriously, but the "
            "wise next step depends on harm and context: how serious the cheating was, "
            "whether others are being harmed, whether there is a safe reporting path, "
            "and whether speaking to your friend first could stop the behavior without "
            "creating unnecessary damage.\n\n"
            "My recommendation: do not cover it up or help them benefit unfairly. If "
            "the stakes are high or others are clearly harmed, use the official process. "
            "If the stakes are lower, consider one direct conversation first: tell them "
            "you are uncomfortable, ask them to correct it, and be clear you will not "
            "protect repeated cheating.\n\n"
            "What is the actual harm here: grade advantage, risk to others, or pressure "
            "on you to stay silent?"
        )
    return (
        "I would not answer this as “always tell the truth” or “always protect comfort.” "
        "Ethical choices depend on harm, consent, responsibility, timing, and power.\n\n"
        "A painful truth may be necessary when someone has a right to know, when hiding "
        "it would manipulate them, or when the truth helps them make an informed choice. "
        "A comforting lie may reduce pain briefly, but it can damage trust if it is "
        "discovered or if it protects the wrong thing.\n\n"
        "My practical rule: tell the truth when it protects autonomy or prevents harm, "
        "but deliver it with care. Avoid lying mainly to avoid your own discomfort. If "
        "work, legal, or safety risk is involved, document facts and get qualified advice."
    )


def _structured_problem_response() -> str:
    return (
        "Let us build a system, not a motivational speech.\n\n"
        "Use three layers: capture, prioritize, and start. Capture means one trusted "
        "place where every task goes. Prioritize means choosing only three meaningful "
        "tasks per day. Start means a 10-minute launch rule so the first step is too "
        "small to argue with.\n\n"
        "If attention regulation is part of this, reduce reliance on memory and "
        "willpower. Use visible cues, timers, body doubling, checklists, and friction "
        "removal. Your daily structure can be: morning top three, one focus block, one "
        "reset, and evening review.\n\n"
        "My recommendation: create the capture list today, then choose tomorrow's top "
        "three before you sleep."
    )


def _user_frustration_response() -> str:
    return (
        "You are right to call that out. If the response pattern is not helping, I "
        "should adapt instead of repeating the same kind of support in nicer words.\n\n"
        "Let me reset the approach. I will not keep pushing the previous suggestion. "
        "We can switch to one of five modes: comfort, practical steps, explanation, "
        "decision support, or just presence. If even choosing a mode feels annoying, "
        "we can keep it very simple and focus only on what would make the next few "
        "minutes less heavy.\n\n"
        "My recommendation: tell me the kind of help you wanted but did not get, and "
        "I will respond in that lane."
    )


def _continuity_repair_response() -> str:
    return (
        "I am treating that as connected to what came before, not as a random new "
        "message. If the previous suggestion did not help, the useful move is to "
        "change strategy instead of repeating it.\n\n"
        "If a body-based tool failed, the next layer may be expression or support: "
        "say the feeling plainly, write one uncensored paragraph, ask for silent "
        "company, change rooms, or do one practical stabilizing action like food, "
        "water, or a shower. The goal is not analysis right now. It is finding one "
        "form of relief that does not make you feel more trapped.\n\n"
        "What did the previous suggestion miss: your body, your thoughts, or the "
        "actual situation?"
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
    if (
        route.knowledge_route == "conversation context"
        and route.response_mode
        not in {
            "Give me advice",
            "Help me make a decision",
            "Challenge my thinking",
        }
    ):
        ontology_response = compose_ontology_response(
            text,
            match_language_ontology(text),
            turn_count=turn_count,
        )
        if ontology_response is not None:
            return ontology_response + memory_note
    continuity = _continuity_response(text, route, conversation_state)
    if continuity is not None:
        return continuity + memory_note
    strategic = strategy_response(text=text, route=route, analysis=analysis)
    if strategic is not None:
        return strategic + memory_note
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
    if route.intent == "failed_intervention_repair":
        return _failed_intervention_response()
    if route.intent == "identity_exploration":
        return _identity_exploration_response()
    if route.intent == "achievement_self_worth":
        return _achievement_self_worth_response()
    if route.intent == "existential_question":
        return _existential_response()
    if route.intent == "ethical_dilemma":
        return _ethical_response(text)
    if route.intent == "structured_problem_solving":
        return _structured_problem_response()
    if route.intent == "user_frustration_repair":
        return _user_frustration_response()
    if route.intent == "conversation_continuity":
        return _continuity_repair_response()
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
    if route.intent == "intervention_request":
        return _intervention_response(text)
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
        return _symptom_education_response(text)
    if route.intent == "general knowledge":
        if "grass" in text.lower() and "green" in text.lower():
            return (
                "Grass looks green mainly because of chlorophyll, the pigment plants "
                "use to absorb light for photosynthesis. Chlorophyll absorbs more red "
                "and blue light, while green light is reflected back to our eyes, so "
                "we perceive the grass as green.\n\n"
                "Small caveat: the exact shade can change with plant health, water, "
                "season, soil, and light."
            )
        return (
            "Here is the honest short version: I can help reason through this, and "
            "if it depends on current facts I would verify it with sources instead "
            "of guessing.\n\n"
            "For stable questions, I will give the clearest general explanation I can "
            "and flag uncertainty where it matters. For current or official details, "
            "the safer move is to use live retrieval."
        )
    if route.intent == "general conversation":
        return (
            "I am following. Give me a bit more context and I will help you work "
            "through it."
        )
    return None
