"""Noema: a single-chatbox, safety-first reflection companion."""

from __future__ import annotations

import importlib
import os

import streamlit as st

import core.router as _router_module

importlib.reload(_router_module)

import core.bias_detector as _bias_detector_module
import core.classifier as _classifier_module
import core.conversation as _conversation_module
import core.emotion_detector as _emotion_detector_module
import core.pipeline as _pipeline_module
import core.response_strategy as _response_strategy_module
import core.routed_responses as _routed_responses_module
import critic as _critic_module


NOEMA_VERSION = "Noema v1.0"
ROUTING_BUILD_VERSION = "topic-aware-critic-2026-06-16-2"


def reload_local_routing_modules() -> None:
    """Keep Streamlit hot reload from using stale routing modules."""
    for module in (
        _router_module,
        _bias_detector_module,
        _classifier_module,
        _emotion_detector_module,
        _pipeline_module,
        _response_strategy_module,
        _routed_responses_module,
        _conversation_module,
        _critic_module,
    ):
        importlib.reload(module)


reload_local_routing_modules()

from ai.assistant import generate_ai_response
from ai.tavily import format_tavily_answer, search_tavily, should_use_tavily
from critic import critique_response, repair_response
from core.conversation import (
    ConversationState,
    continue_conversation,
    conversation_state_snapshot,
)
from core.fewshot import select_fewshot_examples
from core.foundational_research import foundational_references
from core.pipeline import ReflectionResult
from core.response_metadata import build_response_metadata
from core.router import (
    knowledge_for_suggestion,
    mode_for_suggestion,
    suggestions_for_intent,
)
from data.storage import initialize_database, record_event, save_feedback
from data.retrieval_cache import get_cached_answer, put_cached_answer
from rag.retriever import index_available, retrieve


st.set_page_config(
    page_title="Noema",
    page_icon="N",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --noema-ink: #26352f;
        --noema-sage: #638575;
        --noema-mist: #eef4f0;
    }
    .stApp { background: linear-gradient(180deg, #fafcfb 0%, #f3f7f4 100%); }
    .block-container { max-width: 820px; padding-top: 2.2rem; }
    h1, h2, h3 { color: var(--noema-ink); }
    .noema-subtitle { color: #587066; font-size: 1.05rem; margin-top: -0.8rem; }
    .notice {
        background: var(--noema-mist);
        border: 1px solid #dbe8e0;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        color: #40564d;
        margin: 1rem 0 1.4rem;
    }
    .disclaimer { color: #6a7771; font-size: 0.78rem; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

initialize_database()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = ConversationState()
if "approved_memories" not in st.session_state:
    st.session_state.approved_memories = []
if st.session_state.get("routing_build_version") != ROUTING_BUILD_VERSION:
    st.session_state.messages = []
    st.session_state.feedback_given = set()
    st.session_state.conversation_state = ConversationState()
    st.session_state.pop("next_support_mode", None)
    st.session_state.pop("next_knowledge_route", None)
    st.session_state.pop("suggestion_selected", None)
    st.session_state.routing_build_version = ROUTING_BUILD_VERSION


def setting(name: str, default=None):
    try:
        secret_value = st.secrets.get(name)
        if secret_value not in {None, ""}:
            return secret_value
        return os.getenv(name, default)
    except Exception:
        return os.getenv(name, default)


def enabled(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


api_key = setting("OPENAI_API_KEY")
tavily_api_key = setting("TAVILY_API_KEY")
model = setting("OPENAI_MODEL", "gpt-5.5")
web_search_enabled = enabled(setting("ENABLE_WEB_SEARCH", True))
reference_ready = index_available()


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.feedback_given = set()
    st.session_state.conversation_state = ConversationState()
    st.session_state.pop("next_support_mode", None)
    st.session_state.pop("next_knowledge_route", None)
    st.session_state.pop("suggestion_selected", None)


def render_notes(message: dict) -> None:
    result: ReflectionResult = message["result"]
    domains = message.get("clinical_domains", [])

    with st.expander("What Noema considered"):
        st.caption(
            "Tentative text-based signals only. They are not diagnoses or clinical findings."
        )

        intent_col, mode_col = st.columns(2)
        intent_col.write(
            f"**Detected intent:** {message.get('intent', 'general conversation').title()}"
        )
        mode_col.write(f"**Chosen mode:** {message.get('routed_mode', 'Conversation')}")
        st.write(f"**Topic:** {message.get('topic', 'general').title()}")
        st.write(f"**Reasoning style:** {message.get('reasoning_style', 'reflective')}")

        emotion_col, intensity_col = st.columns(2)
        emotion_col.write(f"**Emotion:** {result.emotion.emotion.title()}")
        intensity_col.write(f"**Intensity:** {result.emotion.intensity.title()}")

        internet_col, research_col = st.columns(2)
        internet_col.write(
            f"**Internet used:** {'Yes' if message.get('internet_used') else 'No'}"
        )
        research_col.write(
            f"**Research used:** {'Yes' if message.get('research_used') else 'No'}"
        )
        st.write(f"**Source:** {message.get('retrieval_provider') or 'None'}")
        st.write(
            f"**Number of sources used:** {len(message.get('source_links', []))}"
        )

        sources = message.get("knowledge_sources", [])
        st.write(f"**Sources used:** {', '.join(sources)}")
        source_links = message.get("source_links", [])
        if source_links:
            st.write("**Source links:**")
            for source in source_links:
                st.markdown(f"- [{source['title']}]({source['url']})")
        if message.get("cache_expires_at"):
            st.caption(
                "Cached source snapshot expires: "
                f"{message['cache_expires_at']}"
            )
        st.write(f"**Confidence:** {message.get('confidence_level', 'Medium')}")
        st.caption(message.get("confidence_reason", "Based on the available context."))
        st.write(
            f"**Reviewed response examples used:** "
            f"{message.get('fewshot_example_count', 0)}"
        )

        if result.biases:
            st.write("**Possible thinking patterns:**")
            for bias in result.biases:
                st.write(f"- {bias.name.title()}: {bias.explanation}")

        symptom_profile = message.get("symptom_profile", {})
        clinical_overlaps = message.get("possible_clinical_overlaps", [])
        if symptom_profile or clinical_overlaps:
            st.divider()
            st.write("**DSM Reference Layer**")
            st.caption(
                "Symptom-overlap framework only. This is not a diagnosis or clinical assessment."
            )
            if symptom_profile:
                st.write("**Symptom profile:**")
                for label, score in symptom_profile.items():
                    if score:
                        st.write(f"- {label.replace('_', ' ').title()}: {score}")
            if clinical_overlaps:
                st.write("**Possible clinical overlaps:**")
                for overlap in clinical_overlaps:
                    st.write(
                        f"- {overlap['area']} "
                        f"({overlap['confidence']} confidence)"
                    )
        possible_explanations = message.get("possible_explanations", [])
        if possible_explanations:
            st.write("**Possible explanations, not diagnoses:**")
            for explanation in possible_explanations:
                st.write(
                    f"- {explanation['label']} "
                    f"({explanation['confidence']} confidence): "
                    f"{explanation['reason']}"
                )

        state_snapshot = message.get("conversation_state", {})
        if state_snapshot:
            st.divider()
            st.write("**Conversation State**")
            st.write(
                "**Active Themes:** "
                + ", ".join(state_snapshot.get("active_themes", []) or ["None"])
            )
            st.write(
                "**Active Emotions:** "
                + ", ".join(state_snapshot.get("active_emotions", []) or ["None"])
            )
            st.write(
                "**Current Goals:** "
                + ", ".join(state_snapshot.get("current_goals", []) or ["None"])
            )
            st.write("**Unresolved Questions:**")
            unresolved = state_snapshot.get("unresolved_concerns", []) or ["None"]
            for concern in unresolved:
                st.write(f"- {concern}")
            st.write(
                "**Conversation Stage:** "
                + str(state_snapshot.get("conversation_stage") or "conversation")
            )
            st.write(
                "**Interventions Tried:** "
                + ", ".join(state_snapshot.get("interventions_tried", []) or ["None"])
            )
            st.write(
                "**Interventions Failed:** "
                + ", ".join(state_snapshot.get("interventions_failed", []) or ["None"])
            )
            st.write(
                "**Major Losses:** "
                + ", ".join(state_snapshot.get("major_losses", []) or ["None"])
            )
            st.write(
                "**Recurring Fears:** "
                + ", ".join(state_snapshot.get("recurring_fears", []) or ["None"])
            )
            st.write(
                "**Recurring Conflicts:** "
                + ", ".join(state_snapshot.get("recurring_conflicts", []) or ["None"])
            )
            st.write(
                "**Recurring Values:** "
                + ", ".join(state_snapshot.get("recurring_values", []) or ["None"])
            )
            st.write(
                "**Major Goals:** "
                + ", ".join(state_snapshot.get("major_goals", []) or ["None"])
            )
            if state_snapshot.get("conversation_summary"):
                st.caption(state_snapshot["conversation_summary"])

        reference_pages = message.get("reference_pages", [])
        if reference_pages:
            pages = ", ".join(str(page) for page in reference_pages)
            st.caption(f"Private background reference consulted: PDF page(s) {pages}.")

        if st.session_state.get("debug_mode"):
            st.divider()
            st.caption(f"Response backend: {message.get('response_source', 'local')}")
            st.caption(f"Knowledge route: {message.get('knowledge_route', 'internal')}")
            st.caption(
                f"Support urgency: {message.get('recommendation_type', 'routine')}"
            )
            if domains:
                st.caption("Clinical domains: " + ", ".join(domains))
            if message.get("ai_error"):
                st.error(message["ai_error"])
            if message.get("critic_failures"):
                st.caption(
                    "Response critic repair: "
                    + "; ".join(message["critic_failures"])
                )


def select_suggestion(label: str) -> None:
    mode = mode_for_suggestion(label)
    knowledge_route = knowledge_for_suggestion(label)
    if mode:
        st.session_state.next_support_mode = mode
    if knowledge_route:
        st.session_state.next_knowledge_route = knowledge_route
    if mode or knowledge_route:
        st.session_state.suggestion_selected = label


def render_suggestions(message: dict, index: int) -> None:
    suggestions = suggestions_for_intent(message.get("intent", ""))
    if not suggestions:
        return
    st.caption("Continue in a direction that would help:")
    columns = st.columns(len(suggestions))
    for column, label in zip(columns, suggestions):
        column.button(
            label,
            key=f"suggestion-{index}-{label}",
            on_click=select_suggestion,
            args=(label,),
            width="stretch",
        )


def remember_context(text: str) -> None:
    if text not in st.session_state.approved_memories:
        st.session_state.approved_memories.append(text)


def clear_memories() -> None:
    st.session_state.approved_memories = []


def record_reply(message: dict) -> int:
    result: ReflectionResult = message["result"]
    urgency = message["recommendation_type"]
    return record_event(
        emotion=result.emotion.emotion,
        category=result.category.category,
        intensity=result.emotion.intensity,
        detected_biases=[bias.name for bias in result.biases],
        response_style=result.tone,
        safety_level=result.safety.level,
        support_urgency=urgency,
        checkin_stage=st.session_state.conversation_state.pending_domain
        or "conversation",
        clinical_domains=list(message["clinical_domains"]),
        recommendation_type=urgency,
        response_source=message.get("response_source", "local"),
        support_mode=message.get("support_mode", "Just listen"),
        intent_route=message.get("intent", "general conversation"),
        knowledge_route=message.get("knowledge_route", "conversation context"),
        routed_mode=message.get("routed_mode", "Conversation"),
        internet_used=message.get("internet_used", False),
        research_used=message.get("research_used", False),
        sources_used=message.get("knowledge_sources", []),
        confidence_level=message.get("confidence_level", "Medium"),
    )


st.title("Noema")
st.markdown(
    '<p class="noema-subtitle">Noema is a psychology-informed AI companion that '
    "helps you understand, decide, learn, and grow.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About Noema")
    st.caption(
        "Experimental AI system for emotional reflection, decision support, "
        "and psychological reasoning."
    )
    st.caption("Not a substitute for professional mental health care.")
    st.caption(NOEMA_VERSION)
    st.divider()

    st.header("Conversation settings")
    country_name = st.selectbox(
        "Crisis-resource country",
        ("India", "United States", "Canada"),
        help="This controls the emergency and crisis numbers Noema shows.",
    )
    st.caption(
        "Optional OpenAI enhancement: available"
        if api_key
        else "Optional OpenAI enhancement: unavailable"
    )
    st.caption("Core conversation: available")
    st.caption(
        "Private reference: ready"
        if reference_ready
        else "Private reference: unavailable"
    )
    st.caption(
        "Live knowledge: available"
        if web_search_enabled and tavily_api_key
        else "Live knowledge: unavailable"
    )
    st.caption(
        "Research retrieval: available"
        if web_search_enabled and tavily_api_key
        else "Research retrieval: unavailable"
    )
    st.toggle("Developer/debug mode", key="debug_mode")
    st.caption(
        f"Approved session memories: {len(st.session_state.approved_memories)}"
    )
    if st.session_state.approved_memories:
        st.button("Clear session memory", on_click=clear_memories)

country_code = {
    "India": "IN",
    "United States": "US",
    "Canada": "CA",
}[country_name]

header_left, header_right = st.columns([6, 1])
with header_left:
    privacy_text = (
        "Messages are sent to the configured AI service to generate replies; raw "
        "messages are not written to Noema's local analytics database."
        if api_key or tavily_api_key
        else "Original messages remain only in this live session."
    )
    st.markdown(
        f"""
        <div class="notice">
        Ask a general question, think through a decision, explore an emotion,
        or request current facts and research. Noema chooses the route from
        your words; you do not need to select a mode first.<br><br>
        <strong>Privacy:</strong> {privacy_text}
        Personal context is remembered only for this session and only after
        you explicitly approve it.
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.button("Start over", on_click=reset_conversation, width="stretch")

if st.session_state.get("suggestion_selected"):
    st.caption(
        f"Next response preference: {st.session_state['suggestion_selected']}. "
        "You can still type naturally."
    )

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_notes(message)
            event_id = message["event_id"]
            if event_id not in st.session_state.feedback_given:
                st.caption("Was this response helpful?")
                helpful, unhelpful, _ = st.columns([1, 1, 5])
                if helpful.button("Yes", key=f"yes-{index}"):
                    save_feedback(event_id, 1)
                    st.session_state.feedback_given.add(event_id)
                    st.rerun()
                if unhelpful.button("No", key=f"no-{index}"):
                    save_feedback(event_id, -1)
                    st.session_state.feedback_given.add(event_id)
                    st.rerun()
            else:
                st.caption("Thank you for the feedback.")
            if index == len(st.session_state.messages) - 1:
                render_suggestions(message, index)
                candidate = message.get("memory_candidate")
                if candidate and candidate not in st.session_state.approved_memories:
                    st.button(
                        "Remember this for this session",
                        key=f"remember-{index}",
                        on_click=remember_context,
                        args=(candidate,),
                    )


prompt = st.chat_input("Think out loud or ask anything...")
if prompt and prompt.strip():
    clean_prompt = prompt.strip()
    support_mode = st.session_state.pop("next_support_mode", "Just listen")
    knowledge_override = st.session_state.pop("next_knowledge_route", None)
    st.session_state.pop("suggestion_selected", None)
    history_before = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": clean_prompt})

    reply = continue_conversation(
        clean_prompt,
        st.session_state.conversation_state,
        country_code=country_code,
        support_mode=support_mode,
        knowledge_override=knowledge_override,
        approved_memory=tuple(st.session_state.approved_memories),
    )
    st.session_state.conversation_state = reply.state
    fewshot_examples = tuple(
        select_fewshot_examples(reply.route.intent, clean_prompt)
    )

    safety_routed = (
        reply.analysis.safety.is_crisis
        or reply.recommendation_type
        in {
            "urgent safety support",
            "urgent professional support",
            "emergency help now",
        }
    )
    internet_route = reply.route.knowledge_route == "internet"
    research_route = reply.route.knowledge_route == "research papers"
    tavily_route = should_use_tavily(reply.route)
    external_route = tavily_route
    cached_answer = (
        get_cached_answer(clean_prompt, reply.route.knowledge_route)
        if external_route
        else None
    )
    use_private_reference = (
        not safety_routed
        and bool(api_key)
        and reply.route.knowledge_route == "private clinical reference"
    )
    reference_query = " ".join((clean_prompt, *reply.clinical_domains))
    excerpts = retrieve(reference_query, limit=2) if use_private_reference else []

    if safety_routed:
        response_text = reply.response
        response_source = "local safety system"
        ai_error = None
        internet_used = False
        research_used = False
        retrieval_succeeded = True
        source_links = []
        cache_used = False
        cache_expires_at = None
        retrieval_provider = "Local Safety"
        retrieval_confidence = "High"
    elif cached_answer:
        response_text = cached_answer.answer
        response_source = "validated retrieval cache"
        ai_error = None
        internet_used = tavily_route
        research_used = research_route
        retrieval_succeeded = True
        source_links = list(cached_answer.source_links)
        cache_used = True
        cache_expires_at = cached_answer.expires_at
        retrieval_provider = (
            "Tavily"
        )
        retrieval_confidence = cached_answer.confidence
    elif tavily_route and not (tavily_api_key and web_search_enabled):
        response_text = reply.response
        response_source = "local routing notice"
        ai_error = None
        internet_used = False
        research_used = False
        retrieval_succeeded = False
        source_links = []
        cache_used = False
        cache_expires_at = None
        retrieval_provider = None
        retrieval_confidence = "Low"
    elif tavily_route:
        spinner_text = (
            "Noema is checking academic sources..."
            if research_route
            else "Noema is checking current sources..."
        )
        with st.spinner(spinner_text):
            tavily_result = search_tavily(
                clean_prompt,
                api_key=tavily_api_key,
                research=research_route,
            )
        if tavily_result.succeeded:
            response_text = format_tavily_answer(
                tavily_result,
                research=research_route,
                advice=reply.route.response_mode == "Give me advice",
            )
            response_source = (
                "Tavily academic search"
                if research_route
                else "Tavily live search"
            )
            ai_error = None
            internet_used = True
            research_used = research_route
            retrieval_succeeded = True
            source_links = [
                {"title": source.title, "url": source.url}
                for source in tavily_result.sources
            ]
            retrieval_provider = "Tavily"
            retrieval_confidence = tavily_result.confidence
            cache_used = False
            cache_expires_at = None
            put_cached_answer(
                query=clean_prompt,
                knowledge_route=reply.route.knowledge_route,
                answer=response_text,
                source_links=tuple(source_links),
                confidence=retrieval_confidence,
            )
        else:
            response_text = reply.response
            response_source = "Tavily retrieval fallback"
            ai_error = tavily_result.error
            internet_used = False
            research_used = False
            retrieval_succeeded = False
            source_links = []
            retrieval_provider = "Tavily"
            retrieval_confidence = "Low"
            cache_used = False
            cache_expires_at = None
    else:
        with st.spinner("Noema is reflecting..."):
            ai_kwargs = {
                "user_text": clean_prompt,
                "history": history_before,
                "local_reply": reply,
                "excerpts": excerpts,
                "api_key": api_key,
                "model": model,
                "country_code": country_code,
                "approved_memory": tuple(st.session_state.approved_memories),
                "fewshot_examples": fewshot_examples,
            }
            try:
                ai_result = generate_ai_response(**ai_kwargs)
            except TypeError as exc:
                if "fewshot_examples" not in str(exc):
                    raise
                ai_kwargs.pop("fewshot_examples")
                ai_result = generate_ai_response(**ai_kwargs)
        response_text = ai_result.text
        response_source = ai_result.mode
        ai_error = ai_result.error
        internet_used = ai_result.internet_used
        research_used = ai_result.research_used
        retrieval_succeeded = ai_result.retrieval_succeeded
        source_links = []
        cache_used = False
        cache_expires_at = None
        retrieval_provider = None
        retrieval_confidence = "Medium" if retrieval_succeeded else None

    foundational = (
        foundational_references(clean_prompt)
        if reply.route.knowledge_route == "research papers"
        and not retrieval_succeeded
        else ()
    )
    foundational_used = bool(foundational)
    if foundational_used and not source_links:
        source_links = [
            {"title": reference.title, "url": reference.url}
            for reference in foundational
        ]
    memory_used = bool(
        st.session_state.approved_memories
        and (
            response_source.startswith("OpenAI")
            or reply.route.intent
            in {"decision support", "practical advice", "career / education"}
        )
    )
    critic_result = critique_response(
        user_input=clean_prompt,
        response=response_text,
        route=reply.route,
        internet_used=internet_used,
        research_used=research_used,
        safety_used=safety_routed,
    )
    critic_failures = critic_result.failures
    critic_repaired = False
    if not critic_result.passed:
        repaired_response = repair_response(
            user_input=clean_prompt,
            response=response_text,
            route=reply.route,
            failures=critic_result.failures,
        )
        if repaired_response != response_text:
            response_text = repaired_response
            critic_repaired = True
            critic_result = critique_response(
                user_input=clean_prompt,
                response=response_text,
                route=reply.route,
                internet_used=internet_used,
                research_used=research_used,
                safety_used=safety_routed,
            )

    metadata = build_response_metadata(
        route=reply.route,
        history_used=bool(history_before),
        private_reference_used=bool(excerpts),
        internet_used=internet_used,
        research_used=research_used,
        retrieval_succeeded=retrieval_succeeded,
        memory_used=memory_used,
        foundational_used=foundational_used,
        cache_used=cache_used,
        retrieval_provider=retrieval_provider,
        confidence_override=retrieval_confidence,
    )

    assistant_message = {
        "role": "assistant",
        "content": response_text,
        "result": reply.analysis,
        "clinical_domains": reply.clinical_domains,
        "recommendation_type": reply.recommendation_type,
        "response_source": response_source,
        "reference_pages": sorted({excerpt.page for excerpt in excerpts}),
        "ai_error": ai_error,
        "support_mode": support_mode,
        "intent": reply.route.intent,
        "topic": reply.route.topic,
        "routed_mode": reply.route.response_mode,
        "knowledge_route": reply.route.knowledge_route,
        "knowledge_sources": list(metadata.sources),
        "internet_used": metadata.internet_used,
        "research_used": metadata.research_used,
        "confidence_level": metadata.confidence_level,
        "confidence_reason": metadata.confidence_reason,
        "source_links": source_links,
        "cache_expires_at": cache_expires_at,
        "retrieval_provider": retrieval_provider,
        "memory_candidate": clean_prompt,
        "fewshot_example_count": len(fewshot_examples),
        "symptom_profile": reply.symptom_profile,
        "possible_clinical_overlaps": [
            {
                "area": overlap.area,
                "confidence": overlap.confidence,
                "score": overlap.score,
                "explanation": overlap.explanation,
            }
            for overlap in reply.possible_clinical_overlaps
        ],
        "possible_explanations": [
            {
                "label": explanation.label,
                "confidence": explanation.confidence,
                "reason": explanation.reason,
            }
            for explanation in reply.possible_explanations
        ],
        "reasoning_style": reply.reasoning_style,
        "conversation_state": conversation_state_snapshot(reply.state),
        "critic_passed": critic_result.passed,
        "critic_failures": list(critic_failures),
        "critic_repaired": critic_repaired,
    }
    assistant_message["event_id"] = record_reply(assistant_message)
    st.session_state.messages.append(assistant_message)
    st.rerun()


st.markdown("---")
st.markdown(
    '<p class="disclaimer">Noema is for reflection, education, and support. It is '
    "not a replacement for professional medical, psychological, legal, or financial "
    "advice. In immediate danger, contact local emergency services or a crisis line.</p>",
    unsafe_allow_html=True,
)
