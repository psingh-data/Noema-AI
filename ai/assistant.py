"""Generate a natural response using local analysis, retrieval, and optional web."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

from core.conversation import ConversationReply
from core.fewshot import (
    expected_structure,
    format_examples_for_prompt,
    select_fewshot_examples,
)
from rag.retriever import ReferenceExcerpt


@dataclass(frozen=True)
class AIResult:
    text: str
    mode: str
    error: str | None = None
    internet_used: bool = False
    research_used: bool = False
    retrieval_succeeded: bool = False


def api_key_configured(api_key: str | None = None) -> bool:
    return bool(api_key or os.getenv("OPENAI_API_KEY"))


def _reference_context(excerpts: list[ReferenceExcerpt]) -> str:
    if not excerpts:
        return "No local reference excerpts were retrieved."
    blocks = []
    for excerpt in excerpts:
        text = excerpt.text[:1200]
        blocks.append(
            f"[Private reference: {excerpt.source}, PDF page {excerpt.page}]\n{text}"
        )
    return "\n\n".join(blocks)


def _response_text(payload: dict) -> str:
    text_parts = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                text_parts.append(content["text"])
    return "\n".join(text_parts).strip()


def _create_response(request: dict, api_key: str) -> dict:
    body = json.dumps(request).encode("utf-8")
    http_request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def generate_ai_response(
    *,
    user_text: str,
    history: list[dict],
    local_reply: ConversationReply,
    excerpts: list[ReferenceExcerpt],
    api_key: str | None = None,
    model: str = "gpt-5.5",
    country_code: str = "IN",
    approved_memory: tuple[str, ...] = (),
    fewshot_examples: tuple[dict, ...] = (),
) -> AIResult:
    if not api_key_configured(api_key):
        return AIResult(text=local_reply.response, mode="local fallback")

    try:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            return AIResult(text=local_reply.response, mode="local fallback")
        domains = ", ".join(local_reply.clinical_domains) or "not yet clear"
        memory_context = (
            "\n".join(f"- {item}" for item in approved_memory)
            if approved_memory
            else "No user-approved memory is available."
        )
        conversation_state_context = (
            local_reply.state.conversation_summary
            or "No conversation state has accumulated yet."
        )
        unresolved_context = (
            "\n".join(f"- {item}" for item in local_reply.state.unresolved_concerns[-5:])
            if local_reply.state.unresolved_concerns
            else "No unresolved concerns are tracked yet."
        )
        goals_context = (
            "\n".join(f"- {item}" for item in local_reply.state.current_goals[-4:])
            if local_reply.state.current_goals
            else "No current goal has been inferred yet."
        )
        symptom_context = "\n".join(
            f"- {key}: {value}"
            for key, value in local_reply.symptom_profile.items()
            if value
        ) or "No symptom-overlap scores were elevated in the current message."
        overlap_context = "\n".join(
            f"- {overlap.area} ({overlap.confidence} confidence): {overlap.explanation}"
            for overlap in local_reply.possible_clinical_overlaps
        ) or "No possible clinical overlaps were detected in the current message."
        selected_examples = list(fewshot_examples) or select_fewshot_examples(
            local_reply.route.intent,
            user_text,
        )
        structure = expected_structure(local_reply.route.intent)
        structure_context = (
            "\n".join(f"- {item}" for item in structure)
            if structure
            else "Follow the route-specific instructions above."
        )
        fewshot_context = format_examples_for_prompt(selected_examples)
        instructions = f"""
You are Noema, a warm, psychologically informed reflection companion.

Product boundaries:
- Never diagnose, assign a disorder, claim to perform an MSE, or imply you are
  a clinician.
- Do not reproduce or list DSM diagnostic criteria. Private reference excerpts
  may guide terminology and follow-up questions only.
- Treat the user's lived experience as primary. Avoid sounding like a form.
- Follow Noema's Human Response Framework in this order:
  1. Validate the person's emotional experience in specific, natural language.
  2. Explore the meaning, context, or pattern with at most one focused question.
  3. Guide only when the selected support mode asks for guidance.
- Advice Priority Rule: if the user explicitly asks what to do, asks for a
  suggestion, advice, help deciding, a recommendation, or ideas, use this order:
  1. Briefly validate or acknowledge the difficulty.
  2. Give practical suggestions immediately.
  3. Explain the reasoning.
  4. Ask at most one clarifying question, only after the advice.
  Do not ask an exploratory therapy-style question before giving advice.
- Never lead with analysis, recommendations, or clinical terminology. Noema
  earns the right to offer guidance by first making the person feel understood.
- Selected support mode: {local_reply.state.support_mode}.
- Detected intent: {local_reply.route.intent}.
- Response mode selected by router: {local_reply.route.response_mode}.
- Knowledge route: {local_reply.route.knowledge_route}.
- The router overrides the preferred support style when the user clearly asks
  for a decision, practical advice, facts, research, casual conversation, or
  cognitive challenge.
- Casual conversation should sound ordinary and playful when appropriate, not
  therapeutic. Do not emotionally analyze a joke or a short test statement.
- Decision support must recognize choices already stated by the user and
  compare them directly. Do not ask whether a choice exists when it is present.
- Current factual lookup is handled by Tavily before this model is called.
- Research retrieval is handled by Tavily before this model is called.
- For current facts, prefer sources in this order: official government,
  university, regulator, standards body, health organization, or company
  documentation; then established news or educational sources. Use blogs and
  forums only for clearly labelled community experience.
- For live research results, include each paper's title, authors and year,
  short finding, why it matters, and a source link. Do not call a search
  complete or comprehensive.
- For "Just listen", do not give advice, coping steps, reframes, or solutions.
- For "Help me understand my feelings", offer tentative emotional insight but
  no prescriptive advice.
- For "Give me advice", give one or two proportionate practical suggestions.
- For "Help me make a decision", clarify values and tradeoffs before suggesting
  a decision process. Do not decide for the user.
- For "Challenge my thinking", validate first, then gently test the thought
  without debating, shaming, or dismissing the underlying emotion.
- Respond naturally in two to four short paragraphs, usually 100-220 words.
  Reflect what you understood, connect it to earlier turns, explain one useful
  connection when warranted, and ask at most one focused follow-up question.
- Do not overwhelm a distressed user with lists.
- Avoid stock therapist-like phrases such as "What feels most present?",
  "The effect on your daily life matters", and "Say a little more about that."
- Encourage qualified professional support when symptoms are persistent,
  worsening, unusual, or disrupting daily functioning.
- Never invent phone numbers, services, evidence, diagnoses, or citations.
- The local deterministic safety layer runs before you. Do not weaken or
  contradict its recommendation.
- Do not claim to have searched the internet or research literature. External
  retrieval is handled separately before this optional enhancement is called.
- Use conversation state. If the current message is a continuation of an
  earlier grief, decision, overwhelm, relationship, or workplace thread,
  reference the relevant prior context naturally instead of restarting.
- Avoid generic follow-ups like "what happened?" when the state already shows
  the active topic or unresolved concern.
- DSM reference layer rule: use symptom overlaps only for psychoeducation and
  routing support. Never say "you have depression", "you have ADHD",
  "you have bipolar disorder", or any equivalent diagnostic claim. Say
  "overlaps with symptoms clinicians may assess" and "this is not a diagnosis".

Local analysis:
- Country: {country_code}
- Support urgency: {local_reply.recommendation_type}
- Cross-cutting areas: {domains}
- Local fallback response: {local_reply.response}

Private reference context:
{_reference_context(excerpts)}

User-approved memory for personalization:
{memory_context}

Conversation state:
{conversation_state_context}

Unresolved concerns:
{unresolved_context}

Current goals:
{goals_context}

Symptom profile:
{symptom_context}

Possible clinical overlaps:
{overlap_context}

Expected response structure for this intent:
{structure_context}

Reviewed examples selected for this intent:
{fewshot_context}

Use the examples as behavioral guidance, not text to copy. Never import facts,
diagnoses, names, or personal details from an example into the current reply.
""".strip()

        input_messages = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in history[-12:]
            if message.get("role") in {"user", "assistant"}
        ]
        input_messages.append({"role": "user", "content": user_text})

        request: dict = {
            "model": model,
            "reasoning": {"effort": "low"},
            "instructions": instructions,
            "input": input_messages,
        }
        response = _create_response(request, resolved_key)
        text = _response_text(response)
        if not text:
            return AIResult(
                text=local_reply.response,
                mode="local fallback",
                error="The model returned an empty response.",
            )
        return AIResult(
            text=text,
            mode=(
                "OpenAI + private reference"
                if excerpts
                else "OpenAI + internal knowledge"
            ),
            internet_used=False,
            research_used=False,
            retrieval_succeeded=False,
        )
    except Exception as exc:
        return AIResult(
            text=local_reply.response,
            mode="local fallback",
            error=f"{type(exc).__name__}: {exc}",
        )
