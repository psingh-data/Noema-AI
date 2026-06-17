"""Offline phrase and category matching for Noema's language ontology."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEXICON_PATHS = (
    PROJECT_ROOT / "data" / "processed" / "noema_language_lexicon.json",
    Path(
        r"C:\Users\prabh\Downloads\noema_psych_language_ontology_50000"
        r"\noema_language_ontology_dataset\noema_language_lexicon.json"
    ),
)

OFFLINE_CATEGORIES = {
    "pure_emotional_validation",
    "grief_disclosure",
    "venting",
    "loneliness",
    "casual_conversation",
    "identity_reflection",
    "relationship_feelings",
}

EXTERNAL_KNOWLEDGE_MARKERS = (
    "therapy",
    "therapies",
    "therapist",
    "counseling",
    "counselling",
    "intervention",
    "evidence",
    "research",
    "studies",
    "paper",
    "papers",
    "official",
    "resources",
    "helpline",
    "hotline",
    "current",
    "latest",
    "news",
    "visa",
    "deadline",
    "salary",
    "requirements",
    "regulation",
    "regulations",
)

GEN_Z_MARKERS = (
    " fr",
    " rn",
    "lowkey",
    "af",
    "cooked",
    "fried",
    "yap",
    "lore",
    "main character",
    "vibe",
    "ghosted",
    "situationship",
)

HINGLISH_MARKERS = (
    "yaar",
    "bhai",
    "didi",
    "kya",
    "nahi",
    "matlab",
    "bas",
)

SEMANTIC_RULES = (
    (
        "pure_emotional_validation",
        ("i'm cooked", "im cooked", "i am cooked", "i'm fried", "im fried"),
        "overwhelm",
        ("general emotional pain", "stress"),
        "gen_z_slang",
    ),
    (
        "identity_reflection",
        ("i lost my plot", "lost my plot", "i lost myself", "lost myself"),
        "identity diffusion",
        ("identity", "self-concept"),
        "gen_z_slang",
    ),
    (
        "loneliness",
        (
            "no one got me fr",
            "no one gets me",
            "nobody gets me",
            "nobody understands me",
            "no one understands me",
            "i feel invisible",
        ),
        "loneliness",
        ("belonging", "connection"),
        "gen_z_slang",
    ),
    (
        "venting",
        ("let me yap", "i just wanna yap", "i need to rant", "let me rant"),
        "frustration",
        ("release", "stress"),
        "gen_z_slang",
    ),
    (
        "relationship_feelings",
        (
            "this relationship is draining fr",
            "relationship is draining",
            "relationship feels draining",
            "i love her but i'm exhausted",
            "i love her but im exhausted",
            "i love him but i'm exhausted",
            "i love him but im exhausted",
            "we keep fighting",
        ),
        "attachment anxiety",
        ("relationship", "conflict"),
        "gen_z_slang",
    ),
    (
        "casual_conversation",
        ("hi", "hello", "hey", "yo", "sup", "wassup", "what's up", "whats up"),
        "neutral",
        ("small talk", "casual"),
        "gen_z_slang",
    ),
    (
        "casual_conversation",
        ("i'm bored", "im bored", "i am bored"),
        "boredom",
        ("small talk", "boredom"),
        "gen_z_slang",
    ),
    (
        "grief_disclosure",
        (
            "my grandfather died",
            "my grandmother died",
            "my mother passed away",
            "my father passed away",
            "my cat died",
            "my dog died",
            "i lost someone",
        ),
        "grief",
        ("loss", "bereavement"),
        "plain",
    ),
)

CATEGORY_ROUTE_MAP = {
    "pure_emotional_validation": (
        "emotional reflection",
        "Help me understand my feelings",
        "overwhelm",
    ),
    "grief_disclosure": ("grief", "Help me understand my feelings", "grief"),
    "venting": ("venting", "Just listen", "general"),
    "loneliness": ("emotional reflection", "Help me understand my feelings", "loneliness"),
    "casual_conversation": ("casual conversation", "Friend", "general"),
    "identity_reflection": (
        "identity_exploration",
        "Help me understand my feelings",
        "self_esteem",
    ),
    "relationship_feelings": (
        "emotional reflection",
        "Help me understand my feelings",
        "relationship",
    ),
}


@dataclass(frozen=True)
class LanguageOntologyMatch:
    matched: bool = False
    category: str = ""
    canonical_emotion: str = ""
    register: str = "plain"
    internet_needed: bool = False
    response_strategy: str = ""
    confidence: float = 0.0
    matched_phrase: str = ""
    possible_topics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def empty_language_ontology_match() -> LanguageOntologyMatch:
    return LanguageOntologyMatch()


def _normalize(text: str) -> str:
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _phrase_variants(phrase: str) -> tuple[str, ...]:
    normalized = _normalize(phrase)
    if not normalized:
        return ()
    variants = {normalized, normalized.replace("'", "")}
    return tuple(variants)


def _variant_matches(normalized_text: str, variant: str) -> tuple[bool, bool]:
    """Return (matched, exact) without allowing short tokens inside words."""
    if not variant:
        return False, False
    if normalized_text == variant:
        return True, True
    if len(variant.split()) >= 2 and variant in normalized_text:
        return True, False
    if len(variant.split()) == 1 and len(variant) >= 4:
        pattern = rf"\b{re.escape(variant)}\b"
        return bool(re.search(pattern, normalized_text)), False
    return False, False


def _has_external_request(text: str) -> bool:
    normalized = _normalize(text)
    return any(marker in normalized for marker in EXTERNAL_KNOWLEDGE_MARKERS)


def _detect_register(text: str, fallback: str = "plain") -> str:
    normalized = f" {_normalize(text)} "
    if any(marker in normalized for marker in HINGLISH_MARKERS):
        return "hinglish"
    if any(marker in normalized for marker in GEN_Z_MARKERS):
        return "gen_z_slang"
    return fallback


@lru_cache(maxsize=1)
def load_language_lexicon() -> dict[str, Any]:
    for path in DEFAULT_LEXICON_PATHS:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8-sig"))
    return {}


def _rule_match(text: str) -> LanguageOntologyMatch | None:
    normalized = _normalize(text)
    external = _has_external_request(text)
    for category, phrases, emotion, topics, register in SEMANTIC_RULES:
        for phrase in phrases:
            variants = _phrase_variants(phrase)
            matched = False
            exact = False
            for variant in variants:
                matched, exact = _variant_matches(normalized, variant)
                if matched:
                    break
            if matched:
                return LanguageOntologyMatch(
                    matched=True,
                    category=category,
                    canonical_emotion=emotion,
                    register=_detect_register(text, register),
                    internet_needed=external,
                    response_strategy=str(
                        load_language_lexicon()
                        .get(category, {})
                        .get("response_strategy", "")
                    ),
                    confidence=0.98 if exact else 0.9,
                    matched_phrase=phrase,
                    possible_topics=tuple(topics),
                )
    return None


def _lexicon_match(text: str) -> LanguageOntologyMatch | None:
    lexicon = load_language_lexicon()
    if not lexicon:
        return None
    normalized = _normalize(text)
    best: LanguageOntologyMatch | None = None
    external = _has_external_request(text)

    for category, info in lexicon.items():
        if category not in OFFLINE_CATEGORIES:
            continue
        for field, register in (
            ("phrases", "plain"),
            ("gen_z_slang", "gen_z_slang"),
            ("hinglish", "hinglish"),
        ):
            for phrase in info.get(field, []) or []:
                variants = _phrase_variants(str(phrase))
                if not variants:
                    continue
                matched = False
                exact = False
                for variant in variants:
                    matched, exact = _variant_matches(normalized, variant)
                    if matched:
                        break
                if not matched:
                    continue
                confidence = 0.97 if exact else 0.86
                if best and best.confidence >= confidence:
                    continue
                topics = tuple(str(item) for item in info.get("topics", []) or ())
                emotions = tuple(str(item) for item in info.get("emotions", []) or ())
                best = LanguageOntologyMatch(
                    matched=True,
                    category=str(category),
                    canonical_emotion=emotions[0] if emotions else "neutral",
                    register=_detect_register(text, register),
                    internet_needed=external,
                    response_strategy=str(info.get("response_strategy", "")),
                    confidence=confidence,
                    matched_phrase=str(phrase),
                    possible_topics=topics,
                )
    return best


def _topic_similarity_match(text: str) -> LanguageOntologyMatch | None:
    normalized = _normalize(text)
    external = _has_external_request(text)
    lexicon = load_language_lexicon()

    if "relationship" in normalized and any(
        word in normalized for word in ("draining", "tired", "toxic", "fighting")
    ):
        info = lexicon.get("relationship_feelings", {})
        return LanguageOntologyMatch(
            True,
            "relationship_feelings",
            "hurt",
            _detect_register(text),
            external,
            str(info.get("response_strategy", "")),
            0.84,
            "relationship + draining/conflict",
            tuple(info.get("topics", ()) or ("relationship",)),
        )
    if any(word in normalized for word in ("lonely", "alone", "unseen", "invisible")):
        info = lexicon.get("loneliness", {})
        return LanguageOntologyMatch(
            True,
            "loneliness",
            "loneliness",
            _detect_register(text),
            external,
            str(info.get("response_strategy", "")),
            0.82,
            "loneliness topic similarity",
            tuple(info.get("topics", ()) or ("belonging",)),
        )
    if (
        "grief" in normalized
        or "died" in normalized
        or "passed away" in normalized
        or "lost someone" in normalized
    ):
        info = lexicon.get("grief_disclosure", {})
        return LanguageOntologyMatch(
            True,
            "grief_disclosure",
            "grief",
            _detect_register(text),
            external,
            str(info.get("response_strategy", "")),
            0.84,
            "loss topic similarity",
            tuple(info.get("topics", ()) or ("loss",)),
        )
    return None


def match_language_ontology(text: str) -> LanguageOntologyMatch:
    """Return the strongest offline language-ontology match for a message."""
    for matcher in (_rule_match, _lexicon_match, _topic_similarity_match):
        match = matcher(text)
        if match is not None:
            return match
    return LanguageOntologyMatch()


def route_for_language_category(
    category: str,
) -> tuple[str, str, str] | None:
    return CATEGORY_ROUTE_MAP.get(category)
