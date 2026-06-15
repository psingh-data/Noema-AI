"""Short-lived cache for cited factual and research answers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_CACHE_PATH = Path(__file__).with_name("retrieval_cache.db")
CACHE_VERSION = "tavily-v2-academic"


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    source_links: tuple[dict[str, str], ...]
    fetched_at: str
    expires_at: str
    confidence: str


def _connect(path: str | Path) -> sqlite3.Connection:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(resolved)


def initialize_cache(path: str | Path = DEFAULT_CACHE_PATH) -> None:
    with _connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_cache (
                cache_key TEXT PRIMARY KEY,
                answer TEXT NOT NULL,
                source_links TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                confidence TEXT NOT NULL
            )
            """
        )


def _key(query: str, knowledge_route: str) -> str:
    normalized = " ".join(query.lower().split())
    material = f"{CACHE_VERSION}:{knowledge_route}:{normalized}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def expiry_days(query: str, knowledge_route: str) -> int:
    normalized = query.lower()
    if knowledge_route == "research papers":
        return 30
    if any(
        marker in normalized
        for marker in (
            "visa",
            "deadline",
            "news",
            "price",
            "salary",
            "job market",
            "regulation",
            "law",
        )
    ):
        return 1
    if any(marker in normalized for marker in ("laptop", "computer", "phone", "buy")):
        return 7
    return 3


def get_cached_answer(
    query: str,
    knowledge_route: str,
    path: str | Path = DEFAULT_CACHE_PATH,
) -> CachedAnswer | None:
    initialize_cache(path)
    with _connect(path) as connection:
        row = connection.execute(
            """
            SELECT answer, source_links, fetched_at, expires_at, confidence
            FROM retrieval_cache WHERE cache_key = ?
            """,
            (_key(query, knowledge_route),),
        ).fetchone()
    if row is None:
        return None
    if datetime.fromisoformat(row[3]) <= datetime.now(timezone.utc):
        return None
    return CachedAnswer(
        answer=row[0],
        source_links=tuple(json.loads(row[1])),
        fetched_at=row[2],
        expires_at=row[3],
        confidence=row[4],
    )


def put_cached_answer(
    *,
    query: str,
    knowledge_route: str,
    answer: str,
    source_links: tuple[dict[str, str], ...],
    confidence: str = "Medium",
    path: str | Path = DEFAULT_CACHE_PATH,
) -> None:
    if not source_links:
        return
    initialize_cache(path)
    fetched = datetime.now(timezone.utc)
    expires = fetched + timedelta(days=expiry_days(query, knowledge_route))
    with _connect(path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO retrieval_cache (
                cache_key, answer, source_links, fetched_at, expires_at, confidence
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                _key(query, knowledge_route),
                answer,
                json.dumps(source_links),
                fetched.isoformat(),
                expires.isoformat(),
                confidence,
            ),
        )
