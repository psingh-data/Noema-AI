"""Search the local reference index without sending the whole book to a model."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INDEX = (
    Path(__file__).resolve().parent.parent / "data" / "reference_index.sqlite3"
)

@dataclass(frozen=True)
class ReferenceExcerpt:
    source: str
    page: int
    text: str


def index_available(index_path: str | Path = DEFAULT_INDEX) -> bool:
    return Path(index_path).exists()


def _query(text: str) -> str:
    return " OR ".join(f'"{term}"' for term in _terms(text)[:12])


def _terms(text: str) -> list[str]:
    terms = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text.lower())
    ignored = {
        "the",
        "and",
        "that",
        "this",
        "with",
        "from",
        "for",
        "have",
        "feel",
        "felt",
        "like",
        "what",
        "when",
        "where",
        "about",
        "been",
        "cannot",
        "could",
        "would",
    }
    return list(dict.fromkeys(term for term in terms if term not in ignored))


def retrieve(
    text: str,
    *,
    limit: int = 4,
    index_path: str | Path = DEFAULT_INDEX,
) -> list[ReferenceExcerpt]:
    path = Path(index_path)
    terms = _terms(text)[:12]
    query = _query(text)
    if not path.exists() or not query:
        return []

    try:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                """
                SELECT source, page, text, bm25(reference_chunks)
                FROM reference_chunks
                WHERE reference_chunks MATCH ?
                ORDER BY bm25(reference_chunks)
                LIMIT ?
                """,
                (query, max(limit * 8, 12)),
            ).fetchall()
    except sqlite3.Error:
        return []

    ranked = []
    for row in rows:
        haystack = row[2].lower()
        overlap = sum(term in haystack for term in terms)
        ranked.append((overlap, row[3], row))

    minimum_overlap = 2 if len(terms) >= 3 else 1
    relevant = [item for item in ranked if item[0] >= minimum_overlap]
    selected = sorted(relevant or ranked, key=lambda item: item[1])[:limit]
    return [
        ReferenceExcerpt(source=item[2][0], page=int(item[2][1]), text=item[2][2])
        for item in selected
    ]
