"""Build a local SQLite FTS index from a user-provided PDF."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from pypdf import PdfReader


DEFAULT_INDEX = Path(__file__).resolve().parent.parent / "data" / "reference_index.sqlite3"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _chunks(text: str, size: int = 1400, overlap: int = 180) -> list[str]:
    clean = _clean(text)
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + size)
        if end < len(clean):
            boundary = clean.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(clean[start:end])
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks


def build_index(
    pdf_path: str | Path,
    index_path: str | Path = DEFAULT_INDEX,
) -> dict:
    source = Path(pdf_path)
    destination = Path(index_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(source))

    with sqlite3.connect(destination) as connection:
        connection.execute("DROP TABLE IF EXISTS reference_chunks")
        connection.execute("DROP TABLE IF EXISTS reference_meta")
        connection.execute(
            """
            CREATE VIRTUAL TABLE reference_chunks USING fts5(
                source,
                page UNINDEXED,
                chunk UNINDEXED,
                text,
                tokenize='porter unicode61'
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE reference_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        chunk_count = 0
        skipped_pages = 0
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                skipped_pages += 1
                continue
            for chunk_number, chunk in enumerate(_chunks(page_text), start=1):
                connection.execute(
                    """
                    INSERT INTO reference_chunks(source, page, chunk, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (source.name, page_number, chunk_number, chunk),
                )
                chunk_count += 1

        metadata = {
            "source_name": source.name,
            "page_count": str(len(reader.pages)),
            "chunk_count": str(chunk_count),
            "skipped_pages": str(skipped_pages),
            "edition_note": (
                "The supplied file is DSM-5 (2013), not the later DSM-5-TR."
            ),
        }
        connection.executemany(
            "INSERT INTO reference_meta(key, value) VALUES (?, ?)",
            metadata.items(),
        )

    return {
        "source": source.name,
        "pages": len(reader.pages),
        "chunks": chunk_count,
        "skipped_pages": skipped_pages,
        "index": str(destination),
    }
