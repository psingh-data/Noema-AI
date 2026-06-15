"""SQLite persistence for anonymized analytics only."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).with_name("noema.db")


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    with connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reflection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                emotion TEXT NOT NULL,
                category TEXT NOT NULL,
                intensity TEXT NOT NULL,
                detected_biases TEXT NOT NULL,
                response_style TEXT NOT NULL,
                safety_level TEXT NOT NULL,
                support_urgency TEXT NOT NULL DEFAULT 'routine',
                checkin_stage TEXT NOT NULL DEFAULT 'open reflection',
                clinical_domains TEXT NOT NULL DEFAULT '[]',
                recommendation_type TEXT NOT NULL DEFAULT 'routine',
                response_source TEXT NOT NULL DEFAULT 'local',
                support_mode TEXT NOT NULL DEFAULT 'Just listen',
                intent_route TEXT NOT NULL DEFAULT 'open conversation',
                knowledge_route TEXT NOT NULL DEFAULT 'conversation context',
                routed_mode TEXT NOT NULL DEFAULT 'Just listen',
                internet_used INTEGER NOT NULL DEFAULT 0,
                research_used INTEGER NOT NULL DEFAULT 0,
                sources_used TEXT NOT NULL DEFAULT '[]',
                confidence_level TEXT NOT NULL DEFAULT 'Medium',
                helpfulness_rating INTEGER
            )
            """
        )
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(reflection_events)"
            ).fetchall()
        }
        if "support_urgency" not in columns:
            connection.execute(
                "ALTER TABLE reflection_events ADD COLUMN "
                "support_urgency TEXT NOT NULL DEFAULT 'routine'"
            )
        if "checkin_stage" not in columns:
            connection.execute(
                "ALTER TABLE reflection_events ADD COLUMN "
                "checkin_stage TEXT NOT NULL DEFAULT 'open reflection'"
            )
        if "clinical_domains" not in columns:
            connection.execute(
                "ALTER TABLE reflection_events ADD COLUMN "
                "clinical_domains TEXT NOT NULL DEFAULT '[]'"
            )
        if "recommendation_type" not in columns:
            connection.execute(
                "ALTER TABLE reflection_events ADD COLUMN "
                "recommendation_type TEXT NOT NULL DEFAULT 'routine'"
            )
        if "response_source" not in columns:
            connection.execute(
                "ALTER TABLE reflection_events ADD COLUMN "
                "response_source TEXT NOT NULL DEFAULT 'local'"
            )
        if "support_mode" not in columns:
            connection.execute(
                "ALTER TABLE reflection_events ADD COLUMN "
                "support_mode TEXT NOT NULL DEFAULT 'Just listen'"
            )
        if "intent_route" not in columns:
            connection.execute(
                "ALTER TABLE reflection_events ADD COLUMN "
                "intent_route TEXT NOT NULL DEFAULT 'open conversation'"
            )
        if "knowledge_route" not in columns:
            connection.execute(
                "ALTER TABLE reflection_events ADD COLUMN "
                "knowledge_route TEXT NOT NULL DEFAULT 'conversation context'"
            )
        if "routed_mode" not in columns:
            connection.execute(
                "ALTER TABLE reflection_events ADD COLUMN "
                "routed_mode TEXT NOT NULL DEFAULT 'Just listen'"
            )
        if "internet_used" not in columns:
            connection.execute(
                "ALTER TABLE reflection_events ADD COLUMN "
                "internet_used INTEGER NOT NULL DEFAULT 0"
            )
        if "research_used" not in columns:
            connection.execute(
                "ALTER TABLE reflection_events ADD COLUMN "
                "research_used INTEGER NOT NULL DEFAULT 0"
            )
        if "sources_used" not in columns:
            connection.execute(
                "ALTER TABLE reflection_events ADD COLUMN "
                "sources_used TEXT NOT NULL DEFAULT '[]'"
            )
        if "confidence_level" not in columns:
            connection.execute(
                "ALTER TABLE reflection_events ADD COLUMN "
                "confidence_level TEXT NOT NULL DEFAULT 'Medium'"
            )


def record_event(
    *,
    emotion: str,
    category: str,
    intensity: str,
    detected_biases: list[str],
    response_style: str,
    safety_level: str,
    support_urgency: str = "routine",
    checkin_stage: str = "open reflection",
    clinical_domains: list[str] | None = None,
    recommendation_type: str = "routine",
    response_source: str = "local",
    support_mode: str = "Just listen",
    intent_route: str = "open conversation",
    knowledge_route: str = "conversation context",
    routed_mode: str = "Just listen",
    internet_used: bool = False,
    research_used: bool = False,
    sources_used: list[str] | None = None,
    confidence_level: str = "Medium",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Record classifications only. This API deliberately has no raw-text field."""
    initialize_database(db_path)
    with connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO reflection_events (
                emotion, category, intensity, detected_biases,
                response_style, safety_level, support_urgency, checkin_stage,
                clinical_domains, recommendation_type, response_source
                , support_mode, intent_route, knowledge_route, routed_mode,
                internet_used, research_used, sources_used, confidence_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                emotion,
                category,
                intensity,
                json.dumps(detected_biases),
                response_style,
                safety_level,
                support_urgency,
                checkin_stage,
                json.dumps(clinical_domains or []),
                recommendation_type,
                response_source,
                support_mode,
                intent_route,
                knowledge_route,
                routed_mode,
                int(internet_used),
                int(research_used),
                json.dumps(sources_used or []),
                confidence_level,
            ),
        )
        return int(cursor.lastrowid)


def save_feedback(
    event_id: int,
    rating: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    if rating not in {-1, 1}:
        raise ValueError("Feedback rating must be -1 or 1.")
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE reflection_events SET helpfulness_rating = ? WHERE id = ?",
            (rating, event_id),
        )


def fetch_events(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    initialize_database(db_path)
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM reflection_events ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]
