"""Consent-based feedback storage for Noema sessions."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_FEEDBACK_DB_PATH = Path(__file__).with_name("noema_feedback.db")
DEFAULT_EXPORT_DIR = Path(__file__).with_name("exports")

FEEDBACK_REASONS = (
    "Helpful",
    "Not helpful",
    "Too repetitive",
    "Too generic",
    "Did not understand me",
    "Wrong safety response",
    "Too academic",
    "Too short",
    "Not practical",
    "Other",
)

FAILURE_REASON_TYPES = {
    "Too repetitive": "repetitive",
    "Too generic": "generic",
    "Did not understand me": "misunderstood_user",
    "Wrong safety response": "wrong_safety_response",
    "Too academic": "too_academic",
    "Too short": "too_short",
    "Not practical": "not_practical",
    "Not helpful": "not_helpful",
    "Other": "other",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sanitize_content(content: str) -> str:
    """Remove obvious direct contact identifiers before storage."""
    sanitized = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[redacted-email]",
        content,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"(?<!\w)(?:\+?\d[\d\s().-]{6,}\d)(?!\w)",
        "[redacted-phone]",
        sanitized,
    )
    return sanitized


def connect(db_path: str | Path = DEFAULT_FEEDBACK_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_feedback_database(
    db_path: str | Path = DEFAULT_FEEDBACK_DB_PATH,
) -> None:
    with connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                user_label TEXT,
                consent_given INTEGER NOT NULL,
                app_version TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS response_metadata (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                detected_intent TEXT NOT NULL,
                detected_topic TEXT NOT NULL,
                detected_emotion TEXT NOT NULL,
                emotion_intensity TEXT NOT NULL,
                active_thread TEXT NOT NULL,
                conversation_stage TEXT NOT NULL,
                internet_used INTEGER NOT NULL,
                research_used INTEGER NOT NULL,
                sources_used TEXT NOT NULL,
                ontology_category TEXT NOT NULL,
                symptom_overlap TEXT NOT NULL,
                safety_level TEXT NOT NULL,
                critic_passed INTEGER NOT NULL,
                response_similarity_score REAL NOT NULL,
                FOREIGN KEY(message_id) REFERENCES messages(message_id)
                    ON DELETE CASCADE,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                assistant_message_id TEXT NOT NULL,
                rating TEXT NOT NULL CHECK(rating IN ('helpful', 'not_helpful')),
                reason TEXT NOT NULL,
                free_text_feedback TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE,
                FOREIGN KEY(assistant_message_id) REFERENCES messages(message_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS failure_patterns (
                failure_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                assistant_message_id TEXT NOT NULL,
                failure_type TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE,
                FOREIGN KEY(assistant_message_id) REFERENCES messages(message_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_metadata_session
                ON response_metadata(session_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_session
                ON feedback(session_id);
            CREATE INDEX IF NOT EXISTS idx_failure_patterns_session
                ON failure_patterns(session_id);
            """
        )


def create_session(
    *,
    session_id: str | None = None,
    user_label: str | None = None,
    consent_given: bool,
    app_version: str,
    db_path: str | Path = DEFAULT_FEEDBACK_DB_PATH,
) -> str:
    initialize_feedback_database(db_path)
    session_id = session_id or str(uuid.uuid4())
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO sessions (
                session_id, created_at, user_label, consent_given, app_version
            ) VALUES (?, COALESCE(
                (SELECT created_at FROM sessions WHERE session_id = ?),
                ?
            ), ?, ?, ?)
            """,
            (
                session_id,
                session_id,
                utc_now(),
                sanitize_content(user_label or "") or None,
                int(consent_given),
                app_version,
            ),
        )
    return session_id


def record_message(
    *,
    session_id: str,
    role: str,
    content: str,
    turn_index: int,
    message_id: str | None = None,
    db_path: str | Path = DEFAULT_FEEDBACK_DB_PATH,
) -> str:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")
    initialize_feedback_database(db_path)
    message_id = message_id or str(uuid.uuid4())
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO messages (
                message_id, session_id, timestamp, role, content, turn_index
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                utc_now(),
                role,
                sanitize_content(content),
                turn_index,
            ),
        )
    return message_id


def record_response_metadata(
    *,
    message_id: str,
    session_id: str,
    detected_intent: str,
    detected_topic: str,
    detected_emotion: str,
    emotion_intensity: str,
    active_thread: str,
    conversation_stage: str,
    internet_used: bool,
    research_used: bool,
    sources_used: list[str] | tuple[str, ...],
    ontology_category: str,
    symptom_overlap: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    safety_level: str,
    critic_passed: bool,
    response_similarity_score: float,
    db_path: str | Path = DEFAULT_FEEDBACK_DB_PATH,
) -> None:
    initialize_feedback_database(db_path)
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO response_metadata (
                message_id, session_id, detected_intent, detected_topic,
                detected_emotion, emotion_intensity, active_thread,
                conversation_stage, internet_used, research_used, sources_used,
                ontology_category, symptom_overlap, safety_level, critic_passed,
                response_similarity_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                detected_intent,
                detected_topic,
                detected_emotion,
                emotion_intensity,
                active_thread,
                conversation_stage,
                int(internet_used),
                int(research_used),
                json.dumps(list(sources_used)),
                ontology_category,
                json.dumps(list(symptom_overlap)),
                safety_level,
                int(critic_passed),
                float(response_similarity_score),
            ),
        )


def record_failure_pattern(
    *,
    session_id: str,
    assistant_message_id: str,
    failure_type: str,
    description: str,
    failure_id: str | None = None,
    db_path: str | Path = DEFAULT_FEEDBACK_DB_PATH,
) -> str:
    initialize_feedback_database(db_path)
    failure_id = failure_id or str(uuid.uuid4())
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO failure_patterns (
                failure_id, session_id, assistant_message_id, failure_type,
                description, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                failure_id,
                session_id,
                assistant_message_id,
                failure_type,
                sanitize_content(description),
                utc_now(),
            ),
        )
    return failure_id


def save_interaction_feedback(
    *,
    session_id: str,
    assistant_message_id: str,
    rating: str,
    reason: str,
    free_text_feedback: str = "",
    feedback_id: str | None = None,
    db_path: str | Path = DEFAULT_FEEDBACK_DB_PATH,
) -> str:
    if rating not in {"helpful", "not_helpful"}:
        raise ValueError("rating must be 'helpful' or 'not_helpful'")
    if reason not in FEEDBACK_REASONS:
        raise ValueError("Unsupported feedback reason")
    initialize_feedback_database(db_path)
    feedback_id = feedback_id or str(uuid.uuid4())
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO feedback (
                feedback_id, session_id, assistant_message_id, rating, reason,
                free_text_feedback, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                session_id,
                assistant_message_id,
                rating,
                reason,
                sanitize_content(free_text_feedback),
                utc_now(),
            ),
        )
    failure_type = FAILURE_REASON_TYPES.get(reason)
    if rating == "not_helpful" or failure_type:
        record_failure_pattern(
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            failure_type=failure_type or "not_helpful",
            description=reason,
            db_path=db_path,
        )
    return feedback_id


def delete_session_data(
    session_id: str,
    db_path: str | Path = DEFAULT_FEEDBACK_DB_PATH,
) -> None:
    initialize_feedback_database(db_path)
    with connect(db_path) as connection:
        connection.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,),
        )


def fetch_table(
    table: str,
    db_path: str | Path = DEFAULT_FEEDBACK_DB_PATH,
) -> list[dict[str, Any]]:
    if table not in {
        "sessions",
        "messages",
        "response_metadata",
        "feedback",
        "failure_patterns",
    }:
        raise ValueError("Unsupported table")
    initialize_feedback_database(db_path)
    with connect(db_path) as connection:
        rows = connection.execute(f"SELECT * FROM {table}").fetchall()
    return [dict(row) for row in rows]


def fetch_export_rows(
    db_path: str | Path = DEFAULT_FEEDBACK_DB_PATH,
) -> list[dict[str, Any]]:
    initialize_feedback_database(db_path)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                s.session_id,
                s.created_at AS session_created_at,
                s.user_label,
                s.consent_given,
                s.app_version,
                m.message_id AS assistant_message_id,
                m.timestamp AS assistant_timestamp,
                m.content AS assistant_content,
                m.turn_index,
                rm.detected_intent,
                rm.detected_topic,
                rm.detected_emotion,
                rm.emotion_intensity,
                rm.active_thread,
                rm.conversation_stage,
                rm.internet_used,
                rm.research_used,
                rm.sources_used,
                rm.ontology_category,
                rm.symptom_overlap,
                rm.safety_level,
                rm.critic_passed,
                rm.response_similarity_score,
                f.rating,
                f.reason,
                f.free_text_feedback,
                f.created_at AS feedback_created_at
            FROM response_metadata rm
            JOIN messages m ON m.message_id = rm.message_id
            JOIN sessions s ON s.session_id = rm.session_id
            LEFT JOIN feedback f ON f.assistant_message_id = rm.message_id
            ORDER BY m.timestamp ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def export_feedback_data(
    *,
    export_dir: str | Path = DEFAULT_EXPORT_DIR,
    db_path: str | Path = DEFAULT_FEEDBACK_DB_PATH,
) -> tuple[Path, Path]:
    rows = fetch_export_rows(db_path)
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    csv_path = export_path / "noema_feedback_export.csv"
    jsonl_path = export_path / "noema_feedback_export.jsonl"
    fieldnames = sorted({key for row in rows for key in row}) or ["empty"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return csv_path, jsonl_path


def dashboard_metrics(
    db_path: str | Path = DEFAULT_FEEDBACK_DB_PATH,
) -> dict[str, Any]:
    sessions = fetch_table("sessions", db_path)
    messages = fetch_table("messages", db_path)
    metadata = fetch_table("response_metadata", db_path)
    feedback = fetch_table("feedback", db_path)
    failures = fetch_table("failure_patterns", db_path)
    helpful = sum(1 for row in feedback if row["rating"] == "helpful")
    helpful_rate = helpful / len(feedback) if feedback else 0.0
    return {
        "total_sessions": len(sessions),
        "total_messages": len(messages),
        "helpful_rate": helpful_rate,
        "metadata": metadata,
        "feedback": feedback,
        "failure_patterns": failures,
    }
