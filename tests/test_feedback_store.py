import csv
import json
import sqlite3

from data.feedback_store import (
    FEEDBACK_REASONS,
    create_session,
    delete_session_data,
    export_feedback_data,
    fetch_table,
    initialize_feedback_database,
    record_message,
    record_response_metadata,
    sanitize_content,
    save_interaction_feedback,
)


def test_feedback_database_schema_has_required_tables(tmp_path):
    db_path = tmp_path / "feedback.db"
    initialize_feedback_database(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {
        "sessions",
        "messages",
        "response_metadata",
        "feedback",
        "failure_patterns",
    }.issubset(tables)


def test_feedback_roundtrip_export_and_delete(tmp_path):
    db_path = tmp_path / "feedback.db"
    export_dir = tmp_path / "exports"
    session_id = create_session(
        session_id="session-1",
        user_label="test label",
        consent_given=True,
        app_version="test",
        db_path=db_path,
    )
    user_message_id = record_message(
        session_id=session_id,
        role="user",
        content="hello user@example.com +1 555 111 2222",
        turn_index=0,
        db_path=db_path,
    )
    assistant_message_id = record_message(
        session_id=session_id,
        role="assistant",
        content="Helpful response",
        turn_index=1,
        db_path=db_path,
    )
    record_response_metadata(
        message_id=assistant_message_id,
        session_id=session_id,
        detected_intent="grief",
        detected_topic="grief",
        detected_emotion="sadness",
        emotion_intensity="medium",
        active_thread="grief_thread",
        conversation_stage="initial_disclosure",
        internet_used=False,
        research_used=False,
        sources_used=["Internal Knowledge"],
        ontology_category="grief_disclosure",
        symptom_overlap=[],
        safety_level="none",
        critic_passed=True,
        response_similarity_score=0.0,
        db_path=db_path,
    )
    save_interaction_feedback(
        session_id=session_id,
        assistant_message_id=assistant_message_id,
        rating="not_helpful",
        reason="Too generic",
        free_text_feedback="Use more context.",
        db_path=db_path,
    )

    messages = fetch_table("messages", db_path)
    assert user_message_id in {row["message_id"] for row in messages}
    stored_user = next(row for row in messages if row["message_id"] == user_message_id)
    assert "[redacted-email]" in stored_user["content"]
    assert "[redacted-phone]" in stored_user["content"]

    failures = fetch_table("failure_patterns", db_path)
    assert failures[0]["failure_type"] == "generic"

    csv_path, jsonl_path = export_feedback_data(export_dir=export_dir, db_path=db_path)
    assert csv_path.name == "noema_feedback_export.csv"
    assert jsonl_path.name == "noema_feedback_export.jsonl"
    with csv_path.open(encoding="utf-8") as handle:
        exported_rows = list(csv.DictReader(handle))
    assert exported_rows[0]["detected_intent"] == "grief"
    with jsonl_path.open(encoding="utf-8") as handle:
        exported_json = [json.loads(line) for line in handle]
    assert exported_json[0]["reason"] == "Too generic"

    delete_session_data(session_id, db_path)
    assert fetch_table("sessions", db_path) == []
    assert fetch_table("messages", db_path) == []


def test_feedback_reasons_match_product_buttons():
    assert FEEDBACK_REASONS == (
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


def test_sanitize_content_redacts_contact_identifiers():
    sanitized = sanitize_content("Email me at person@example.com or 987-654-3210.")
    assert "person@example.com" not in sanitized
    assert "987-654-3210" not in sanitized
    assert "[redacted-email]" in sanitized
    assert "[redacted-phone]" in sanitized
