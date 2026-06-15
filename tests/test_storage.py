import inspect
import sqlite3

from data.storage import initialize_database, record_event, save_feedback


def test_storage_schema_has_no_raw_text_column(tmp_path):
    db_path = tmp_path / "test.db"
    initialize_database(db_path)

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(reflection_events)").fetchall()
        }

    assert "text" not in columns
    assert "message" not in columns
    assert "transcript" not in columns
    assert "content" not in columns
    assert "support_urgency" in columns
    assert "checkin_stage" in columns
    assert "clinical_domains" in columns
    assert "recommendation_type" in columns
    assert "response_source" in columns
    assert "support_mode" in columns
    assert "intent_route" in columns
    assert "knowledge_route" in columns
    assert "routed_mode" in columns
    assert "internet_used" in columns
    assert "research_used" in columns
    assert "sources_used" in columns
    assert "confidence_level" in columns


def test_record_api_does_not_accept_raw_text():
    parameters = inspect.signature(record_event).parameters
    assert "text" not in parameters
    assert "message" not in parameters


def test_feedback_updates_an_event(tmp_path):
    db_path = tmp_path / "test.db"
    event_id = record_event(
        emotion="anxiety",
        category="study",
        intensity="medium",
        detected_biases=[],
        response_style="calm and reassuring",
        safety_level="none",
        db_path=db_path,
    )
    save_feedback(event_id, 1, db_path)

    with sqlite3.connect(db_path) as connection:
        rating = connection.execute(
            "SELECT helpfulness_rating FROM reflection_events WHERE id = ?",
            (event_id,),
        ).fetchone()[0]
    assert rating == 1
