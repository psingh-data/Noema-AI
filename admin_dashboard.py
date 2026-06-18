"""Password-protected Noema feedback analytics dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from core.admin_auth import exports_visible, is_admin_authenticated
from data.feedback_store import (
    DEFAULT_EXPORT_DIR,
    dashboard_metrics,
    export_feedback_data,
    fetch_table,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _count_chart(frame: pd.DataFrame, column: str, title: str) -> None:
    if frame.empty or column not in frame:
        st.info(f"No {title.lower()} yet.")
        return
    counts = frame[column].fillna("Unknown").value_counts().reset_index()
    counts.columns = [column, "count"]
    st.plotly_chart(px.bar(counts, x=column, y="count", title=title), width="stretch")


def _helpfulness_by_flag(
    metadata: pd.DataFrame,
    feedback: pd.DataFrame,
    flag: str,
    title: str,
) -> None:
    if metadata.empty or feedback.empty or flag not in metadata:
        st.info(f"No {title.lower()} data yet.")
        return
    joined = metadata.merge(
        feedback,
        left_on="message_id",
        right_on="assistant_message_id",
        how="inner",
    )
    if joined.empty:
        st.info(f"No {title.lower()} feedback yet.")
        return
    joined[flag] = joined[flag].map({1: "Yes", 0: "No"})
    summary = (
        joined.groupby(flag)["rating"]
        .apply(lambda values: (values == "helpful").mean())
        .reset_index(name="helpful_rate")
    )
    st.plotly_chart(px.bar(summary, x=flag, y="helpful_rate", title=title), width="stretch")


def _render_export_downloads(csv_path: Path, jsonl_path: Path) -> None:
    if not csv_path.exists() or not jsonl_path.exists():
        st.warning("Export files were not found. Generate the export again.")
        return
    left, right = st.columns(2)
    left.download_button(
        "Download CSV",
        data=csv_path.read_bytes(),
        file_name=csv_path.name,
        mime="text/csv",
        width="stretch",
    )
    right.download_button(
        "Download JSONL",
        data=jsonl_path.read_bytes(),
        file_name=jsonl_path.name,
        mime="application/x-jsonlines",
        width="stretch",
    )


def render_admin_dashboard() -> None:
    st.title("Noema Admin Dashboard")
    st.caption("Private feedback analytics. Password required before any data is shown.")

    if not is_admin_authenticated():
        st.info("Admin analytics, raw messages, feedback, and exports are hidden.")
        return

    metrics = dashboard_metrics()
    metadata = _frame(metrics["metadata"])
    feedback = _frame(metrics["feedback"])
    failures = _frame(metrics["failure_patterns"])
    sessions = _frame(fetch_table("sessions"))
    messages = _frame(fetch_table("messages"))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total sessions", metrics["total_sessions"])
    col2.metric("Total messages", metrics["total_messages"])
    col3.metric("Helpful rate", f"{metrics['helpful_rate']:.0%}")

    left, right = st.columns(2)
    with left:
        _count_chart(metadata, "detected_topic", "Most common topics")
    with right:
        _count_chart(metadata, "detected_emotion", "Most common emotions")

    left, right = st.columns(2)
    with left:
        _count_chart(feedback, "reason", "Most common failure reasons")
    with right:
        _count_chart(failures, "failure_type", "Failure patterns")

    repetitive = failures[failures["failure_type"] == "repetitive"] if not failures.empty else failures
    wrong_safety = (
        failures[failures["failure_type"] == "wrong_safety_response"]
        if not failures.empty
        else failures
    )
    col1, col2 = st.columns(2)
    col1.metric("Responses marked repetitive", 0 if repetitive.empty else len(repetitive))
    col2.metric("Wrong safety triggers", 0 if wrong_safety.empty else len(wrong_safety))

    left, right = st.columns(2)
    with left:
        _helpfulness_by_flag(
            metadata,
            feedback,
            "internet_used",
            "Internet used vs helpfulness",
        )
    with right:
        _helpfulness_by_flag(
            metadata,
            feedback,
            "research_used",
            "Research used vs helpfulness",
        )

    if not metadata.empty and "sources_used" in metadata:
        source_names: list[str] = []
        for value in metadata["sources_used"].fillna("[]"):
            try:
                source_names.extend(json.loads(value))
            except json.JSONDecodeError:
                continue
        if source_names:
            source_frame = (
                pd.Series(source_names)
                .value_counts()
                .reset_index(name="count")
                .rename(columns={"index": "source"})
            )
            st.plotly_chart(
                px.bar(source_frame, x="source", y="count", title="Sources used"),
                width="stretch",
            )

    st.divider()
    st.subheader("Admin export")
    if exports_visible(st.session_state.get("admin_authenticated", False)):
        if st.button("Generate feedback export"):
            csv_path, jsonl_path = export_feedback_data(export_dir=DEFAULT_EXPORT_DIR)
            st.session_state.noema_export_paths = (str(csv_path), str(jsonl_path))
            st.success("Feedback export generated.")
            st.code(str(csv_path))
            st.code(str(jsonl_path))
        export_paths = st.session_state.get("noema_export_paths")
        if export_paths:
            _render_export_downloads(Path(export_paths[0]), Path(export_paths[1]))
        st.caption(
            "Exports are written to data/exports/noema_feedback_export.csv and "
            "data/exports/noema_feedback_export.jsonl."
        )
    else:
        st.info("Exports are hidden until admin authentication succeeds.")

    with st.expander("Raw feedback tables"):
        st.caption("Visible only after admin authentication.")
        st.write("Sessions")
        st.dataframe(sessions, width="stretch")
        st.write("Messages")
        st.dataframe(messages, width="stretch")
        st.write("Response metadata")
        st.dataframe(metadata, width="stretch")
        st.write("Feedback")
        st.dataframe(feedback, width="stretch")
        st.write("Failure patterns")
        st.dataframe(failures, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Noema Admin", page_icon="N", layout="wide")
    render_admin_dashboard()


if __name__ == "__main__":
    main()
