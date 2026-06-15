"""Aggregate, anonymous research dashboard."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from data.storage import fetch_events


st.set_page_config(page_title="Noema Dashboard", page_icon="N", layout="wide")
st.title("Noema Research Dashboard")
st.caption("Aggregate analytics only. Original reflections are never stored.")

events = fetch_events()
if not events:
    st.info("No anonymous reflection events have been recorded yet.")
    st.stop()

frame = pd.DataFrame(events)
total = len(frame)
helpful = int((frame["helpfulness_rating"] == 1).sum())
feedback_count = int(frame["helpfulness_rating"].notna().sum())

col1, col2, col3 = st.columns(3)
col1.metric("Reflections", total)
col2.metric("Feedback responses", feedback_count)
col3.metric(
    "Helpful responses",
    f"{(helpful / feedback_count):.0%}" if feedback_count else "No ratings",
)

emotion_counts = frame["emotion"].value_counts().rename_axis("emotion").reset_index(name="count")
category_counts = frame["category"].value_counts().rename_axis("category").reset_index(name="count")
intensity_counts = (
    frame["intensity"].value_counts().rename_axis("intensity").reset_index(name="count")
)
urgency_counts = (
    frame["support_urgency"]
    .value_counts()
    .rename_axis("support_urgency")
    .reset_index(name="count")
)
intent_counts = (
    frame["intent_route"]
    .value_counts()
    .rename_axis("intent_route")
    .reset_index(name="count")
)

left, right = st.columns(2)
left.plotly_chart(
    px.bar(emotion_counts, x="emotion", y="count", title="Most common emotions"),
    width="stretch",
)
right.plotly_chart(
    px.bar(category_counts, x="category", y="count", title="Life categories"),
    width="stretch",
)

st.plotly_chart(
    px.bar(
        intent_counts,
        x="intent_route",
        y="count",
        title="Conversation routes",
    ),
    width="stretch",
)

confidence_counts = (
    frame["confidence_level"]
    .value_counts()
    .rename_axis("confidence_level")
    .reset_index(name="count")
)
st.plotly_chart(
    px.bar(
        confidence_counts,
        x="confidence_level",
        y="count",
        title="Response confidence",
    ),
    width="stretch",
)

retrieval_frame = pd.DataFrame(
    {
        "retrieval": ["Internet used", "Research used"],
        "count": [
            int(frame["internet_used"].sum()),
            int(frame["research_used"].sum()),
        ],
    }
)
st.plotly_chart(
    px.bar(
        retrieval_frame,
        x="retrieval",
        y="count",
        title="External retrieval usage",
    ),
    width="stretch",
)

left, right = st.columns(2)
left.plotly_chart(
    px.pie(intensity_counts, names="intensity", values="count", title="Emotional intensity"),
    width="stretch",
)
right.plotly_chart(
    px.bar(
        urgency_counts,
        x="support_urgency",
        y="count",
        title="Suggested support urgency",
    ),
    width="stretch",
)

bias_names = [
    bias
    for value in frame["detected_biases"]
    for bias in json.loads(value or "[]")
]
if bias_names:
    bias_frame = (
        pd.Series(bias_names)
        .value_counts()
        .rename_axis("thinking_pattern")
        .reset_index(name="count")
    )
    st.plotly_chart(
        px.bar(
            bias_frame,
            x="thinking_pattern",
            y="count",
            title="Possible thinking patterns",
        ),
        width="stretch",
    )
else:
    st.info("No thinking patterns have been recorded yet.")

clinical_domains = [
    domain
    for value in frame["clinical_domains"]
    for domain in json.loads(value or "[]")
]
if clinical_domains:
    domain_frame = (
        pd.Series(clinical_domains)
        .value_counts()
        .rename_axis("clinical_domain")
        .reset_index(name="count")
    )
    st.plotly_chart(
        px.bar(
            domain_frame,
            x="clinical_domain",
            y="count",
            title="Cross-cutting conversation domains",
        ),
        width="stretch",
    )

rated = frame.dropna(subset=["helpfulness_rating"]).copy()
if not rated.empty:
    rated["feedback"] = rated["helpfulness_rating"].map({1: "Helpful", -1: "Not helpful"})
    feedback_frame = (
        rated["feedback"].value_counts().rename_axis("feedback").reset_index(name="count")
    )
    st.plotly_chart(
        px.bar(feedback_frame, x="feedback", y="count", title="User feedback"),
        width="stretch",
    )
