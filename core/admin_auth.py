"""Admin authentication helpers for protected Noema analytics."""

from __future__ import annotations

import hmac
import os
from collections.abc import Mapping
from typing import Any

import streamlit as st


def configured_admin_password(
    secrets: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Read the configured admin password without exposing it."""
    secrets = secrets if secrets is not None else st.secrets
    environ = environ if environ is not None else os.environ
    try:
        value = secrets.get("ADMIN_PASSWORD", "")
    except Exception:
        value = ""
    return str(value or environ.get("ADMIN_PASSWORD", "") or "")


def check_admin_password(candidate: str, configured: str) -> bool:
    if not configured or configured == "change-this-password":
        return False
    return hmac.compare_digest(str(candidate or ""), configured)


def exports_visible(is_authenticated: bool) -> bool:
    return bool(is_authenticated)


def is_admin_authenticated() -> bool:
    """Render admin login UI and return True only after a correct password."""
    if st.session_state.get("admin_authenticated") is True:
        return True

    configured = configured_admin_password()
    if not configured or configured == "change-this-password":
        st.warning(
            "Admin dashboard is locked because ADMIN_PASSWORD is not configured."
        )
        st.session_state.admin_authenticated = False
        return False

    st.subheader("Admin access")
    st.caption("Analytics and exports are private. Enter the admin password to continue.")
    password = st.text_input("Admin password", type="password")
    if st.button("Unlock admin dashboard"):
        if check_admin_password(password, configured):
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.session_state.admin_authenticated = False
            st.error("Incorrect admin password.")
    return False
