import inspect
from pathlib import Path

import admin_dashboard
from core.admin_auth import (
    check_admin_password,
    configured_admin_password,
    exports_visible,
)


def test_admin_page_blocked_without_password():
    assert not check_admin_password("", "secret-admin-password")


def test_wrong_password_denied():
    assert not check_admin_password("wrong", "secret-admin-password")


def test_correct_password_allowed():
    assert check_admin_password("secret-admin-password", "secret-admin-password")


def test_default_placeholder_password_is_not_allowed():
    assert not check_admin_password("change-this-password", "change-this-password")


def test_exports_hidden_without_auth():
    assert not exports_visible(False)
    assert exports_visible(True)


def test_admin_dashboard_uses_download_buttons_for_exports():
    source = inspect.getsource(admin_dashboard)
    assert "download_button" in source
    assert "Download CSV" in source
    assert "Download JSONL" in source
    assert "exports_visible" in source


def test_secret_is_read_from_mapping_without_exposure():
    assert (
        configured_admin_password(
            secrets={"ADMIN_PASSWORD": "secret-admin-password"},
            environ={},
        )
        == "secret-admin-password"
    )


def test_no_public_dashboard_page_in_sidebar():
    assert not Path("pages/1_Dashboard.py").exists()


def test_no_secret_exposed_in_admin_dashboard_source():
    source = inspect.getsource(admin_dashboard)
    assert "my-real-admin-password" not in source
    assert "st.write(configured" not in source
    assert "st.code(configured" not in source
    assert "ADMIN_PASSWORD = " not in source
