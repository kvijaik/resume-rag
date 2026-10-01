"""Streamlit UI, driven headlessly with Streamlit's own AppTest harness.

The app runs in-process, so the conftest monkeypatches on app.config (temp
index, DEMO_MODE) apply to it. The cached vectorstore is cleared before each
test so every test loads the index it configured.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import config

APP_FILE = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


@pytest.fixture
def app(vectorstore):
    """A fresh AppTest instance wired to the temp demo index."""
    st.cache_resource.clear()
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    yield at
    st.cache_resource.clear()


def test_first_load_shows_greeting_banner_and_sidebar(app):
    app.run()
    assert not app.exception
    assert app.title[0].value.endswith("Resume RAG Assistant")
    assert "DEMO_MODE" in app.info[0].value
    assert "Hi! Describe a hiring requirement" in app.chat_message[0].markdown[0].value
    sidebar_text = " ".join(m.value for m in app.sidebar.markdown)
    assert "Machine Learning Engineer" in sidebar_text
    assert len(app.sidebar.button) == 4


def test_no_demo_banner_outside_demo_mode(app, monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", False)
    # Keep loading the demo index: only the banner check reads DEMO_MODE before the load.
    from app import rag_chain

    demo_store = rag_chain.load_vectorstore()
    monkeypatch.setattr(rag_chain, "load_vectorstore", lambda: demo_store)
    app.run()
    assert not app.exception
    assert len(app.info) == 0


def test_typed_query_shows_match_badge_and_sources(app):
    app.run()
    app.chat_input[0].set_value("Need a React frontend developer with Redux experience").run()
    assert not app.exception
    assistant = app.chat_message[-1]
    assert assistant.markdown[0].value == "**✅ Match found**"
    assert "react_ui_developer.txt" in assistant.caption[0].value
    assert app.session_state["messages"][-2]["content"].startswith("Need a React")


def test_unclear_query_shows_badge_without_sources(app):
    app.run()
    app.chat_input[0].set_value("help").run()
    assistant = app.chat_message[-1]
    assert assistant.markdown[0].value == "**❓ Requirement unclear**"
    assert len(assistant.caption) == 0


def test_example_button_runs_its_query(app):
    app.run()
    query = "Need a React frontend developer with Redux experience"
    app.button(key=f"ex_{query}").click().run()
    assert not app.exception
    assert app.chat_message[-2].markdown[0].value == query
    assert app.chat_message[-1].markdown[0].value == "**✅ Match found**"
    assert "pending_query" not in app.session_state


def test_missing_index_shows_error_and_stops(app, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VECTORSTORE_DIR", tmp_path)
    app.run()
    assert "Run `python -m app.ingest`" in app.error[0].value
    assert len(app.chat_message) == 0


def test_importing_the_module_does_not_render_the_page(vectorstore):
    import runpy

    namespace = runpy.run_path(APP_FILE, run_name="streamlit_app")
    assert set(namespace["STATUS_BADGES"]) == {"matched", "no_match", "unclear"}
