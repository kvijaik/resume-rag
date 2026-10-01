"""Guardrail 1: queries that are too short or generic must be treated as unclear."""

import pytest

from app import config
from app.rag_chain import answer_query, is_query_too_vague


@pytest.mark.parametrize(
    "query",
    ["", "   ", "help", "hi there", "hello help resume", "test test test", "Resume? Help! Hi.", "? ? ?"],
)
def test_vague_queries_are_rejected(query):
    assert is_query_too_vague(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "Need a React frontend developer",
        "AWS DevOps engineer with Terraform",
        "help me find a Java developer",
    ],
)
def test_specific_queries_pass(query):
    assert is_query_too_vague(query) is False


def test_word_count_boundary(monkeypatch):
    monkeypatch.setattr(config, "MIN_QUERY_WORDS", 3)
    assert is_query_too_vague("Java developer") is True
    assert is_query_too_vague("Java backend developer") is False


def test_unclear_query_returns_capability_message_without_retrieval():
    # No vectorstore is passed and none is loaded: an unclear query must
    # short-circuit before any retrieval happens.
    result = answer_query("help")
    assert result == {"answer": config.CAPABILITY_MESSAGE, "status": "unclear", "sources": []}


def test_none_query_is_handled():
    assert answer_query(None)["status"] == "unclear"
