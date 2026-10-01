"""The OpenAI generation path, tested offline with a fake chat model.

We reuse the demo FAISS index for retrieval, switch DEMO_MODE off, and replace
ChatOpenAI with LangChain's FakeListChatModel so no network call is made.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import langchain_openai
from app import config
from app.rag_chain import RESUME_MATCH_PROMPT, SYSTEM_PROMPT, answer_query

QUERY = "Need a React frontend developer with Redux experience"


@pytest.fixture
def fake_llm(vectorstore, monkeypatch):
    """Patch ChatOpenAI; return a setter for the canned reply and a log of prompts."""
    calls = []
    state = {"reply": ""}

    class RecordingFake(FakeListChatModel):
        def invoke(self, messages, *args, **kwargs):  # noqa: D401
            calls.append(messages)
            return super().invoke(messages, *args, **kwargs)

    def factory(**_kwargs):
        return RecordingFake(responses=[state["reply"]])

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", factory)
    monkeypatch.setattr(config, "DEMO_MODE", False)
    monkeypatch.setattr(config, "MIN_SIMILARITY_SCORE", 0.0)  # TF-IDF scores are lower than OpenAI's

    def set_reply(text):
        state["reply"] = text

    return set_reply, calls


def test_llm_answer_is_returned_as_matched(vectorstore, fake_llm):
    set_reply, _ = fake_llm
    set_reply("**Priya Sharma — React UI Developer**\n- Source file: react_ui_developer.txt")
    result = answer_query(QUERY, vectorstore=vectorstore)
    assert result["status"] == "matched"
    assert "react_ui_developer.txt" in result["sources"]


def test_llm_refusal_is_reported_as_no_match(vectorstore, fake_llm):
    set_reply, _ = fake_llm
    set_reply("Sorry, I cannot help with this query.")
    result = answer_query(QUERY, vectorstore=vectorstore)
    assert result["status"] == "no_match"
    assert result["sources"] == []


def test_prompt_contains_labelled_context_and_question(vectorstore, fake_llm):
    set_reply, calls = fake_llm
    set_reply("ok")
    answer_query(QUERY, vectorstore=vectorstore)
    prompt_value = calls[0]
    text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
    assert "[Source: " in text
    assert QUERY in text
    assert "Ground every claim strictly in the CONTEXT" in text


def test_system_prompt_keeps_exact_refusal_contract():
    # rag_chain matches on this prefix; the prompt and the check must stay in sync.
    assert '"Sorry, I cannot help with this query."' in SYSTEM_PROMPT
    assert set(RESUME_MATCH_PROMPT.input_variables) == {"context", "question"}
