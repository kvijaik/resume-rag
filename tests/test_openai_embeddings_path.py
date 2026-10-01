"""The OpenAI embeddings path (DEMO_MODE=0) for ingest and load, tested offline.

OpenAIEmbeddings is replaced with LangChain's DeterministicFakeEmbedding, so
the non-demo branches build and reload a real FAISS index with no network call.
"""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import langchain_openai
from app import config, ingest, rag_chain


@pytest.fixture
def fake_openai_embeddings(tmp_path, monkeypatch):
    """Switch to the non-demo path with fake embeddings and a temp index dir."""
    created = []

    def factory(**kwargs):
        created.append(kwargs)
        return DeterministicFakeEmbedding(size=64)

    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", factory)
    monkeypatch.setattr(config, "DEMO_MODE", False)
    monkeypatch.setattr(config, "VECTORSTORE_DIR", tmp_path)
    return created


def test_ingest_uses_openai_embeddings_when_not_in_demo_mode(fake_openai_embeddings, tmp_path, capsys):
    ingest.build_vectorstore()
    assert (tmp_path / "faiss_index" / "index.faiss").exists()
    assert not (tmp_path / "demo_vectorizer.pkl").exists()
    assert fake_openai_embeddings[0]["model"] == config.EMBEDDING_MODEL_NAME
    assert "Using OpenAI embeddings model" in capsys.readouterr().out


def test_load_vectorstore_uses_openai_embeddings_when_not_in_demo_mode(fake_openai_embeddings):
    ingest.build_vectorstore()
    store = rag_chain.load_vectorstore()
    assert isinstance(store.embeddings, DeterministicFakeEmbedding)
    assert len(fake_openai_embeddings) == 2  # once for ingest, once for load
    assert store.similarity_search("Spring Boot microservices", k=1)
