"""Scoring and per-resume ranking: cosine scores, top resumes (not top chunks), per-resume threshold."""

from collections import Counter

import pytest

from app import config
from app.rag_chain import answer_query, cosine_relevance, retrieve_resumes


@pytest.mark.parametrize("distance, score", [(0.0, 1.0), (1.0, 0.5), (2.0, 0.0), (2.5, 0.0), (-0.0001, 1.0)])
def test_cosine_relevance_maps_squared_distance_to_unit_range(distance, score):
    assert cosine_relevance(distance) == pytest.approx(score)


def test_returns_top_k_distinct_resumes_best_first(vectorstore):
    hits = retrieve_resumes(vectorstore, "Java backend developer with Spring Boot microservices")
    sources = list(dict.fromkeys(doc.metadata["source"] for doc, _ in hits))
    assert len(sources) == config.TOP_K
    assert sources[0] == "java_backend_springboot.txt"
    best = [next(s for d, s in hits if d.metadata["source"] == src) for src in sources]
    assert best == sorted(best, reverse=True)


def test_one_resume_cannot_fill_every_slot(vectorstore, monkeypatch):
    monkeypatch.setattr(config, "MAX_CHUNKS_PER_RESUME", 1)
    hits = retrieve_resumes(vectorstore, "Java backend developer with Spring Boot microservices", top_k=3)
    assert len(hits) == 3
    assert len({doc.metadata["source"] for doc, _ in hits}) == 3


def test_chunks_per_resume_are_capped(vectorstore):
    hits = retrieve_resumes(vectorstore, "Java backend developer with Spring Boot microservices", top_k=8)
    assert max(Counter(doc.metadata["source"] for doc, _ in hits).values()) <= config.MAX_CHUNKS_PER_RESUME


def test_weak_secondary_resumes_are_dropped(vectorstore, monkeypatch):
    query = "Java backend developer with Spring Boot microservices"
    hits = retrieve_resumes(vectorstore, query)
    best = {}
    for doc, score in hits:
        best.setdefault(doc.metadata["source"], score)
    top, second = sorted(best.values(), reverse=True)[:2]
    monkeypatch.setattr(config, "DEMO_MIN_SIMILARITY_SCORE", (top + second) / 2)  # only the top resume passes
    result = answer_query(query, vectorstore=vectorstore)
    assert result["status"] == "matched"
    assert result["sources"] == ["java_backend_springboot.txt"]
