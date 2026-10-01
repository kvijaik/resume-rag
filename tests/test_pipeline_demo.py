"""End-to-end retrieval + guardrail behaviour in offline DEMO_MODE."""

import pytest

from app import config
from app.rag_chain import answer_query

MATCH_CASES = [
    ("Need a React frontend developer with Redux experience", "react_ui_developer.txt"),
    ("Looking for an AWS network security engineer with firewall experience", "network_security_engineer_aws.txt"),
    ("Application security engineer with OWASP and penetration testing skills", "application_security_engineer.txt"),
    ("Java backend developer with Spring Boot microservices", "java_backend_springboot.txt"),
]


@pytest.mark.parametrize("query,expected_source", MATCH_CASES)
def test_known_roles_match_and_cite_the_right_resume(vectorstore, query, expected_source):
    result = answer_query(query, vectorstore=vectorstore)
    assert result["status"] == "matched"
    assert expected_source in result["sources"]
    assert expected_source in result["answer"]


@pytest.mark.parametrize(
    "query",
    [
        "what is the weather",
        "I need a professional chef with 10 years of experience in Italian cuisine",
    ],
)
def test_out_of_scope_queries_are_refused(vectorstore, query):
    result = answer_query(query, vectorstore=vectorstore)
    assert result == {"answer": config.REJECTION_MESSAGE, "status": "no_match", "sources": []}


def test_zero_vocabulary_overlap_is_refused(vectorstore):
    # All-zero TF-IDF vector would otherwise yield a misleading ~0.5 score.
    assert not any(vectorstore.embeddings.embed_query("zzqx blorf wibble"))
    assert answer_query("zzqx blorf wibble", vectorstore=vectorstore)["status"] == "no_match"


def test_threshold_is_enforced(vectorstore, monkeypatch):
    monkeypatch.setattr(config, "DEMO_MIN_SIMILARITY_SCORE", 1.01)
    result = answer_query(MATCH_CASES[0][0], vectorstore=vectorstore)
    assert result["status"] == "no_match"


def test_relevance_scores_are_in_unit_range(vectorstore):
    results = vectorstore.similarity_search_with_relevance_scores(MATCH_CASES[0][0], k=config.TOP_K)
    assert len(results) == config.TOP_K
    scores = [s for _, s in results]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_answer_only_contains_retrieved_candidates(vectorstore):
    # Grounding check: every candidate named in the answer must come from a cited source.
    result = answer_query(MATCH_CASES[0][0], vectorstore=vectorstore)
    all_docs = vectorstore.docstore._dict.values()
    names_in_answer = {
        d.metadata["candidate_name"] for d in all_docs if d.metadata["candidate_name"] in result["answer"]
    }
    cited = {d.metadata["candidate_name"] for d in all_docs if d.metadata["source"] in result["sources"]}
    assert names_in_answer and names_in_answer <= cited


def test_missing_index_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VECTORSTORE_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="Run `python -m app.ingest`"):
        answer_query("Need a React frontend developer")
