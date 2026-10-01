"""Ingestion: resume loading, metadata parsing, and chunking."""

from app import config
from app.ingest import _extract_field, load_resumes, split_documents


def test_extract_field_reads_header_line():
    text = "RESUME\n\nName: Jane Doe\nTitle: Senior Golang Developer\n"
    assert _extract_field(text, "Name") == "Jane Doe"
    assert _extract_field(text, "Title") == "Senior Golang Developer"


def test_extract_field_missing_returns_empty():
    assert _extract_field("no header here", "Name") == ""


def test_extract_field_only_matches_line_start():
    assert _extract_field("Summary mentions Name: Not A Header", "Name") == ""


def test_every_resume_is_loaded_with_metadata():
    docs = load_resumes()
    files = sorted(p.name for p in config.RESUMES_DIR.glob("*.txt"))
    assert [d.metadata["source"] for d in docs] == files
    for d in docs:
        assert d.metadata["candidate_name"], d.metadata["source"]
        assert d.metadata["candidate_title"], d.metadata["source"]


def test_name_falls_back_to_file_stem(tmp_path, monkeypatch):
    (tmp_path / "no_header.txt").write_text("Just some text without a header.", encoding="utf-8")
    monkeypatch.setattr(config, "RESUMES_DIR", tmp_path)
    [doc] = load_resumes()
    assert doc.metadata["candidate_name"] == "no_header"
    assert doc.metadata["candidate_title"] == ""


def test_chunks_respect_size_and_keep_metadata():
    docs = load_resumes()
    chunks = split_documents(docs)
    assert len(chunks) >= len(docs)
    for c in chunks:
        assert len(c.page_content) <= 800
        assert {"source", "candidate_name", "candidate_title"} <= c.metadata.keys()
    # Every resume contributes at least one chunk.
    assert {c.metadata["source"] for c in chunks} == {d.metadata["source"] for d in docs}
