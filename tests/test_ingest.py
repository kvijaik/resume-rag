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


import pytest  # noqa: E402

from app.ingest import _guess_name_and_title  # noqa: E402


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Darshan Raj K P\nJava Full Stack Developer (Junior)\n+91 98 | a@b.com", ("Darshan Raj K P", "Java Full Stack Developer (Junior)")),
        ("RESUME\n\nPRIYA SHARMA\nFrontend Engineer", ("Priya Sharma", "Frontend Engineer")),
        ("Curriculum Vitae:\nJohn O'Neil\njohn@example.com | +1 555 0100", ("John O'Neil", "")),
        ("Jane Doe\nSenior Engineer at Acme since 2019", ("Jane Doe", "")),
        ("Jane Doe", ("Jane Doe", "")),
        ("jane.doe@example.com | +1 555 0100\nJane Doe", ("", "")),
        ("Madonna\nSinger", ("", "")),
        ("", ("", "")),
    ],
)
def test_guess_name_and_title_from_opening_lines(text, expected):
    assert _guess_name_and_title(text) == expected


def test_real_style_resume_gets_name_from_first_line(tmp_path, monkeypatch):
    (tmp_path / "cv_final_v2.txt").write_text("ANANYA IYER\nSecurity Engineer\nSkills: OWASP", encoding="utf-8")
    monkeypatch.setattr(config, "RESUMES_DIR", tmp_path)
    [doc] = load_resumes()
    assert doc.metadata["candidate_name"] == "Ananya Iyer"
    assert doc.metadata["candidate_title"] == "Security Engineer"
