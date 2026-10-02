"""Reading .txt / .pdf / .docx resumes and accepting uploads."""

import pytest

from app import config, ingest
from app.rag_chain import answer_query
from tests.conftest import GOLANG_RESUME, make_docx, make_pdf

HEADER = ["Name: Jane Doe", "Title: Senior Golang Developer", "Skills: Go, gRPC"]


@pytest.mark.parametrize(
    "filename, data",
    [
        ("jane.pdf", make_pdf(HEADER)),
        ("jane.docx", make_docx(HEADER)),
        ("jane.txt", "\n".join(HEADER).encode()),
        ("JANE.PDF", make_pdf(HEADER)),
    ],
)
def test_each_format_is_read_with_name_and_title(tmp_path, monkeypatch, filename, data):
    monkeypatch.setattr(config, "RESUMES_DIR", tmp_path)
    (tmp_path / filename).write_bytes(data)
    [doc] = ingest.load_resumes()
    assert doc.metadata == {
        "source": filename,
        "candidate_name": "Jane Doe",
        "candidate_title": "Senior Golang Developer",
    }
    assert "gRPC" in doc.page_content


def test_unsupported_hidden_and_textless_files_are_skipped(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "RESUMES_DIR", tmp_path)
    (tmp_path / "notes.md").write_text("Name: Not A Resume")
    (tmp_path / ".upload-half.pdf").write_bytes(make_pdf(HEADER))
    (tmp_path / "scanned.pdf").write_bytes(make_pdf([]))
    (tmp_path / "real.docx").write_bytes(make_docx(HEADER))
    assert [d.metadata["source"] for d in ingest.load_resumes()] == ["real.docx"]
    assert "Skipping scanned.pdf: no extractable text." in capsys.readouterr().out


def test_read_resume_text_rejects_unknown_type(tmp_path):
    path = tmp_path / "cv.rtf"
    path.write_text("x")
    with pytest.raises(ValueError, match="Unsupported file type"):
        ingest.read_resume_text(path)


def test_upload_is_saved_into_the_library(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESUMES_DIR", tmp_path / "new_folder")
    saved = ingest.save_uploaded_resume("jane.docx", make_docx(HEADER))
    assert saved == tmp_path / "new_folder" / "jane.docx"
    assert ingest.read_resume_text(saved).startswith("Name: Jane Doe")


def test_upload_cannot_escape_the_library_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESUMES_DIR", tmp_path / "resumes")
    saved = ingest.save_uploaded_resume("../../evil.txt", b"Name: Eve\nTitle: Hacker")
    assert saved == tmp_path / "resumes" / "evil.txt"
    assert not (tmp_path / "evil.txt").exists()


def test_reupload_replaces_the_earlier_version(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESUMES_DIR", tmp_path)
    ingest.save_uploaded_resume("jane.txt", b"Name: Jane Doe\nTitle: Old Title")
    ingest.save_uploaded_resume("jane.txt", b"Name: Jane Doe\nTitle: New Title")
    [doc] = ingest.load_resumes()
    assert doc.metadata["candidate_title"] == "New Title"


@pytest.mark.parametrize(
    "filename, data, error",
    [
        ("cv.rtf", b"Name: Jane", "unsupported file type"),
        ("scan.pdf", make_pdf([]), "no readable text"),
        ("empty.txt", b"   \n", "no readable text"),
        ("broken.pdf", b"this is not really a pdf", "could not read the file"),
        ("broken.docx", b"this is not really a docx", "could not read the file"),
    ],
)
def test_bad_uploads_are_rejected_and_leave_nothing_behind(tmp_path, monkeypatch, filename, data, error):
    monkeypatch.setattr(config, "RESUMES_DIR", tmp_path)
    (tmp_path / "keep.txt").write_text("Name: Existing\nTitle: Kept")
    with pytest.raises(ValueError, match=error):
        ingest.save_uploaded_resume(filename, data)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]


@pytest.mark.parametrize("filename, make", [("meera.pdf", make_pdf), ("meera.docx", make_docx)])
def test_uploaded_resume_becomes_searchable(library, filename, make):
    ingest.save_uploaded_resume(filename, make(GOLANG_RESUME))
    ingest.build_vectorstore()
    result = answer_query("Need a Golang developer with gRPC and protobuf")
    assert result["status"] == "matched"
    assert filename in result["sources"]
    assert "Meera Pillai" in result["answer"]
