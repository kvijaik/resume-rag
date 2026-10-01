"""`python -m app.ingest` / `python -m app.cli` entry points, plus ingest edge cases."""

import runpy
import sys

import pytest

from app import config, ingest

# runpy re-executes a module that is already imported; that is intended here.
pytestmark = pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")


def test_ingest_module_main_builds_index(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "VECTORSTORE_DIR", tmp_path)
    monkeypatch.setattr(config, "DEMO_MODE", True)
    runpy.run_module("app.ingest", run_name="__main__")
    assert (tmp_path / "faiss_index" / "index.faiss").exists()
    assert (tmp_path / "demo_vectorizer.pkl").exists()
    assert "FAISS index saved" in capsys.readouterr().out


def test_cli_module_main_answers_query(vectorstore, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cli", "Need", "a", "React", "frontend", "developer"])
    runpy.run_module("app.cli", run_name="__main__")
    assert "Status: matched" in capsys.readouterr().out


def test_ingest_with_no_resumes_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESUMES_DIR", tmp_path / "empty")
    monkeypatch.setattr(config, "VECTORSTORE_DIR", tmp_path / "out")
    with pytest.raises(RuntimeError, match="No resumes found"):
        ingest.build_vectorstore()
