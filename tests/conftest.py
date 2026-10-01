"""Shared pytest fixtures.

Every test runs fully offline in DEMO_MODE against a FAISS index built into a
temporary directory, so the suite needs no API key and never touches the
committed vectorstore/ folder.
"""

import os
import sys
from pathlib import Path

import pytest

# DEMO_MODE must be set before app.config is imported, because config reads
# environment variables at import time.
os.environ["DEMO_MODE"] = "1"
os.environ.pop("OPENAI_API_KEY", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app import ingest, rag_chain  # noqa: E402


@pytest.fixture(scope="session")
def built_index_dir(tmp_path_factory):
    """Build the demo FAISS index once per test session in a temp folder."""
    out_dir = tmp_path_factory.mktemp("vectorstore")
    original = config.VECTORSTORE_DIR
    config.VECTORSTORE_DIR = out_dir
    try:
        ingest.build_vectorstore()
    finally:
        config.VECTORSTORE_DIR = original
    return out_dir


@pytest.fixture
def vectorstore(built_index_dir, monkeypatch):
    """A loaded FAISS store backed by the temp index."""
    monkeypatch.setattr(config, "VECTORSTORE_DIR", built_index_dir)
    monkeypatch.setattr(config, "DEMO_MODE", True)
    return rag_chain.load_vectorstore()
