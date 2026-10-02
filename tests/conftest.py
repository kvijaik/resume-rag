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


def make_pdf(lines):
    """Bytes of a minimal one-page PDF with the given text lines (no lines = no text layer, like a scan)."""
    esc = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
    content = "BT /F1 11 Tf 72 720 Td 14 TL " + " ".join(f"({line}) Tj T*" for line in esc) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return out


def make_docx(lines):
    """Bytes of a minimal Word document with one paragraph per line."""
    import io
    import zipfile
    from xml.sax.saxutils import escape

    paragraphs = "".join(f"<w:p><w:r><w:t>{escape(line)}</w:t></w:r></w:p>" for line in lines)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paragraphs}</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)
    return buffer.getvalue()


GOLANG_RESUME = [
    "Name: Meera Pillai",
    "Title: Golang Backend Developer",
    "PROFESSIONAL SUMMARY",
    "Golang developer building gRPC microservices with goroutines, channels and protobuf.",
    "TECHNICAL SKILLS",
    "- Golang, gRPC, protobuf, goroutines, Gin, CockroachDB",
]


@pytest.fixture
def library(tmp_path, monkeypatch):
    """An isolated copy of the resume library plus its own demo index, safe to add files to."""
    import shutil

    resumes = tmp_path / "resumes"
    shutil.copytree(config.RESUMES_DIR, resumes)
    monkeypatch.setattr(config, "RESUMES_DIR", resumes)
    monkeypatch.setattr(config, "VECTORSTORE_DIR", tmp_path / "vectorstore")
    monkeypatch.setattr(config, "DEMO_MODE", True)
    ingest.build_vectorstore()
    return resumes
