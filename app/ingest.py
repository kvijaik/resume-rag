"""
Ingestion pipeline: load resumes -> split into chunks -> embed -> store in FAISS.

Supported resume formats: .txt, .pdf (text-based, not scanned) and .docx.

Run directly to (re)build the vector store:

    python -m app.ingest
"""

from __future__ import annotations  # allow modern type hints (e.g. `list[Document]`) on older Python versions

import re  # standard library: regex, used to parse "Name:" / "Title:" fields out of resume text
import sys  # standard library: used to tweak sys.path so `app` is importable when run as a script
from pathlib import Path  # standard library: filesystem paths

import docx2txt  # extracts plain text from Word (.docx) files
from langchain_community.vectorstores import FAISS  # FAISS vector store wrapper: stores embeddings + supports similarity search
from langchain_core.documents import Document  # LangChain's Document type: page_content (text) + metadata (dict)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits long text into overlapping chunks along natural boundaries
from pypdf import PdfReader  # extracts the text layer from PDF files

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add the project root to sys.path so `from app import ...` resolves when this file is run directly
from app import config  # noqa: E402  # project settings (paths, DEMO_MODE, model names, etc.)
from app.demo_stubs import DemoEmbeddings  # noqa: E402  # offline TF-IDF embedding stand-in, used only when DEMO_MODE=1


def _extract_field(text: str, field: str) -> str:
    # Pull a single "Field: value" line out of a resume's raw text (e.g. "Name: Arjun Mehta").
    match = re.search(rf"^[ \t]*{field}:\s*(.+)$", text, flags=re.MULTILINE)  # match "<field>:" at the start of any line (PDF/Word text may indent it), capture the rest of that line
    return match.group(1).strip() if match else ""  # return the captured value (trimmed), or "" if the field wasn't found


SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx")  # resume file types the app can read and accept as uploads


def read_resume_text(path: Path) -> str:
    """Return the plain text of a .txt, .pdf or .docx resume ("" if it has no extractable text)."""
    suffix = path.suffix.lower()  # compare extensions case-insensitively (e.g. "CV.PDF")
    if suffix == ".pdf":
        pages = PdfReader(str(path)).pages  # one entry per PDF page
        return "\n".join(page.extract_text() or "" for page in pages).strip()  # scanned (image-only) pages yield no text
    if suffix == ".docx":
        return (docx2txt.process(str(path)) or "").strip()  # paragraphs and table cells as plain text
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace").strip()  # tolerate stray non-UTF-8 bytes instead of failing
    raise ValueError(f"Unsupported file type '{path.suffix}'. Use one of: {', '.join(SUPPORTED_EXTENSIONS)}")


def load_resumes() -> list[Document]:
    """Load every supported resume in data/resumes as a LangChain Document,
    tagging each with candidate_name / candidate_title metadata parsed
    from the resume header."""
    documents = []  # accumulator for one Document per resume file
    paths = sorted(
        p for p in config.RESUMES_DIR.glob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith(".")  # skip hidden files, e.g. an in-progress upload
    )  # every supported file in data/resumes, sorted for deterministic ordering
    for path in paths:
        text = read_resume_text(path)  # extract the full resume text, whatever the file format
        if not text:  # e.g. a scanned PDF with no text layer: nothing to search, so skip it rather than index an empty document
            print(f"Skipping {path.name}: no extractable text.")
            continue
        name = _extract_field(text, "Name") or path.stem  # parse the candidate's name; fall back to the filename (without extension) if missing
        title = _extract_field(text, "Title") or ""  # parse the candidate's job title; default to empty string if missing
        documents.append(
            Document(
                page_content=text,  # the full resume text (this is what gets chunked/embedded)
                metadata={
                    "source": path.name,          # original filename, used later to cite/dedupe results
                    "candidate_name": name,        # parsed name, shown in the UI/CLI answer
                    "candidate_title": title,      # parsed title, shown in the UI/CLI answer
                },
            )
        )
    return documents  # one Document per resume file, not yet chunked


def save_uploaded_resume(filename: str, data: bytes) -> Path:
    """Validate an uploaded resume and save it into data/resumes.

    Raises ValueError for an unsupported type or a file with no readable text
    (nothing is left on disk in that case). Re-uploading a file with the same
    name replaces the earlier version. Call build_vectorstore() afterwards so
    the new resume becomes searchable.
    """
    safe_name = Path(filename).name  # drop any directory parts, so an upload can't write outside data/resumes (e.g. "../../x.txt")
    if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"{safe_name}: unsupported file type. Use one of: {', '.join(SUPPORTED_EXTENSIONS)}")

    config.RESUMES_DIR.mkdir(parents=True, exist_ok=True)  # make sure the resume folder exists
    target = config.RESUMES_DIR / safe_name
    tmp = target.with_name(f".upload-{safe_name}")  # write to a hidden temp file first, so a bad upload never replaces a good resume
    tmp.write_bytes(data)
    try:
        text = read_resume_text(tmp)  # prove the file is readable before accepting it
    except Exception as exc:  # corrupt or mislabelled files make the PDF/Word readers raise various errors
        tmp.unlink()
        raise ValueError(f"{safe_name}: could not read the file ({exc}).") from exc
    if not text:
        tmp.unlink()
        raise ValueError(f"{safe_name}: no readable text found (scanned PDFs are not supported).")
    tmp.replace(target)  # accept: move into place (overwrites an earlier upload with the same name)
    return target


def split_documents(documents: list[Document]) -> list[Document]:
    # Break each full-resume Document into smaller overlapping chunks so retrieval can return focused excerpts instead of whole resumes.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,                                  # target max characters per chunk
        chunk_overlap=100,                                # characters shared between consecutive chunks, to avoid cutting context at a hard boundary
        separators=["\n\n", "\n", ". ", " "],             # preferred split points, tried in order: paragraph, line, sentence, then word
    )
    return splitter.split_documents(documents)  # returns new Document objects (chunks), each inheriting the parent resume's metadata


def build_vectorstore() -> None:
    # Top-level ingestion entry point: (re)builds the FAISS index from scratch and writes it to disk.
    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)  # ensure the output directory exists (create parents, don't error if it already exists)

    documents = load_resumes()  # load all resumes as whole-file Documents
    if not documents:  # guard against an empty/misconfigured resumes folder
        raise RuntimeError(f"No resumes found in {config.RESUMES_DIR}")
    chunks = split_documents(documents)  # split each resume into smaller overlapping chunks for retrieval
    print(f"Loaded {len(documents)} resumes -> split into {len(chunks)} chunks.")  # progress output for the CLI user

    relevance_score_fn = None  # optional custom function FAISS uses to convert raw distance -> a 0-1 relevance score; None = use FAISS's default
    if config.DEMO_MODE:  # offline path: no OpenAI API key available/used
        print("DEMO_MODE=1 -> fitting local TF-IDF DemoEmbeddings (no API key needed).")
        embeddings = DemoEmbeddings.fit([c.page_content for c in chunks])  # fit a fresh TF-IDF vectorizer on the chunk texts (learns vocabulary once, here)
        embeddings.save(config.VECTORSTORE_DIR / "demo_vectorizer.pkl")  # persist the fitted vectorizer so query time uses the exact same vocabulary/IDF
        # TF-IDF vectors are L2-normalized, so squared L2 distance `d` relates
        # to cosine similarity by cos_sim = 1 - d/2. Use that as the 0-1
        # relevance score instead of FAISS's default (which assumes
        # OpenAI-style embedding magnitudes and produces out-of-range values
        # for TF-IDF vectors).
        relevance_score_fn = lambda distance: max(0.0, min(1.0, 1.0 - distance / 2.0))  # noqa: E731  # clamp the cosine-derived score into [0, 1]
    else:  # production path: real OpenAI embeddings
        from langchain_openai import OpenAIEmbeddings  # imported lazily so this dependency/API key is only needed when actually used

        print(f"Using OpenAI embeddings model: {config.EMBEDDING_MODEL_NAME}")
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,   # which OpenAI embedding model to call
            api_key=config.OPENAI_API_KEY,        # credentials for the OpenAI API
        )

    vectorstore = FAISS.from_documents(
        chunks, embeddings, relevance_score_fn=relevance_score_fn
    )  # embed every chunk and build an in-memory FAISS similarity index over them
    vectorstore.save_local(str(config.VECTORSTORE_DIR / "faiss_index"))  # persist the FAISS index (and its docstore) to disk for later loading
    print(f"FAISS index saved to {config.VECTORSTORE_DIR / 'faiss_index'}")  # confirm completion to the CLI user


if __name__ == "__main__":  # only run ingestion when this file is executed directly (`python -m app.ingest`), not on import
    build_vectorstore()
