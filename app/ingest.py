"""
Ingestion pipeline: load resumes -> split into chunks -> embed -> store in FAISS.

Run directly to (re)build the vector store:

    python -m app.ingest
"""

from __future__ import annotations  # allow modern type hints (e.g. `list[Document]`) on older Python versions

import re  # standard library: regex, used to parse "Name:" / "Title:" fields out of resume text
import sys  # standard library: used to tweak sys.path so `app` is importable when run as a script
from pathlib import Path  # standard library: filesystem paths

from langchain_community.vectorstores import FAISS  # FAISS vector store wrapper: stores embeddings + supports similarity search
from langchain_core.documents import Document  # LangChain's Document type: page_content (text) + metadata (dict)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits long text into overlapping chunks along natural boundaries

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add the project root to sys.path so `from app import ...` resolves when this file is run directly
from app import config  # noqa: E402  # project settings (paths, DEMO_MODE, model names, etc.)
from app.demo_stubs import DemoEmbeddings  # noqa: E402  # offline TF-IDF embedding stand-in, used only when DEMO_MODE=1


def _extract_field(text: str, field: str) -> str:
    # Pull a single "Field: value" line out of a resume's raw text (e.g. "Name: Arjun Mehta").
    match = re.search(rf"^{field}:\s*(.+)$", text, flags=re.MULTILINE)  # match "<field>:" at the start of any line, capture the rest of that line
    return match.group(1).strip() if match else ""  # return the captured value (trimmed), or "" if the field wasn't found


def load_resumes() -> list[Document]:
    """Load every .txt resume in data/resumes as a LangChain Document,
    tagging each with candidate_name / candidate_title metadata parsed
    from the resume header."""
    documents = []  # accumulator for one Document per resume file
    for path in sorted(config.RESUMES_DIR.glob("*.txt")):  # iterate every .txt file in data/resumes, sorted for deterministic ordering
        text = path.read_text(encoding="utf-8")  # read the full resume text from disk
        name = _extract_field(text, "Name") or path.stem  # parse the candidate's name; fall back to the filename (without extension) if missing
        title = _extract_field(text, "Title") or ""  # parse the candidate's job title; default to empty string if missing
        documents.append(
            Document(
                page_content=text,  # the full raw resume text (this is what gets chunked/embedded)
                metadata={
                    "source": path.name,          # original filename, used later to cite/dedupe results
                    "candidate_name": name,        # parsed name, shown in the UI/CLI answer
                    "candidate_title": title,      # parsed title, shown in the UI/CLI answer
                },
            )
        )
    return documents  # one Document per resume file, not yet chunked


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
