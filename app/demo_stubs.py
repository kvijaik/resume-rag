"""
Offline stand-ins used ONLY when DEMO_MODE=1.

Why this file exists
---------------------
This project is designed to run against the real OpenAI API (embeddings +
chat model). To let the app be demonstrated / screenshotted without an
OpenAI API key (e.g. for grading or offline review), this module provides:

1. `DemoEmbeddings` - a deterministic, local TF-IDF based embedding class
   that implements the same interface LangChain expects (`embed_documents`,
   `embed_query`), so the FAISS vector store and similarity search work
   exactly the same way structurally as they would with OpenAI embeddings.

2. `demo_answer()` - a template-based "generation" step that mimics the
   grounded, non-hallucinating answer format the real prompt (see
   app/rag_chain.py -> RESUME_MATCH_PROMPT) asks the OpenAI chat model to
   produce. It only ever uses text taken directly from the retrieved resume
   chunks - it does not invent any information.

None of this is used when DEMO_MODE=0 (the default). In that case the app
uses `langchain_openai.OpenAIEmbeddings` and `langchain_openai.ChatOpenAI`
directly.
"""

from __future__ import annotations  # allow modern type hints (e.g. `list[float]`) to work on older Python versions

import pickle  # standard library: (de)serialize the fitted TfidfVectorizer to/from disk
from pathlib import Path  # standard library: typed filesystem paths for save()/load()
from typing import List  # typing: annotate list return types

from langchain_core.embeddings import Embeddings  # LangChain's abstract base class that any embedding provider must implement
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer  # scikit-learn: TF-IDF vectorizer + its built-in English stop-word list

# Generic resume/boilerplate words that appear in almost every resume
# regardless of technology (headers, filler verbs, etc.). Excluding these
# from the TF-IDF vocabulary keeps the local demo similarity score focused
# on genuinely distinguishing, technology-specific terms (e.g. "spring",
# "kubernetes", "kafka") instead of words every resume shares.
_RESUME_BOILERPLATE_STOP_WORDS = {
    "resume", "name", "title", "location", "email", "phone", "summary",
    "professional", "experience", "experienced", "years", "skills",
    "technical", "education", "certifications", "strong", "background",
    "designed", "built", "developed", "implemented", "managed", "led",
    "using", "including", "responsible", "worked", "role", "team",
}  # custom stop-word set specific to this resume corpus's boilerplate
_STOP_WORDS = list(ENGLISH_STOP_WORDS.union(_RESUME_BOILERPLATE_STOP_WORDS))  # final stop-word list = sklearn's generic English words + our resume-specific ones


class DemoEmbeddings(Embeddings):
    """A local, deterministic embedding function based on TF-IDF.

    Subclasses LangChain's `Embeddings` base class so it is a proper,
    drop-in replacement for `OpenAIEmbeddings` (same `embed_documents` /
    `embed_query` interface) when no API key is available.
    """

    def __init__(self, vectorizer: TfidfVectorizer):
        self.vectorizer = vectorizer  # store the already-fitted (or loaded) scikit-learn vectorizer instance

    @classmethod
    def fit(cls, texts: List[str]) -> "DemoEmbeddings":
        # Build a fresh vectorizer and learn its vocabulary/IDF weights from the given corpus (called once, during ingestion).
        vectorizer = TfidfVectorizer(
            stop_words=_STOP_WORDS,       # drop generic English + resume-boilerplate words from the vocabulary
            max_features=4096,            # cap vocabulary size to the 4096 highest-frequency remaining terms
            ngram_range=(1, 2),           # index both single words and two-word phrases (e.g. "spring boot")
            sublinear_tf=True,            # use 1 + log(tf) instead of raw term frequency, so very repeated words don't dominate
        )
        vectorizer.fit(texts)  # learn vocabulary + IDF weights from the resume chunk texts (no transform yet)
        return cls(vectorizer)  # wrap the fitted vectorizer in a DemoEmbeddings instance and return it

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:  # open the target path in binary-write mode
            pickle.dump(self.vectorizer, f)  # serialize the fitted vectorizer (vocabulary + IDF weights) to disk

    @classmethod
    def load(cls, path: Path) -> "DemoEmbeddings":
        with open(path, "rb") as f:  # open the saved vectorizer file in binary-read mode
            vectorizer = pickle.load(f)  # deserialize the previously-fitted vectorizer
        return cls(vectorizer)  # wrap it back into a DemoEmbeddings instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # LangChain interface method: embed a batch of documents (used when building the FAISS index).
        matrix = self.vectorizer.transform(texts)  # transform texts into a sparse TF-IDF matrix using the already-learned vocabulary/IDF
        return matrix.toarray().tolist()  # convert sparse matrix -> dense NumPy array -> plain Python list of lists (one vector per text)

    def embed_query(self, text: str) -> List[float]:
        # LangChain interface method: embed a single query string at search time.
        vector = self.vectorizer.transform([text])  # transform the single query into the same TF-IDF vector space as the documents
        return vector.toarray()[0].tolist()  # dense-ify and unwrap to a single flat vector (list of floats)


def demo_answer(query: str, docs_with_scores) -> str:
    """Build a grounded answer purely from retrieved resume chunks.

    This mirrors the structure the real OpenAI prompt (RESUME_MATCH_PROMPT)
    asks the model to follow, so the demo output looks like what the live
    OpenAI-backed app would produce - but every word here is generated with
    plain Python string formatting from the retrieved text, not an LLM.
    """
    lines = [f"Based on the available resumes, here are the matches for: \"{query}\"\n"]  # opening line of the answer, echoes the user's query back

    seen_sources = set()  # track which source resume files have already been included, to avoid listing the same resume twice
    rank = 1  # 1-based ranking counter shown to the user (1., 2., 3., ...)
    for doc, score in docs_with_scores:  # iterate retrieved (Document, relevance_score) pairs in the order FAISS ranked them
        source = doc.metadata.get("source", "unknown")  # resume filename this chunk came from (set during ingestion, app/ingest.py)
        if source in seen_sources:  # a resume can contribute multiple chunks; only show each resume once
            continue  # skip duplicate chunks from a resume we've already listed
        seen_sources.add(source)  # mark this resume's source file as included

        name = doc.metadata.get("candidate_name", "Unknown Candidate")  # candidate name parsed from the resume header at ingest time
        title = doc.metadata.get("candidate_title", "")  # candidate title parsed from the resume header at ingest time
        snippet = " ".join(doc.page_content.split())[:320]  # collapse whitespace/newlines to single spaces, then truncate to 320 chars for a readable excerpt

        lines.append(f"**{rank}. {name} — {title}**")  # bold markdown heading line: rank, candidate name, and title
        lines.append(f"- Match confidence: {score:.2f}")  # relevance score for this match, formatted to 2 decimal places
        lines.append(f"- Relevant excerpt: \"{snippet}...\"")  # the actual resume text grounding this recommendation
        lines.append(f"- Source file: `{source}`")  # which file in data/resumes/ this came from, for traceability
        lines.append("")  # blank line to separate this candidate block from the next
        rank += 1  # advance the rank counter for the next unique resume

    lines.append(
        "*Note: This recommendation is grounded strictly in the resume excerpts "
        "retrieved above; no information was invented beyond what is written in "
        "these resumes.*"
    )  # closing disclaimer reinforcing the no-hallucination guarantee, mirroring the real prompt's rules
    return "\n".join(lines)  # join all lines into the final multi-line markdown answer string
