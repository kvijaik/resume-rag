"""
RAG chain: retrieval + prompt-engineered, grounded generation over resumes.

Pipeline for every user query:
  1. Guardrail: is the query long/specific enough to even attempt retrieval?
     If not -> return config.CAPABILITY_MESSAGE ("requirement is not clear...").
  2. Retrieve the top-k resumes (up to MAX_CHUNKS_PER_RESUME chunks each) from
     the FAISS vector store, scored by cosine similarity (0-1).
  3. Guardrail: keep only resumes whose best chunk scores at least
     MIN_SIMILARITY_SCORE. If none do -> return config.REJECTION_MESSAGE
     ("Sorry, I cannot help...").
  4. Generate the final answer, strictly grounded in the retrieved chunks:
       - DEMO_MODE=1 -> app.demo_stubs.demo_answer() (template-based, offline)
       - DEMO_MODE=0 -> ChatOpenAI with RESUME_MATCH_PROMPT (real LLM call)
"""

from __future__ import annotations  # allow modern type hints (e.g. `FAISS | None`) on older Python versions

import sys  # standard library: used to tweak sys.path so `app` is importable when run as a script
from pathlib import Path  # standard library: filesystem paths

from langchain_community.vectorstores import FAISS  # FAISS vector store: loads the persisted index and runs similarity search
from langchain_core.prompts import ChatPromptTemplate  # builds a reusable system+human prompt template for the chat model

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add the project root to sys.path so `from app import ...` resolves when this file is run directly
from app import config  # noqa: E402  # project settings (paths, DEMO_MODE, thresholds, messages, etc.)
from app.demo_stubs import DemoEmbeddings, demo_answer  # noqa: E402  # offline embedding + template-based answer generator, used only when DEMO_MODE=1


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------
# This system prompt is the core "prompt engineering" artifact of the project:
# it forces the model to (a) answer ONLY from the supplied resume context,
# (b) never invent candidates, skills or experience, and (c) explicitly say
# it cannot help when the context does not contain a real match, instead of
# guessing. This is what keeps the RAG pipeline from hallucinating.
SYSTEM_PROMPT = """You are a professional technical recruiting assistant.
Your job is to recommend the best-matching resume(s) for a hiring requirement,
using ONLY the resume excerpts provided in the CONTEXT below.

Rules you must always follow:
1. Ground every claim strictly in the CONTEXT. Never invent a candidate name,
   skill, employer, or years of experience that is not written in the CONTEXT.
2. If the CONTEXT does not contain a resume that reasonably matches the
   requirement, respond with exactly: "Sorry, I cannot help with this query."
   Do not guess or recommend a loosely related resume.
3. If you do find matches, for each candidate include: name, title, the
   specific skills/experience from the resume that match the requirement, and
   the source file name. Format each candidate as a markdown bullet list
   (name/title as a bold heading line, followed by "- " bullet points), not
   as a single run-on paragraph.
4. Be concise, professional, and do not repeat the entire resume text.
5. Never fabricate contact details, certifications, or companies beyond what
   is in the CONTEXT.
"""  # sent once as the "system" role message; sets the model's persona and hard rules

HUMAN_PROMPT = """CONTEXT (retrieved resume excerpts):
{context}

HIRING REQUIREMENT / QUESTION:
{question}

Respond following the rules in the system message."""  # sent as the "human" role message; {context} and {question} are filled in per-query

RESUME_MATCH_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
)  # combine both messages into a single reusable prompt template, invoked later via `RESUME_MATCH_PROMPT | llm`


# ---------------------------------------------------------------------------
# Vector store / retriever loading
# ---------------------------------------------------------------------------
def cosine_relevance(distance: float) -> float:
    """Turn a FAISS distance into a 0-1 cosine-similarity score.

    FAISS returns the squared L2 distance `d`. Both embedding backends produce
    unit-length vectors (OpenAI embeddings are normalized; TF-IDF rows are
    L2-normalized), and for unit vectors cosine similarity = 1 - d/2.
    LangChain's default conversion assumes plain (not squared) distance, which
    squashes scores and can make them negative - and LangChain then logs a
    warning containing the matched resume text.
    """
    return max(0.0, min(1.0, 1.0 - distance / 2.0))  # clamp: unrelated text can score slightly below 0


def load_vectorstore() -> FAISS:
    # Load the previously-built FAISS index from disk (see app/ingest.py for how it's built), wired to the right embedding backend.
    index_path = config.VECTORSTORE_DIR / "faiss_index"  # expected location of the persisted FAISS index directory
    if not index_path.exists():  # fail fast with a clear message if ingestion was never run
        raise RuntimeError(
            "Vector store not found. Run `python -m app.ingest` first to build it."
        )

    if config.DEMO_MODE:  # offline path: reconstruct the same TF-IDF embedder used at ingest time
        embeddings = DemoEmbeddings.load(config.VECTORSTORE_DIR / "demo_vectorizer.pkl")  # load the vectorizer fitted during ingestion, so query vectors land in the same space as the stored document vectors
    else:  # production path: real OpenAI embeddings
        from langchain_openai import OpenAIEmbeddings  # imported lazily so this dependency/API key is only required when actually used

        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME, api_key=config.OPENAI_API_KEY
        )

    return FAISS.load_local(
        str(index_path),
        embeddings,                                    # embedding backend used to embed future queries (must match what built the index)
        allow_dangerous_deserialization=True,          # required by langchain_community FAISS to unpickle the locally-saved index (safe here: we trust our own vectorstore/ dir)
        relevance_score_fn=cosine_relevance,           # same 0-1 cosine score for both backends
    )


def retrieve_resumes(vs: FAISS, query: str, top_k: int | None = None) -> list:
    """Return (chunk, score) pairs for the `top_k` best-matching resumes, best first.

    Searching returns chunks, and one resume with several strong chunks could
    otherwise fill every slot and hide other candidates. So we fetch a wider
    pool of chunks and keep at most MAX_CHUNKS_PER_RESUME per resume. Each
    resume's first pair is its best-scoring chunk.
    """
    top_k = top_k or config.TOP_K
    pool = vs.similarity_search_with_relevance_scores(query, k=top_k * 10)  # best-first; wide enough to reach top_k distinct resumes
    per_resume: dict[str, list] = {}  # source file -> its kept chunks, in first-seen (= best-score) order
    for doc, score in pool:
        source = doc.metadata.get("source", "unknown")
        if source not in per_resume:
            if len(per_resume) == top_k:  # already have enough candidates; ignore new resumes
                continue
            per_resume[source] = []
        if len(per_resume[source]) < config.MAX_CHUNKS_PER_RESUME:
            per_resume[source].append((doc, score))
    return [hit for hits in per_resume.values() for hit in hits]


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
def is_query_too_vague(query: str) -> bool:
    """Very short / empty / generic queries are treated as unclear rather
    than attempting (and likely botching) a retrieval."""
    words = [w for w in query.strip().split() if w]  # split on whitespace into non-empty word tokens
    if len(words) < config.MIN_QUERY_WORDS:  # too few words to describe a real hiring requirement
        return True
    generic_only = {"hi", "hello", "hey", "help", "resume", "resumes", "test", "?"}  # words that carry no role/skill information on their own
    normalized = {w.lower().strip("?.! ") for w in words} - {""}  # normalize each word (lowercase, strip punctuation); drop tokens that were only punctuation (e.g. "?")
    if not normalized or normalized <= generic_only:  # nothing substantive left, or EVERY word is in the generic set
        return True  # the whole query is made up of only punctuation or generic/greeting words -> treat as unclear
    return False  # query has enough substantive words to attempt retrieval


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def answer_query(query: str, vectorstore: FAISS | None = None) -> dict:
    """Run the full guardrail -> retrieve -> generate pipeline for a query.

    Returns a dict with:
      - answer: str (final text response)
      - status: "unclear" | "no_match" | "matched"
      - sources: list of matched source filenames (empty unless matched)
    """
    query = (query or "").strip()  # normalize None/whitespace-only input to a clean string

    if is_query_too_vague(query):  # guardrail #1: reject queries that don't say enough to search on
        return {"answer": config.CAPABILITY_MESSAGE, "status": "unclear", "sources": []}

    vs = vectorstore or load_vectorstore()  # reuse a passed-in (e.g. cached-by-Streamlit) vectorstore, or load one fresh

    if config.DEMO_MODE:
        # The demo relevance score (1 - L2_distance/2) assumes both vectors
        # are unit-normalized, which only holds when the query shares at
        # least one vocabulary term with the resume corpus. TfidfVectorizer
        # returns an all-zero vector for a query with zero overlap, and the
        # formula degenerates to a constant ~0.5 "confidence" for that case -
        # a false positive for genuinely unrelated queries (e.g. "what is
        # the weather"). Catch it explicitly before it reaches similarity
        # search.
        import numpy as np  # local import: numpy is only needed for this demo-mode zero-vector check

        query_vector = np.array(vs.embeddings.embed_query(query))  # embed the query with the same TF-IDF vectorizer used to build the index
        if not np.any(query_vector):  # True only if every component is exactly 0 (no vocabulary overlap at all)
            return {"answer": config.REJECTION_MESSAGE, "status": "no_match", "sources": []}  # short-circuit: refuse instead of trusting an undefined similarity score

    hits = retrieve_resumes(vs, query)  # top-k resumes, up to MAX_CHUNKS_PER_RESUME chunks each, best first

    threshold = config.DEMO_MIN_SIMILARITY_SCORE if config.DEMO_MODE else config.MIN_SIMILARITY_SCORE  # pick the threshold matching the active embedding backend's score range
    best_score: dict[str, float] = {}  # each resume's best chunk score (its first hit)
    for doc, score in hits:
        best_score.setdefault(doc.metadata.get("source", "unknown"), score)
    results = [(doc, score) for doc, score in hits if best_score[doc.metadata.get("source", "unknown")] >= threshold]  # guardrail #2: drop resumes that don't match well enough
    if not results:  # no resume scored above the confidence threshold
        return {"answer": config.REJECTION_MESSAGE, "status": "no_match", "sources": []}

    sources = sorted({doc.metadata.get("source", "unknown") for doc, _ in results})  # unique, sorted list of resume filenames behind the retrieved chunks

    if config.DEMO_MODE:  # offline path: build the answer with plain string templating, no LLM call
        answer = demo_answer(query, results)
        return {"answer": answer, "status": "matched", "sources": sources}

    # --- Real OpenAI generation path ---
    from langchain_openai import ChatOpenAI  # imported lazily so this dependency/API key is only required when actually used

    llm = ChatOpenAI(
        model=config.CHAT_MODEL_NAME,          # which OpenAI chat model to call
        temperature=config.LLM_TEMPERATURE,    # sampling temperature (0 = deterministic)
        api_key=config.OPENAI_API_KEY,          # credentials for the OpenAI API
    )
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source')} | Candidate: {doc.metadata.get('candidate_name')}"
        f"{' — ' + doc.metadata['candidate_title'] if doc.metadata.get('candidate_title') else ''}]\n{doc.page_content}"
        for doc, _ in results
    )  # concatenate the retrieved chunks into one context block, each labeled with its source file and candidate (later chunks of a resume don't repeat the name), separated by a divider
    chain = RESUME_MATCH_PROMPT | llm  # LangChain Expression Language: pipe the filled-in prompt straight into the chat model
    response = chain.invoke({"context": context, "question": query})  # fill {context}/{question} into the prompt and call the model
    answer_text = response.content  # extract the plain-text answer from the model's response object

    if answer_text.strip().startswith("Sorry, I cannot help"):  # the model itself decided (per SYSTEM_PROMPT rule 2) that nothing in the context matches
        return {"answer": answer_text, "status": "no_match", "sources": []}  # report as no_match and drop sources, since the model didn't actually use them

    cited = [s for s in sources if s in answer_text]  # the prompt requires a source file per candidate; keep only resumes the answer actually recommends
    return {"answer": answer_text, "status": "matched", "sources": cited or sources}  # fall back to all retrieved resumes if the model cited none by file name
