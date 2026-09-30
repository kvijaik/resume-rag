"""
Configuration for the Resume RAG application.

All settings are read from environment variables (see .env.example).
DEMO_MODE lets the whole pipeline run end-to-end without a real OpenAI API key,
by swapping in local, deterministic stand-ins for the embedding model and the
chat model (see app/demo_stubs.py). This is ONLY meant for offline demos /
screenshots. Set DEMO_MODE=0 (default) and provide OPENAI_API_KEY to run the
application against the real OpenAI API.
"""

import os  # standard library: read environment variables (os.getenv)
from pathlib import Path  # standard library: filesystem paths, used to build project directory constants

from dotenv import load_dotenv  # python-dotenv: loads key=value pairs from a .env file into the environment

load_dotenv()  # populate os.environ from a .env file (if present) before we read any settings below

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root: two levels up from this file (app/config.py -> app/ -> project root)
RESUMES_DIR = BASE_DIR / "data" / "resumes"  # folder containing the raw *.txt resume files ingested by app/ingest.py
VECTORSTORE_DIR = BASE_DIR / "vectorstore"  # folder where the built FAISS index (and demo TF-IDF vectorizer) are persisted

# --- OpenAI / LLM settings ---------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # API key for the real OpenAI-backed embeddings/chat model; blank string if unset
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "gpt-4o-mini")  # which OpenAI chat model to call for answer generation (non-demo mode)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")  # which OpenAI embedding model to use for building/querying the vector store (non-demo mode)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))  # sampling temperature for the chat model; 0 = deterministic, no creative drift

# --- Demo mode ------------------------------------------------------------
# DEMO_MODE=1 -> use local stub embeddings + stub chat model (no network/API key
# required). Used only to generate the sample screenshots in this project's
# documentation when no OpenAI API key is available. Set to 0 (or unset,
# together with a real OPENAI_API_KEY) to use the real OpenAI backend.
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"  # bool flag: True only if the env var is literally the string "1"

# --- Retrieval / guardrail settings -------------------------------------
# Number of resume chunks to retrieve per query.
TOP_K = int(os.getenv("TOP_K", "3"))  # how many top-scoring chunks the retriever returns per query

# Minimum similarity score (0-1, higher = more similar) required before we
# trust a retrieved chunk enough to answer from it. Below this threshold we
# treat the query as having "no matching resume" and return the refusal
# response instead of guessing / hallucinating.
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.35"))  # threshold used only in the real OpenAI-embedding path

# The local TF-IDF DemoEmbeddings (see app/demo_stubs.py) produce cosine
# similarities in a much lower, "compressed" numeric range than OpenAI's
# embedding model does on this kind of short-query / long-document text, so
# demo mode uses its own (lower) threshold. Tune independently if needed.
DEMO_MIN_SIMILARITY_SCORE = float(os.getenv("DEMO_MIN_SIMILARITY_SCORE", "0.12"))  # threshold used only when DEMO_MODE is on

# Minimum number of words a query must have for us to attempt retrieval at
# all; very short / vague queries are treated as "unclear".
MIN_QUERY_WORDS = int(os.getenv("MIN_QUERY_WORDS", "3"))  # guardrail #1 input: queries shorter than this are rejected as too vague

# Message shown when the query is too short/generic to even attempt retrieval
# (see app/rag_chain.py -> is_query_too_vague). Lists what the assistant can do.
CAPABILITY_MESSAGE = (
    "The requirement is not clear. This assistant can search and recommend resumes "
    "for the following technology profiles: Java Backend Developer (Spring Boot), "
    "React UI / Frontend Developer, Full Stack Web Developer (Python), "
    "Network Security Engineer (AWS), Application Security Engineer, "
    "DevOps / Cloud Engineer (AWS), Data Engineer, and Machine Learning Engineer. "
    "Please describe the role, technology stack, or skills you are hiring for, "
    "for example: 'Need a Java full stack developer with Spring Boot and React' "
    "or 'Looking for an AWS network security engineer'."
)

# Message shown when retrieval ran but nothing scored above the similarity
# threshold (i.e. no resume in the knowledge base actually matches the ask).
REJECTION_MESSAGE = (
    "Sorry, I cannot help with this query. It falls outside the scope of the "
    "resumes available in this system, so I won't guess or fabricate a match. "
    "Please rephrase your requirement in terms of a technology role I support."
)
