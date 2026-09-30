# Resume RAG Assistant

A learning project demonstrating **Retrieval-Augmented Generation (RAG)** and
**Prompt Engineering** with **LangChain** and **Python**. The app answers
natural-language hiring requirements ("Have requirement for Java Full stack
developer, provide the matching resume") by retrieving the most relevant
resumes from a local knowledge base and generating a grounded, non-hallucinated
recommendation.

## How it works

```
data/resumes/*.txt  --load & chunk-->  RecursiveCharacterTextSplitter
                                             |
                                             v
                                   Embeddings (OpenAI or local demo)
                                             |
                                             v
                                     FAISS vector store
                                             |
User query --guardrails--> retriever (top-k + relevance score) --> prompt-engineered
                                             |                        LLM (ChatOpenAI)
                                             v                              |
                                   grounded, cited answer  <-----------------
```

Three guardrails keep the app honest:

1. **Unclear query** - if the query is too short/generic, the app returns the
   capability message instead of guessing.
2. **Low retrieval confidence** - if the best matching resume chunk scores
   below a similarity threshold, the app replies "Sorry, I cannot help with
   this query" instead of forcing an unrelated match. In `DEMO_MODE`, queries
   that share no vocabulary at all with the resume corpus (e.g. "what is the
   weather") produce an all-zero TF-IDF query vector; this is detected
   explicitly and short-circuited to the same rejection, since the
   distance-based confidence score is undefined (and otherwise misleadingly
   constant) for a zero vector (see `app/rag_chain.py`).
3. **Grounded prompt** - the system prompt (see `app/rag_chain.py`) forces the
   LLM to answer only from the retrieved resume excerpts and never invent
   candidates, skills, or experience.

## Project structure

```
resume-rag/
  data/resumes/          8 sample resumes (Java/Spring Boot, React, Full Stack
                          Python, Network Security/AWS, App Security, DevOps/AWS,
                          Data Engineer, ML Engineer)
  app/
    config.py             settings (models, thresholds, messages)
    ingest.py              build the FAISS vector store
    rag_chain.py            prompt template + retrieval + generation + guardrails
    demo_stubs.py            offline stand-ins used only in DEMO_MODE
    streamlit_app.py          chat UI
    cli.py                     command-line interface
  vectorstore/            generated FAISS index (created by ingest.py)
  requirements.txt
  .env.example
```

## Setup

```bash
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

## Build the vector store

```bash
python -m app.ingest
```

## Run

Streamlit web app:

```bash
streamlit run app/streamlit_app.py
```

Command line:

```bash
python -m app.cli "Have requirement for Java Full stack developer provide the matching resume"
```

## Offline demo mode (no API key required)

Set `DEMO_MODE=1` (in `.env` or as an environment variable) before running
`ingest.py` and the app. This swaps in a local TF-IDF embedding model and a
template-based grounded response generator (see `app/demo_stubs.py`) so the
full pipeline - retrieval, guardrails, and grounded answers - can be exercised
and demoed without any external API calls. This is how the screenshots in the
project documentation were generated. Set `DEMO_MODE=0` (default) with a real
`OPENAI_API_KEY` for production-quality answers from OpenAI's chat model.

## Example queries

- "Have requirement for Java Full stack developer provide the matching resume"
- "Looking for an AWS network security engineer with firewall experience"
- "Need a React frontend developer with Redux experience"
- "Application security engineer with OWASP and penetration testing skills"
- "help" -> triggers the "requirement is not clear" capability message
- "I need a professional chef with 10 years of experience in Italian cuisine"
  -> triggers the polite refusal ("Sorry, I cannot help with this query")
- "what is the weather" -> also triggers the refusal (no vocabulary overlap
  with the resume corpus in `DEMO_MODE`)
