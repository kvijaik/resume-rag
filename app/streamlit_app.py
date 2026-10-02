"""
Streamlit chat UI for the Resume RAG application.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys  # standard library: used to tweak sys.path so `app` is importable when run as a script
from pathlib import Path  # standard library: filesystem paths

import streamlit as st  # Streamlit: builds the web UI (widgets, chat components, session state) from a single script

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add the project root to sys.path so `from app import ...` resolves when this file is run directly
from app import config  # noqa: E402  # project settings, used here to check DEMO_MODE
from app.ingest import SUPPORTED_EXTENSIONS, build_vectorstore, save_uploaded_resume  # noqa: E402  # upload validation + index rebuild
from app.rag_chain import answer_query, load_vectorstore  # noqa: E402  # RAG pipeline entry point + vector store loader

st.set_page_config(page_title="Resume RAG Assistant", page_icon="🧑‍💻", layout="centered")  # browser tab title/icon and page width, must be the first Streamlit call

STATUS_BADGES = {
    "matched": "✅ Match found",
    "no_match": "🚫 No match / out of scope",
    "unclear": "❓ Requirement unclear",
}  # maps answer_query()'s status string to a human-readable badge shown above each answer

EXAMPLE_QUERIES = [
    "Have requirement for Java Full stack developer provide the matching resume",
    "Looking for an AWS network security engineer with firewall experience",
    "Need a React frontend developer with Redux experience",
    "Application security engineer with OWASP and penetration testing skills",
]  # one-click example prompts rendered as sidebar buttons


@st.cache_resource(show_spinner="Loading resume knowledge base...")
def get_vectorstore():
    # Loads the FAISS index once per Streamlit server process and reuses it across reruns/users
    # (st.cache_resource caches non-serializable objects like this, unlike st.cache_data).
    return load_vectorstore()


def library_candidates(vectorstore) -> list[tuple[str, str]]:
    """(name, title) of every resume in the loaded index, sorted by name."""
    by_source = {}  # source file -> (name, title); every chunk of a resume carries the same metadata
    for doc in vectorstore.docstore._dict.values():  # InMemoryDocstore keeps the indexed chunks in _dict
        by_source.setdefault(doc.metadata.get("source"), (doc.metadata.get("candidate_name", ""), doc.metadata.get("candidate_title", "")))
    return sorted(by_source.values())


def add_uploaded_resumes(files) -> None:
    """Save uploaded resumes into data/resumes, then rebuild the index so they are searchable right away."""
    saved = []  # filenames accepted this time
    for f in files:  # each item is a Streamlit UploadedFile (has .name and .getvalue())
        try:
            saved.append(save_uploaded_resume(f.name, f.getvalue()).name)  # validates type + readable text before keeping the file
        except ValueError as e:
            st.error(str(e))  # e.g. unsupported type or a scanned PDF; other files in the batch still go through
    if saved:
        with st.spinner("Rebuilding the resume index..."):
            build_vectorstore()  # re-embed the whole library, including the new resumes
        get_vectorstore.clear()  # drop the cached old index so the next load picks up the new one
        st.success(f"Added {len(saved)} resume(s): {', '.join(saved)}")


def main():
    st.title("🧑‍💻 Resume RAG Assistant")  # page heading
    st.caption(
        "A Retrieval-Augmented Generation demo: ask for a technology role and the "
        "assistant retrieves and recommends matching resumes from the internal "
        "resume knowledge base — grounded only in the resumes it has, no guessing."
    )  # subtitle explaining what the app does

    if config.DEMO_MODE:  # surface a visible banner whenever running without a real OpenAI backend
        st.info(
            "🔧 Running in **DEMO_MODE** — using a local offline embedding/answer "
            "generator instead of the OpenAI API, so this demo works without an API "
            "key. Set `DEMO_MODE=0` and provide `OPENAI_API_KEY` to use the real "
            "OpenAI-backed pipeline.",
            icon="🔧",
        )

    with st.sidebar:  # left sidebar: example query shortcuts, resume upload, and (below, once loaded) the library contents
        st.subheader("Try an example")
        for q in EXAMPLE_QUERIES:  # render one button per example query
            if st.button(q, use_container_width=True, key=f"ex_{q}"):  # unique widget key per query text, required by Streamlit for buttons in a loop
                st.session_state["pending_query"] = q  # stash the clicked example so it gets processed as if typed into chat_input below
        st.divider()
        st.subheader("Add resumes")
        uploads = st.file_uploader(
            "Upload .txt, .pdf or .docx files",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],  # the browser file picker only offers these types
            accept_multiple_files=True,
            key="resume_upload",
        )
        if st.button("Add to library", disabled=not uploads, use_container_width=True, key="add_resumes"):  # disabled until at least one file is chosen
            add_uploaded_resumes(uploads)

    try:
        vectorstore = get_vectorstore()  # load (or fetch cached) FAISS index; raises RuntimeError if `python -m app.ingest` was never run
    except RuntimeError as e:
        st.error(str(e))  # show the "run ingest first" message to the user
        st.stop()  # halt script execution here; nothing below runs without a vector store

    with st.sidebar:  # list who is actually searchable, so users know what the library covers
        candidates = library_candidates(vectorstore)
        st.divider()
        st.subheader(f"In the library ({len(candidates)})")
        for name, title in candidates:
            st.markdown(f"- {name}" + (f" — {title}" if title else ""))

    if "messages" not in st.session_state:  # first run for this browser session: seed the chat history with a greeting
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! Describe a hiring requirement (technology / role) and I'll "
                    "recommend matching resumes. For example: *\"Have requirement for "
                    "Java Full stack developer, provide the matching resume\"*."
                ),
            }
        ]

    for msg in st.session_state.messages:  # replay the full chat history on every rerun (Streamlit reruns the whole script per interaction)
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Describe the role / technology you're hiring for...")  # the persistent chat input box at the bottom of the page
    if "pending_query" in st.session_state:  # an example button was clicked this rerun -> treat it as if the user typed it
        query = st.session_state.pop("pending_query")  # consume it so it only fires once, not on every subsequent rerun

    if query:  # a new query arrived this run (typed or from an example button)
        st.session_state.messages.append({"role": "user", "content": query})  # persist the user's message in chat history
        with st.chat_message("user"):
            st.markdown(query)  # render the user's message bubble immediately

        with st.chat_message("assistant"):
            with st.spinner("Retrieving matching resumes..."):
                result = answer_query(query, vectorstore=vectorstore)  # run the full guardrail -> retrieve -> generate pipeline, reusing the cached vectorstore
            badge = STATUS_BADGES.get(result["status"], "")  # look up the human-readable status badge for this result
            st.markdown(f"**{badge}**")  # show the status badge above the answer
            st.markdown(result["answer"])  # render the generated (or template) answer text
            if result["sources"]:  # only show a sources line when there actually are matched resumes
                st.caption("Sources: " + ", ".join(result["sources"]))

        st.session_state.messages.append(
            {"role": "assistant", "content": f"**{badge}**\n\n{result['answer']}"}
        )  # persist the assistant's reply (badge + answer) so it survives the next rerun/replay


if __name__ == "__main__":  # `streamlit run app/streamlit_app.py` executes this file as __main__
    main()
