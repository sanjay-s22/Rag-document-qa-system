import streamlit as st
import requests, time, os, uuid, re
from audiorecorder import audiorecorder

API = os.getenv("API_URL", "http://127.0.0.1:8000")
MAX_Q_LENGTH = 500

st.set_page_config(page_title="Querify", page_icon="📄", layout="wide")

st.markdown("""<style>
.stButton>button {width: 100%;}
header a {visibility: hidden;}
h1 a, h2 a {visibility: hidden;}
</style>""", unsafe_allow_html=True)

# Session ID setup
# On first visit: generate a UUID, write it to the URL, store in session state.
# On return visits: read uid from URL param so the user keeps their Qdrant collection.
# If the user loses the URL they can paste their old session ID in the sidebar to restore.

params = st.query_params

if "user_id" not in st.session_state:
    if "uid" in params:
        st.session_state.user_id = params["uid"]
    else:
        new_id = str(uuid.uuid4())
        st.session_state.user_id = new_id
        st.query_params["uid"] = new_id

USER_ID = st.session_state.user_id

# Session state defaults
for k, v in {
    "processed": False,
    "pdf_name": None,
    "chat_history": [],
    "debug_enabled": False,
    "show_history": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# On first load check if this user already has a document indexed in Qdrant.
# This handles the case where the backend restarted but Qdrant Cloud still has their data.
if not st.session_state.processed:
    try:
        r = requests.get(f"{API}/status", params={"user_id": USER_ID}, timeout=3)
        if r.status_code == 200 and r.json().get("has_document"):
            st.session_state.processed = True
            st.session_state.pdf_name = "Previously uploaded document"
    except Exception:
        pass


# Header
st.markdown('<p style="font-size:3.5rem;font-weight:800;color:#1F77B4;text-align:center;">Querify</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#555;margin-bottom:0.5rem;">Ask questions about the uploaded document.</p>', unsafe_allow_html=True)
st.info("⚙️ Use the **sidebar** (tap **>** on mobile) to change model, chunking settings, and more.")


# Sidebar
with st.sidebar:
    st.header("Settings")
    st.markdown("---")

    with st.expander("📋 Supported files & limitations"):
        st.markdown("""
**Supported**
- Text-based PDFs
- Tables inside PDFs

**Not fully supported**
- Scanned PDFs (no OCR)
- Images / charts
- Handwritten text
        """)

    st.markdown("---")

    st.subheader("Groq Status")
    try:
        health = requests.get(f"{API}/health", timeout=3).json()
        if health.get("groq_ready"):
            st.success("Groq API key set!")
        else:
            st.error("No valid Groq API key")
    except Exception:
        st.warning("Backend Not Reachable")

    st.markdown("---")

    model_name = st.selectbox(
        "Choose model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        help="8B = fastest, 70B = smarter and more detailed"
    )
    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, 100)
    overlap    = st.slider("Overlap", 50, 500, 200, 50)
    top_k      = st.slider("Results to retrieve", 1, 6, 3)

    st.markdown("---")
    st.session_state.debug_enabled = st.toggle(
        "Show RAG debug panel", value=st.session_state.debug_enabled
    )

    st.markdown("---")

    # Session management 
    st.subheader("Session")
    base_url = os.getenv("APP_URL", "http://localhost:8501")
    st.code(f"{base_url}/?uid={USER_ID}", language=None)
    st.caption("Bookmark this URL to return to your session.")

    restore_id = st.text_input("Restore a previous session ID:")
    if st.button("Restore Session") and restore_id.strip():
        # Validate it looks like a UUID before trusting it
        if re.match(r'^[a-f0-9\-]{36}$', restore_id.strip()):
            st.query_params["uid"] = restore_id.strip()
            st.session_state.user_id = restore_id.strip()
            # Reset processed state so the status check reruns for the restored session
            st.session_state.processed = False
            st.session_state.pdf_name = None
            st.session_state.chat_history = []
            st.rerun()
        else:
            st.error("Invalid session ID.")

    st.markdown("---")
    st.caption("Groq + LangChain + Qdrant Cloud")
    st.caption(f"Session: `{USER_ID[:8]}…`")


# Chat history panel
if st.session_state.show_history:
    with st.container():
        st.markdown("###  Chat History (last 7 days)")
        try:
            r = requests.get(f"{API}/history", params={"user_id": USER_ID}, timeout=5)
            if r.status_code == 200:
                entries = r.json().get("history", [])
                if not entries:
                    st.info("No history found.")
                else:
                    for i, entry in enumerate(reversed(entries)):
                        created = entry.get("created_at", "")[:19].replace("T", " ")
                        with st.expander(f"Q: {entry['question'][:80]}  —  {created}"):
                            st.markdown(f"**Question:** {entry['question']}")
                            if entry.get("rewritten_question") and entry["rewritten_question"] != entry["question"]:
                                st.caption(f"Rewritten: {entry['rewritten_question']}")
                            st.markdown(f"**Answer:** {entry['answer']}")
                            if entry.get("citations"):
                                st.markdown("**Sources:**")
                                for cit in entry["citations"]:
                                    st.caption(f"Page {cit['page']}: {cit['snippet'][:200]}")
            else:
                st.error("Failed to load history.")
        except Exception as e:
            st.error(f"Error: {e}")

        if st.button("Close History"):
            st.session_state.show_history = False
            st.rerun()

    st.markdown("---")


# Main columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload PDF")
    file = st.file_uploader("Choose PDF", type="pdf")

    if file and not st.session_state.processed:
        st.info(f"{file.name} ({file.size/1024:.2f} KB)")

        if st.button("Process", type="primary"):
            with st.spinner("Processing PDF..."):
                try:
                    r = requests.post(
                        f"{API}/upload",
                        files={"file": (file.name, file, "application/pdf")},
                        params={
                            "user_id": USER_ID,
                            "chunk_size": chunk_size,
                            "chunk_overlap": overlap,
                        }
                    )
                    if r.status_code == 200:
                        st.session_state.processed = True
                        st.session_state.pdf_name = file.name
                        st.success(r.json()["message"])
                    elif r.status_code == 429:
                        st.error("Too many requests. Please wait a moment and try again.")
                    else:
                        st.error(r.json().get("detail", "Processing failed."))
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.processed:
        st.markdown("---")
        st.info(f"{st.session_state.pdf_name}")

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("🔄 Reset Session"):
                try:
                    requests.post(f"{API}/reset", params={"user_id": USER_ID})
                except Exception:
                    pass
                st.session_state.update({
                    "processed": False,
                    "pdf_name": None,
                    "chat_history": [],
                    "show_history": False,
                })
                st.rerun()
        with bcol2:
            if st.button("📜 View History"):
                st.session_state.show_history = not st.session_state.show_history
                st.rerun()


with col2:
    st.header("Ask Questions")

    if st.session_state.processed:
        # Render in-session Q&A pairs oldest → newest
        for i, chat in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {chat['q']}")
            st.markdown(f"**A{i+1}:** {chat['a']}")
            st.caption(f"⏱️ {chat['t']:.2f}s")

            if chat.get("citations"):
                with st.expander(f"📎 Sources ({len(chat['citations'])} page(s))"):
                    for cit in chat["citations"]:
                        st.markdown(f"**Page {cit['page']}**")
                        st.caption(cit["snippet"])
                        st.markdown("---")

            if st.session_state.debug_enabled and chat.get("debug"):
                with st.expander(f"🔍 RAG Debug — Q{i+1}"):
                    debug = chat["debug"]
                    st.markdown(f"**Rewritten query:** {debug.get('rewritten_question', '—')}")
                    st.markdown(f"**Chunks used:** {debug.get('chunks_used', '—')}")
                    st.markdown("**Retrieved chunk previews:**")
                    for j, chunk in enumerate(debug.get("chunks_retrieved", [])):
                        st.text_area(f"Chunk {j+1}", chunk, height=80, key=f"chunk_{i}_{j}")

            st.markdown("---")

        # Voice input (STT)
        st.markdown("##### 🎙️ Voice Input")
        audio = audiorecorder("🎤 Record", "⏹ Stop")

        if len(audio) > 0:
            audio_bytes = audio.export().read()
            with st.spinner("Transcribing..."):
                try:
                    r = requests.post(
                        f"{API}/transcribe",
                        files={"file": ("recording.wav", audio_bytes, "audio/wav")}
                    )
                    if r.status_code == 200:
                        st.success(f"Transcribed: {r.json().get('transcript', '')}")
                        st.session_state["stt_transcript"] = r.json().get("transcript", "")
                    elif r.status_code == 429:
                        st.error("Too many requests. Please wait a moment and try again.")
                    else:
                        st.error(r.json().get("detail", "Transcription failed."))
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("---")

        # Pre-fill the text area with the STT transcript if one was just produced
        prefill = st.session_state.pop("stt_transcript", "")
        question = st.text_area(
            "Your question:",
            value=prefill,
            placeholder="What is this document about?",
            height=100,
            max_chars=MAX_Q_LENGTH
        )

        if question:
            remaining = MAX_Q_LENGTH - len(question)
            color = "red" if remaining < 50 else "gray"
            st.markdown(f'<p style="color:{color};font-size:0.8rem;">{remaining} characters remaining</p>', unsafe_allow_html=True)

        colb1, colb2 = st.columns([1, 1])
        with colb1:
            ask = st.button("Get Answer", type="primary")
        with colb2:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()

        if ask and question.strip():
            with st.spinner("Generating answer..."):
                try:
                    start = time.time()
                    r = requests.post(
                        f"{API}/query",
                        params={
                            "question": question,
                            "k": top_k,
                            "model": model_name,
                            "user_id": USER_ID,
                        }
                    )
                    elapsed = time.time() - start

                    if r.status_code == 200:
                        data = r.json()
                        ans = data.get("answer")
                        citations = data.get("citations", [])

                        st.markdown("### Answer")
                        st.write(ans)
                        st.caption(f"⏱️ {elapsed:.2f}s")

                        # Show citations inline right after the answer
                        if citations:
                            with st.expander(f" Sources ({len(citations)} page(s))"):
                                for cit in citations:
                                    st.markdown(f"**Page {cit['page']}**")
                                    st.caption(cit["snippet"])
                                    st.markdown("---")

                        # Bundle debug info to store alongside this chat turn
                        debug_info = {
                            "rewritten_question": data.get("rewritten_question"),
                            "chunks_used": data.get("chunks_used"),
                            "chunks_retrieved": data.get("chunks_retrieved", []),
                        }

                        st.session_state.chat_history.append({
                            "q": question,
                            "a": ans,
                            "t": elapsed,
                            "citations": citations,
                            "debug": debug_info,
                        })
                    elif r.status_code == 429:
                        st.error("Too many requests. Please wait a moment and try again.")
                    else:
                        st.error(r.json().get("detail", "Query failed."))
                except Exception as e:
                    st.error(f"Error: {e}")

        elif ask:
            st.warning("Enter a question first.")
    else:
        st.info("Upload and process a PDF first.")

st.markdown("---")
st.markdown("Built by Sanjay S.")