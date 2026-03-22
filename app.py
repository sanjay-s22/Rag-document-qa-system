import streamlit as st
import requests, time, os
from audiorecorder import audiorecorder

API = os.getenv("API_URL", "http://127.0.0.1:8000")
MAX_Q = 500

st.set_page_config(page_title="Querify", page_icon="📄", layout="wide")
st.markdown("""<style>.stButton>button{width:100%}header a,h1 a,h2 a{visibility:hidden}</style>""", unsafe_allow_html=True)
st.markdown('<p style="font-size:3.5rem;font-weight:800;color:#1F77B4;text-align:center;">Querify</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#555;margin-bottom:0.5rem;">Ask questions about the uploaded document.</p>', unsafe_allow_html=True)
st.info("⚙️ Use the **sidebar** (tap **>** on mobile) to change model, chunking settings, and more.")

# session state defaults
for k, v in {"processed": False, "pdf_name": None, "chat_history": [], "debug_enabled": False}.items():
    if k not in st.session_state: st.session_state[k] = v

with st.sidebar:
    st.header("Settings")
    with st.expander("📋 Supported files & limitations"):
        st.markdown("**Supported:** Text-based PDFs, tables\n\n**Not supported:** Scanned PDFs, images, handwriting")

    # groq status
    st.subheader("Groq Status")
    try:
        h = requests.get(f"{API}/health", timeout=3).json()
        if h.get("groq_ready"):
            st.success("Groq API key set!")
        else:
            st.error("No valid Groq API key")
    except Exception:
        st.warning("Backend Not Reachable")

    # model selection
    model_name = st.selectbox("Choose model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
                               help="8B = fastest, 70B = smarter")
    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, 100)
    overlap    = st.slider("Overlap", 50, 500, 200, 50)
    top_k      = st.slider("Results to retrieve", 1, 6, 3)
    # debug panel toggle
    st.session_state.debug_enabled = st.toggle("Show RAG debug panel", value=st.session_state.debug_enabled)
    st.caption("Groq + LangChain + Qdrant (in-memory)")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload PDF")
    file = st.file_uploader("Choose PDF", type="pdf")

    if file and not st.session_state.processed:
        st.info(f"{file.name} ({file.size/1024:.2f} KB)")
        if st.button("Process", type="primary"):
            with st.spinner("Processing PDF..."):
                try:
                    r = requests.post(f"{API}/upload",
                        files={"file": (file.name, file, "application/pdf")},
                        params={"chunk_size": chunk_size, "chunk_overlap": overlap})
                    if r.status_code == 200:
                        st.session_state.processed = True
                        st.session_state.pdf_name  = file.name
                        st.success(r.json()["message"])
                    elif r.status_code == 429:
                        st.error("Too many requests. Please wait and try again.")
                    else:
                        st.error(r.json().get("detail", "Processing failed."))
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.processed:
        st.markdown("---")
        st.info(st.session_state.pdf_name)
        if st.button("🔄 Reset Session"):
            try:
                requests.post(f"{API}/reset")
            except Exception:
                pass
            st.session_state.update({"processed": False, "pdf_name": None, "chat_history": []})
            st.rerun()

# ── Chat ──────────────────────────────────────────────────────────────────────
with col2:
    st.header("Ask Questions")

    if not st.session_state.processed:
        st.info("Upload and process a PDF first.")
    else:
        for i, chat in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {chat['q']}\n\n**A{i+1}:** {chat['a']}")
            st.caption(f"⏱️ {chat['t']:.2f}s")
            # citations
            if chat.get("citations"):
                with st.expander(f"📎 Sources ({len(chat['citations'])} page(s))"):
                    for c in chat["citations"]:
                        st.markdown(f"**Page {c['page']}**")
                        st.caption(c["snippet"])
                        st.markdown("---")
            # debug panel
            if st.session_state.debug_enabled and chat.get("debug"):
                with st.expander(f"🔍 RAG Debug — Q{i+1}"):
                    d = chat["debug"]
                    st.markdown(f"**Rewritten query:** {d.get('rewritten_question','—')}\n\n**Chunks used:** {d.get('chunks_used','—')}")
                    for j, chunk in enumerate(d.get("chunks_retrieved", [])):
                        st.text_area(f"Chunk {j+1}", chunk, height=80, key=f"chunk_{i}_{j}")
            st.markdown("---")

        # stt
        st.markdown("##### 🎙️ Voice Input")
        audio = audiorecorder("🎤 Record", "⏹ Stop")
        if len(audio) > 0:
            with st.spinner("Transcribing..."):
                try:
                    r = requests.post(f"{API}/transcribe", files={"file": ("recording.wav", audio.export().read(), "audio/wav")})
                    if r.status_code == 200:
                        st.success(f"Transcribed: {r.json().get('transcript','')}")
                        st.session_state["stt_transcript"] = r.json().get("transcript", "")
                    elif r.status_code == 429:
                        st.error("Too many requests.")
                    else:
                        st.error(r.json().get("detail", "Transcription failed."))
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("---")
        prefill  = st.session_state.pop("stt_transcript", "")
        question = st.text_area("Your question:", value=prefill,
                                placeholder="What is this document about?", height=100, max_chars=MAX_Q)
        if question:
            remaining = MAX_Q - len(question)
            st.markdown(f'<p style="color:{"red" if remaining<50 else "gray"};font-size:0.8rem;">{remaining} characters remaining</p>', unsafe_allow_html=True)

        colb1, colb2 = st.columns(2)
        ask = colb1.button("Get Answer", type="primary")
        if colb2.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

        if ask and question.strip():
            with st.spinner("Generating answer..."):
                try:
                    start = time.time()
                    r = requests.post(f"{API}/query", params={"question": question, "k": top_k, "model": model_name})
                    elapsed = time.time() - start
                    if r.status_code == 200:
                        data = r.json()
                        st.markdown("### Answer")
                        st.write(data.get("answer"))
                        st.caption(f"⏱️ {elapsed:.2f}s")
                        # citations
                        if data.get("citations"):
                            with st.expander(f"📎 Sources ({len(data['citations'])} page(s))"):
                                for c in data["citations"]:
                                    st.markdown(f"**Page {c['page']}**")
                                    st.caption(c["snippet"])
                                    st.markdown("---")
                        st.session_state.chat_history.append({
                            "q": question, "a": data.get("answer"), "t": elapsed,
                            "citations": data.get("citations", []),
                            "debug": {"rewritten_question": data.get("rewritten_question"),
                                      "chunks_used": data.get("chunks_used"),
                                      "chunks_retrieved": data.get("chunks_retrieved", [])},
                        })
                    elif r.status_code == 429:
                        st.error("Too many requests.")
                    else:
                        st.error(r.json().get("detail", "Query failed."))
                except Exception as e:
                    st.error(f"Error: {e}")
        elif ask:
            st.warning("Enter a question first.")

st.markdown("---")
st.markdown("Built by Sanjay S.")