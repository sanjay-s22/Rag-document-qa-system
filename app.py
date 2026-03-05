import streamlit as st
import requests, time, os

API = os.getenv("API_URL", "http://127.0.0.1:8000")
MAX_Q_LENGTH = 500

st.set_page_config(page_title="Querify", page_icon="📄", layout="wide")

st.markdown("""<style>
.stButton>button {width: 100%;}
header a {visibility: hidden;}
h1 a, h2 a {visibility: hidden;}
</style>""", unsafe_allow_html=True)

st.markdown('<p style="font-size:3.5rem;font-weight:800;color:#1F77B4;text-align:center;">Querify</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#555;margin-bottom:2rem;">Ask questions about the uploaded document.</p>', unsafe_allow_html=True)

# Session state defaults
for k, v in {"processed": False, "pdf_name": None, "chat_history": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v


with st.sidebar:
    st.header("Settings")
    st.markdown("---")
     # Groq status
    st.subheader("Groq Status")
    try:
        health = requests.get(f"{API}/health", timeout=3).json()
        if health.get("groq_ready"):
            st.success("Groq API key set!")
        else:
            st.error("No valid Groq API key")
    except Exception:
        st.warning("Backend not reachable")

    st.markdown("---")

    # Model selection
    model_name = st.selectbox("Choose model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"], help="8B = fastest, 70B = smarter and more detailed")
    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, 100)
    overlap    = st.slider("Overlap", 50, 500, 200, 50)
    top_k      = st.slider("Results to retrieve", 1, 6, 3)
    st.markdown("---")
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
                    r = requests.post(
                        f"{API}/upload",
                        files={"file": (file.name, file, "application/pdf")},
                        params={"chunk_size": chunk_size, "chunk_overlap": overlap}
                    )
                    if r.status_code == 200:
                        st.session_state.processed = True
                        st.session_state.pdf_name = file.name
                        st.success(f"{r.json()['message']}")
                    elif r.status_code == 429:
                        st.error("Too many requests. Please wait a moment and try again.")
                    else:
                        st.error(f"{r.json().get('detail', 'Processing failed.')}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.session_state.processed:
        st.markdown("---")
        st.info(f"{st.session_state.pdf_name}")
        if st.button("🔄 Reset Session"):
            try:
                requests.post(f"{API}/reset")
            except Exception:
                pass
            st.session_state.update({"processed": False, "pdf_name": None, "chat_history": []})
            st.rerun()

with col2:
    st.header("Ask Questions")

    if st.session_state.processed:
        for i, chat in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {chat['q']}")
            st.markdown(f"**A{i+1}:** {chat['a']}")
            st.caption(f"⏱️ {chat['t']:.2f}s")
            st.markdown("---")

        question = st.text_area("Your question:", placeholder="What is this document about?", height=100, max_chars=MAX_Q_LENGTH)

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
                    r = requests.post(f"{API}/query", params={"question": question, "k": top_k, "model": model_name})
                    elapsed = time.time() - start

                    if r.status_code == 200:
                        ans = r.json().get("answer")
                        st.markdown("### Answer")
                        st.write(ans)
                        st.caption(f"⏱️ {elapsed:.2f}s")
                        st.session_state.chat_history.append({"q": question, "a": ans, "t": elapsed})
                    elif r.status_code == 429:
                        st.error("Too many requests. Please wait a moment and try again.")
                    else:
                        st.error(f"{r.json().get('detail', 'Query failed.')}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif ask:
            st.warning("Enter a question first.")
    else:
        st.info("Upload and process a PDF first.")

st.markdown("---")
st.markdown("Built by Sanjay S.")