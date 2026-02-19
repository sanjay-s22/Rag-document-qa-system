import streamlit as st
import time, tempfile, os
from rag_analyzer import RAGAnalyzer, check_groq

st.set_page_config(page_title="Querify", page_icon="📄", layout="wide")

# Session state
for key, default in [('analyzer', None), ('processed', False), ('pdf_name', None), ('chat_history', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("""<style>
.stButton>button {width: 100%;}
header a {visibility: hidden;}
h1 a, h2 a, h3 a {visibility: hidden;}
</style>""", unsafe_allow_html=True)

st.markdown('<p style="font-size: 4rem; font-weight: 800; color: #1F77B4; text-align: center; margin-bottom: 0.5rem;">Querify</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.4rem; color: #555; text-align: center; margin-bottom: 2rem;">Ask questions about the uploaded document.</p>', unsafe_allow_html=True)

groq_api_key = st.secrets.get("GROQ_API_KEY")

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown("---")

    # Groq status 
    st.subheader("🔧 Groq Status")
    is_ready, models = check_groq()
    if is_ready:
        st.success("✅ Groq API key set!")
    else:
        st.error("❌ No valid Groq API key")
        st.caption("Groq API key must be configured in Streamlit secrets.")

    # Model selection 
    st.markdown("---")
    st.subheader("Model")
    model_name = st.selectbox(
        "Choose model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        help="8b model = faster responses, 70b = better reasoning (slower)"
    )

    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Overlap", 50, 500, 200, 50)
    top_k = st.slider("Results to retrieve", 1, 5, 3)

    st.markdown("---")
    
    if st.session_state.processed:
        st.subheader("Session")
        st.info(f"📝 {st.session_state.pdf_name}")
        if st.button("🔄 Reset"):
            for k in ['analyzer', 'processed', 'pdf_name', 'chat_history']:
                st.session_state[k] = None if k != 'chat_history' else []
            st.rerun()

    st.markdown("---")
    st.caption("Groq LLM + LangChain + Qdrant (in-memory)")

# Main logic  
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader("Choose PDF", type="pdf")

    if uploaded_file and not st.session_state.processed:
        st.info(f"📁 {uploaded_file.name} ({uploaded_file.size/1024:.2f} KB)")

        if st.button("Process", type="primary", disabled=not is_ready):
            if not is_ready:
                st.error("❌ Add your Groq API key in the sidebar first!")
            else:
                with st.spinner("Processing PDF..."):
                    try:
                        # PyPDFLoader needs a local path—writing bytes to disk temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getbuffer())
                            tmp_path = tmp.name

                        analyzer = RAGAnalyzer(model_name=model_name)
                        result = analyzer.process_pdf(tmp_path, chunk_size, chunk_overlap)
                        os.unlink(tmp_path)

                        if result["success"]:
                            st.session_state.analyzer = analyzer
                            st.session_state.processed = True
                            st.session_state.pdf_name = uploaded_file.name
                            st.success(result["message"])
                        else:
                            st.error(result["message"])
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

with col2:
    st.header("Ask Questions")

    if st.session_state.processed:
        # Chat history
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                st.markdown(f"**Q{i+1}:** {chat['question']}")
                st.markdown(f"**A{i+1}:** {chat['answer']}")
                if 'time' in chat:
                    st.caption(f"⏱️ {chat['time']:.2f}s")
                st.markdown("---")

        question = st.text_area("Your question:", placeholder="What is this document about?", height=100)
        col_btn1, col_btn2 = st.columns([1, 1])

        with col_btn1:
            ask_button = st.button("Get Answer", type="primary")
        with col_btn2:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()

        if ask_button and question:
            with st.spinner("Generating answer..."):
                try:
                    start = time.time()
                    result = st.session_state.analyzer.answer_question(question, top_k)
                    elapsed = time.time() - start
                    
                    if result["success"]:
                        st.markdown("### Answer")
                        st.write(result["answer"])
                        st.caption(f"⏱️ {elapsed:.2f}s")
                        st.session_state.chat_history.append({       # keep previous Q&A so user can scroll back
                            'question': question,
                            'answer': result["answer"],
                            'time': elapsed
                        })                                                
                        st.success("Response generated.")
                    else:
                        st.error(result["message"])
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        elif ask_button:
            st.warning("Enter a question first")
        else:
            st.info("Upload and process a PDF first")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Built by Sanjay. S")