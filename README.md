## Querify — RAG Document QA System

AI-powered PDF document analyzer using Retrieval-Augmented Generation (RAG) with voice input support

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logoColor=white)](https://groq.com)

**Live App:** https://rag-document-app-system-kc9htnpnjbzbexjdk4ws98.streamlit.app/  
**API (Backend):** https://rag-document-qa-system-1-6xi0.onrender.com

> **Note:** Both services are on free tiers and spin down after inactivity. The first request may take 1–3 minutes while the backend wakes up. Subsequent requests are fast.

---

## What's New in v3

- **Voice input (STT)** — Ask questions by speaking directly in the browser. Powered by Groq's `whisper-large-v3-turbo` — transcription runs on Groq's servers, zero RAM cost on the backend
- **Query rewriting** — Before retrieval, the LLM rephrases the user's question to be more search-friendly, improving chunk matching on vague or short queries
- **Structured source citations** — Every answer now returns a collapsible citations panel showing the exact page number and chunk snippet the answer was pulled from
- **RAG debug panel** — Toggle in the sidebar to inspect the rewritten query, number of chunks used, and chunk previews for every response
- **PyMuPDF** — Switched from PyPDF to PyMuPDF for significantly better text extraction from tables, multi-column layouts, and complex formatting
- **Supported files info** — Users are clearly informed about what Querify can and can't handle before uploading

> v2 (FastAPI + Streamlit refactor) and v1 (pure Streamlit) are preserved in git history.

---

## Features

- **Multimodal Input** — Type or speak your questions; voice is transcribed via Groq Whisper
- **Table-Aware Parsing** — PyMuPDF preserves table structure and layout during extraction
- **Query Rewriting** — LLM-powered query reformulation before retrieval for better results
- **Low-Latency QA** — LLM inference offloaded to Groq for near-instant responses
- **Smart Chunking** — Recursive character splitting with configurable size and overlap
- **Semantic Search** — Qdrant in-memory vector store for meaning-based retrieval
- **Source Citations** — Collapsible panel with page number and chunk snippet for every answer
- **RAG Debug Panel** — Inspect rewritten queries and retrieved chunks per response
- **Model Selection** — Switch between Llama 3.1 8B and Llama 3.3 70B per query
- **Security** — Rate limiting, input validation, prompt injection filtering, PDF magic bytes check

---

## Tech Stack

**Frontend**
- ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) Streamlit — web UI, file upload, voice recorder

**Backend**
- ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) FastAPI — REST API with automatic `/docs` swagger UI
- ![Groq](https://img.shields.io/badge/Groq-F55036?style=flat&logoColor=white) Groq — LLM inference (Llama 3.1 8B, Llama 3.3 70B) + Whisper STT
- ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black) HuggingFace — embeddings (all-MiniLM-L6-v2)
- ![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=flat&logoColor=white) Qdrant — in-memory vector database
- ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white) LangChain — RAG pipeline orchestration
- ![PyMuPDF](https://img.shields.io/badge/PyMuPDF-3776AB?style=flat&logo=python&logoColor=white) PyMuPDF — PDF text & table extraction

---

## Prerequisites

- Python 3.12
- Groq API key (free tier at [console.groq.com](https://console.groq.com))
- ffmpeg (required locally for voice recording — already available on Streamlit Cloud and Render)

```bash
# Windows
winget install ffmpeg

# macOS
brew install ffmpeg
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/sanjay-s22/rag-document-qa-system.git
cd rag-document-qa-system
```

**2. Create and activate virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
# Backend
pip install -r backend/requirements.txt

# Frontend
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the project root:
```
GROQ_API_KEY=gsk_your_api_key_here  # Get yours free at console.groq.com
```

---

## Usage

**Start the backend first:**
```bash
cd backend
uvicorn main:app --reload
```

**Then start the frontend in a separate terminal (from root):**
```bash
streamlit run app.py
```

The FastAPI backend runs on `http://localhost:8000` — visit `/docs` for the interactive API explorer.

**Workflow:**
1. Upload a PDF via the file uploader and click **Process**
2. Type your question or click **🎤 Record** to ask by voice
3. Click **Get Answer** to get an AI-generated response
4. Expand **📎 Sources** to see which pages the answer came from
5. Enable the **RAG debug panel** in the sidebar to inspect retrieval internals

---

## Example Queries

- *"Summarize the main findings of this report"*
- *"What were the cash and cash equivalents in 2009?"*
- *"What technical skills are listed in this resume?"*
- *"List all key recommendations from section 3"*

---

## Configuration

Adjustable in the sidebar:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Model | 8B / 70B | 8B | Speed vs quality tradeoff |
| Chunk Size | 500–2000 | 1000 | Characters per chunk |
| Chunk Overlap | 50–500 | 200 | Overlap between chunks |
| Top-K | 1–6 | 3 | Chunks retrieved per query |

---

## Project Structure

```
rag-document-qa-system/
├── backend/
│   ├── main.py            # FastAPI routes, rate limiting, input validation
│   ├── rag_service.py     # RAG pipeline — embeddings, query rewriting, vector store, LLM
│   ├── stt_service.py     # Speech-to-text via Groq Whisper API
│   └── requirements.txt   # Backend dependencies
├── app.py                 # Streamlit frontend
├── requirements.txt       # Frontend dependencies
├── .env                   # Environment variables (not tracked)
├── .gitignore
└── README.md
```

---

## Architecture

```
User (text or voice)
 │
 ▼
Streamlit Frontend (app.py)
 │  HTTP requests
 ▼
FastAPI Backend (main.py)
 ├── Rate limiting (slowapi)
 ├── Input validation
 ├── Prompt injection filter
 │
 ├── /transcribe ──► Groq Whisper API (STT)
 │
 └── /query ──► RAG Service (rag_service.py)
                 ├── Query rewriting (Groq LLM)
                 ├── PyMuPDF — extract text & tables
                 ├── RecursiveCharacterTextSplitter — chunk
                 ├── SentenceTransformer — embed (all-MiniLM-L6-v2)
                 ├── Qdrant (in-memory) — vector search
                 └── Groq LLM — generate answer + citations
```

---

## API Endpoints

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| `GET` | `/health` | Check API and Groq key status | — |
| `POST` | `/upload` | Upload and process a PDF | 10/min |
| `POST` | `/query` | Ask a question against the document | 20/min |
| `POST` | `/transcribe` | Transcribe audio to text via Groq Whisper | 10/min |
| `POST` | `/reset` | Clear session and vector store | 10/min |

Visit `/docs` on the backend URL for the full interactive API explorer.

---

## Supported Files

| Type | Supported |
|------|-----------|
| Text-based PDFs | ✅ |
| PDFs with tables | ✅ |
| Scanned PDFs | ❌ (no OCR) |
| Images / charts | ❌ |
| Handwritten text | ❌ |

---

## Security

- API keys never committed to version control
- Rate limiting on all endpoints (10–20 req/min per IP)
- Server-side input validation on all parameters
- Regex-based prompt injection filtering
- PDF magic bytes verification
- Safe temp file handling — no race conditions, always cleaned up
- CORS locked to Streamlit Cloud URL + localhost

> **Note:** Single-user system — the backend maintains one shared RAG session. Multi-user session management is on the roadmap.

---

## Roadmap

- Multi-user session management
- LLM Guard integration for stronger prompt safety
- Persistent vector database (replace in-memory Qdrant)
- Streaming LLM responses

---

## Author

**Sanjay**  
GitHub: [@sanjay-s22](https://github.com/sanjay-s22)

---

## Acknowledgments

Built with:
- [LangChain](https://langchain.com) — RAG orchestration
- [Groq](https://groq.com) — LLM inference + Whisper STT
- [FastAPI](https://fastapi.tiangolo.com) — Backend framework
- [Qdrant](https://qdrant.tech) — Vector similarity search
- [Streamlit](https://streamlit.io) — Frontend framework
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF extraction
- [HuggingFace](https://huggingface.co) — Embedding models


---

⭐ Star this repo if you found it helpful!
