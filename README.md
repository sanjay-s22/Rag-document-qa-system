# Querify — RAG Document QA System

AI-powered PDF document analyzer using Retrieval-Augmented Generation (RAG)

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logoColor=white)](https://groq.com)

**Live App:** https://rag-document-app-system-kc9htnpnjbzbexjdk4ws98.streamlit.app/  
**API (Backend):** https://rag-document-qa-system.onrender.com

> **Note:** Both the frontend and backend are hosted on free tiers. After a period of inactivity, they spin down automatically. The first request after waking up may take 1–3 minutes while the backend spins up. Subsequent requests will be fast.

---

## What's New in v2

The project has been refactored from a single Streamlit app into a proper **FastAPI backend + Streamlit frontend** architecture:

- **Decoupled architecture** — frontend and backend are fully separated; backend is independently testable via `/docs`
- **Rate limiting** — all API endpoints are protected with per-IP request limits using `slowapi`
- **Input validation** — chunk params, question length, and model selection are all validated server-side
- **Prompt injection filtering** — regex-based filter blocks common injection attempts before they reach the LLM
- **PDF magic bytes check** — verifies uploaded files are actually PDFs, not just renamed files
- **Safe temp file handling** — no race condition on concurrent uploads, always cleaned up after processing
- **Blank page filtering** — empty pages are stripped before indexing to reduce noise
- **Embedding model caching** — `lru_cache` prevents reloading the SentenceTransformer on every request

> v1 (pure Streamlit) is preserved in git history.

---

## Features

- **Low-Latency QA** — Offloads LLM inference to Groq for near-instant responses
- **Smart Chunking** — Recursive character splitting with configurable size and overlap
- **Semantic Search** — Qdrant in-memory vector store for meaning-based retrieval, not just keyword matching
- **Source Citations** — Every answer includes page numbers so you can verify the AI isn't hallucinating
- **Model Selection** — Switch between Llama 3.1 8B and Llama 3.3 70B per query
- **Basic Security** — Rate limiting, input validation, and prompt injection filtering

---

## Tech Stack

**Frontend**
- ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) Streamlit — web UI & file upload

**Backend**
- ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) FastAPI — REST API with automatic `/docs` swagger UI
- ![Groq](https://img.shields.io/badge/Groq-F55036?style=flat&logoColor=white) Groq — LLM inference (Llama 3.1 8B, Llama 3.3 70B)
- ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black) HuggingFace — embeddings (all-MiniLM-L6-v2)
- ![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=flat&logoColor=white) Qdrant — in-memory vector database
- ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white) LangChain — RAG pipeline orchestration
- ![PyPDF](https://img.shields.io/badge/PyPDF-3776AB?style=flat&logo=python&logoColor=white) PyPDF — PDF text extraction

---

## Prerequisites

- Python 3.12 or higher
- Groq API key (free tier available at [console.groq.com](https://console.groq.com))

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
python -m uvicorn backend.main:app --reload
```

**Then start the frontend in a separate terminal:**
```bash
streamlit run app.py
```

The FastAPI backend runs on `http://localhost:8000` — visit `/docs` for the interactive API explorer.

**Workflow:**
1. Upload a PDF document via the file uploader
2. Click "Process" to chunk and index the document
3. Enter your question in natural language
4. Receive an AI-generated answer with source page citations

---

## Example Queries
- "Summarize the main findings of this research paper"
- "What technical skills are listed in this resume?"
- "List all the key recommendations from this report"
- "Explain the methodology used in section 3"

---

## Configuration

Adjustable parameters in the sidebar:

- **Model Selection** — Llama 3.1 8B (fastest) or Llama 3.3 70B (smarter)
- **Chunk Size** — 500–2000 characters (default: 1000)
- **Chunk Overlap** — 50–500 characters (default: 200)
- **Top-K Retrieval** — 1–6 document chunks per query (default: 3)

---

## Project Structure

```
rag-document-qa-system/
├── backend/
│   ├── main.py            # FastAPI routes, rate limiting, input validation
│   ├── rag_service.py     # Core RAG pipeline — embeddings, vector store, LLM
│   ├── requirements.txt   # Backend dependencies
│   └── .python-version    # Pins Python 3.12 for Render
├── app.py                 # Streamlit frontend
├── requirements.txt       # Frontend dependencies
├── .env                   # Environment variables (not tracked)
├── .gitignore
└── README.md
```

---

## Architecture

```
User
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
 ▼
RAG Service (rag_service.py)
 ├── PyPDF — extract text from PDF
 ├── RecursiveCharacterTextSplitter — chunk documents
 ├── SentenceTransformer — embed chunks (all-MiniLM-L6-v2)
 ├── Qdrant (in-memory) — store and search vectors
 └── Groq LLM — generate answer from retrieved context
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Check API status and Groq key validity |
| `POST` | `/upload` | Upload and process a PDF (rate limited: 10/min) |
| `POST` | `/query` | Ask a question against the indexed document (rate limited: 20/min) |
| `POST` | `/reset` | Clear the current session and vector store (rate limited: 10/min) |

Visit `/docs` on the backend URL for the full interactive API explorer.

---

## Security

- API keys are never committed to version control
- Rate limiting on all endpoints (10–20 req/min per IP)
- Server-side input validation on all parameters
- Regex-based prompt injection filtering
- PDF magic bytes verification
- All processing is in-memory with no persistent storage

> **Note:** This is a single-user system — the backend maintains one shared session. Multi-user session management and LLM Guard integration are planned for a future release.

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
- [LangChain](https://langchain.com) — RAG orchestration framework
- [Groq](https://groq.com) — High-performance LLM inference
- [FastAPI](https://fastapi.tiangolo.com) — Modern Python web framework
- [Qdrant](https://qdrant.tech) — Vector similarity search engine
- [Streamlit](https://streamlit.io) — Interactive web application framework
- [HuggingFace](https://huggingface.co) — Embedding models

---

> Requires a free Groq API key for LLM inference. The free tier offers generous rate limits suitable for personal use and prototyping.

---

⭐ Star this repo if you found it helpful!