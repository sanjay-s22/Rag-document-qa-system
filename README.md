# RAG Document QA System

AI-powered PDF document analyzer using Retrieval-Augmented Generation (RAG)

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)

Live App: https://rag-document-app-system-kc9htnpnjbzbexjdk4ws98.streamlit.app/

## Features

- **Low-Latency QA: Handles documents up to 200MB with near-instant responses by offloading inference to Groq.**
- **Smart Context: Uses recursive character splitting so chunks actually make sense.**
- **Semantic Search: Qdrant-powered vector search for finding meaning, not just keywords.**
- **Fact Checking: Every answer includes page citations so you can verify the AI isn't hallucinating.**
- **Configurable Settings: Sidebar controls for chunk size and overlap to handle different document densities.**


## Tech Stack

**Interface**
- ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) Streamlit — web UI & file upload

**AI & Models**
- ![Groq](https://img.shields.io/badge/Groq-F55036?style=flat&logoColor=white) Groq — LLM inference (Llama 3.1, Mixtral, Gemma2)
- ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black) HuggingFace — embeddings (all-MiniLM-L6-v2)

**Data & Retrieval**
- ![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=flat&logoColor=white) Qdrant — in-memory vector database
- ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white) LangChain — RAG pipeline orchestration
- ![PyPDF](https://img.shields.io/badge/PyPDF-3776AB?style=flat&logo=python&logoColor=white) PyPDF — PDF text extraction

## Prerequisites

- Python 3.12 or higher
- Groq API key (free tier available at [console.groq.com](https://console.groq.com))

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
GROQ_API_KEY=gsk_your_api_key_here  #Get yours free at console.groq.com
```

## Usage

**Running locally:**
```bash
streamlit run app.py
```

**Workflow:**
1. Enter your Groq API key in the sidebar (or load from `.env`)
2. Upload a PDF document via the file uploader
3. Click "Process" to chunk and index the document
4. Enter questions in natural language
5. Receive AI-generated answers with source page citations

## Example Queries
- "Summarize the main findings of this research paper"
- "What technical skills are listed in this resume?"
- "List all the key recommendations from this report"
- "Explain the methodology used in section 3"

## Configuration

Adjustable parameters in the sidebar:

- **Model Selection**: Choose from Llama 3.1 (8B/70B), Mixtral 8x7B, or Gemma2 9B
- **Chunk Size**: 500-2000 characters (default: 1000)
- **Chunk Overlap**: 50-500 characters (default: 200)  
- **Top-K Retrieval**: 1-5 document chunks per query (default: 3)

## Project Structure

```
rag-document-qa-system/
├── app.py                 # Streamlit user interface
├── rag_analyzer.py        # Core RAG pipeline logic
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not tracked)
├── .gitignore             # Git exclusion rules
└── README.md              # Project documentation(you are here)
```

## Security

- API keys are never committed to version control
- Streamlit Cloud uses encrypted secrets management
- All document processing occurs in-memory with no persistent storage
- Session data is cleared on app restart

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

**Sanjay**  
GitHub: [@sanjay-s22](https://github.com/sanjay-s22)

## Acknowledgments

Built with:
- [LangChain](https://langchain.com) - RAG orchestration framework
- [Groq](https://groq.com) - High-performance LLM inference
- [Qdrant](https://qdrant.tech) — Vector similarity search engine
- [Streamlit](https://streamlit.io) - Interactive web application framework
- [HuggingFace](https://huggingface.co) - Embedding models

---

> **Note:** Requires a free Groq API key for LLM inference. The free tier offers generous rate limits suitable for personal use and prototyping.

---

⭐ Star this repo if you found it helpful!
