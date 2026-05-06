from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import tempfile
import os
from rag_service import RAGService, check_groq
from stt_service import transcribe_audio  # Handles audio → text transcription via faster-whisper

# Rate Limiter Setup
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Querify API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501",
                   "https://rag-document-app-system-kc9htnpnjbzbexjdk4ws98.streamlit.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGService()

MAX_SIZE        = 5 * 1024 * 1024   # 5MB
MAX_Q_LENGTH    = 500               # Max question characters
MAX_CHUNK_SIZE  = 2000              # Match slider max
MIN_CHUNK_SIZE  = 500               # Match slider min
MAX_AUDIO_SIZE  = 10 * 1024 * 1024  # 10MB audio cap — keeps transcription snappy on free tier

# Allowed audio MIME types — covers browser recordings (webm) and common uploads
ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/webm", "audio/ogg",
    "audio/mp4", "audio/m4a",
}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "groq_ready": check_groq()
    }


@app.post("/upload")
@limiter.limit("10/minute")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    # Validate content type
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files allowed.")

    # Validate chunk params
    if not (MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE):
        raise HTTPException(400, f"chunk_size must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}.")
    if not (50 <= chunk_overlap <= 500):
        raise HTTPException(400, "chunk_overlap must be between 50 and 500.")
    if chunk_overlap >= chunk_size:
        raise HTTPException(400, "chunk_overlap must be less than chunk_size.")

    data = await file.read()

    if len(data) > MAX_SIZE:
        raise HTTPException(400, "File too large (max 5MB).")

    # Validate it's actually a PDF by checking magic bytes
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "Invalid PDF file.")

    # Safe temp file — no race condition
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = rag.process_pdf(tmp_path, chunk_size, chunk_overlap)
    finally:
        os.unlink(tmp_path)  # Always clean up even if processing fails

    if not result["success"]:
        raise HTTPException(422, result["message"])

    return result


@app.post("/query")
@limiter.limit("20/minute")
async def query(
    request: Request,
    question: str = Query(...),
    k: int = Query(default=3),
    model: str = Query(default=None)
):
    # Input length validation
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    if len(question) > MAX_Q_LENGTH:
        raise HTTPException(400, f"Question too long (max {MAX_Q_LENGTH} characters).")

    # Validate k
    if not (1 <= k <= 6):
        raise HTTPException(400, "k must be between 1 and 6.")

    # Validate model if provided
    allowed_models = {"llama-3.1-8b-instant", "llama-3.3-70b-versatile"}
    if model and model not in allowed_models:
        raise HTTPException(400, f"Invalid model. Choose from: {allowed_models}")

    result = rag.query(question, k, model_name=model)

    if not result["success"]:
        raise HTTPException(422, result["message"])

    return result


@app.post("/transcribe")
@limiter.limit("10/minute")
async def transcribe(
    request: Request,
    file: UploadFile = File(...)
):
    # Reject unsupported audio formats upfront
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(400, f"Unsupported audio format: {file.content_type}")

    data = await file.read()

    if len(data) > MAX_AUDIO_SIZE:
        raise HTTPException(400, "Audio file too large (max 10MB).")

    if len(data) == 0:
        raise HTTPException(400, "Audio file is empty.")

    # Derive file extension from MIME type for the temp file faster-whisper will read
    ext_map = {
        "audio/wav": "wav", "audio/wave": "wav", "audio/x-wav": "wav",
        "audio/mpeg": "mp3", "audio/mp3": "mp3",
        "audio/webm": "webm", "audio/ogg": "ogg",
        "audio/mp4": "m4a", "audio/m4a": "m4a",
    }
    ext = ext_map.get(file.content_type, "wav")

    result = transcribe_audio(data, file_extension=ext)

    if not result["success"]:
        raise HTTPException(422, result["message"])

    return result

@app.post("/reset")
@limiter.limit("10/minute")
async def reset(request: Request):
    rag.clear()
    return {"success": True, "message": "Session reset."}
