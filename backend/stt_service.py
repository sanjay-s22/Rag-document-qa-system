import os
import tempfile
from groq import Groq

# This offloads transcription entirely to Groq's servers — zero RAM cost on Render's free tier.
# Model: whisper-large-v3-turbo 
_client = None

def get_groq_client():
    # Lazy-init the Groq client — only created on first transcription request
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_STT_KEY"))
    return _client


def transcribe_audio(audio_bytes: bytes, file_extension: str = "wav") -> dict:
    """
    Takes raw audio bytes from the frontend and returns the transcribed text.
    Sends the audio to Groq's Whisper API — supports wav, mp3, webm, ogg, m4a.
    """
    # Groq's API needs a file-like object with a name — write to temp file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        client = get_groq_client()
        with open(tmp_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                response_format="text" ,  # Returns plain string, no extra parsing needed
                language="en"
            )
        # When response_format="text", Groq returns the transcript directly as a string
        transcript = response.strip() if isinstance(response, str) else response.text.strip()
        return {"success": True, "transcript": transcript, "language": "auto"}
    except Exception as e:
        return {"success": False, "message": f"Transcription failed: {str(e)}"}
    finally:
        os.unlink(tmp_path) 