"""Audio -> text transcription via Groq's Whisper-compatible endpoint."""

from fastapi import HTTPException, status

from src.app.core.config import get_settings
from src.app.services.groq_client import get_groq_client, translate_groq_error


async def transcribe_audio(audio_bytes: bytes, filename: str, language: str | None) -> str:
    settings = get_settings()
    client = get_groq_client()

    kwargs: dict[str, object] = {
        "file": (filename, audio_bytes),
        "model": settings.groq_transcription_model,
        "response_format": "json",
    }
    if language:
        kwargs["language"] = language

    try:
        response = await client.audio.transcriptions.create(**kwargs)
    except Exception as exc:
        raise translate_groq_error(exc) from exc

    text = (response.text or "").strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio transcription produced empty text.",
        )
    return text
