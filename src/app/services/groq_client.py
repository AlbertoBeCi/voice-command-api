"""Shared Groq client and error translation for the instruction/transcription services."""

from functools import lru_cache

from fastapi import HTTPException, status
from groq import AsyncGroq, APITimeoutError, GroqError

from src.app.core.config import get_settings


@lru_cache
def get_groq_client() -> AsyncGroq:
    settings = get_settings()
    return AsyncGroq(api_key=settings.groq_api_key, timeout=settings.request_timeout_seconds)


def translate_groq_error(exc: Exception) -> HTTPException:
    """Map a Groq SDK exception to an appropriate HTTP error for our API."""
    if isinstance(exc, APITimeoutError):
        return HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Groq did not respond in time.",
        )
    if isinstance(exc, GroqError):
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Groq request failed: {exc}",
        )
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Unexpected error while calling Groq: {exc}",
    )
