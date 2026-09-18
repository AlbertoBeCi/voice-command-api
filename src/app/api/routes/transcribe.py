import json

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from src.app.schemas.voice import InstructionRequest, TranscribeFlowResponse
from src.app.services.dispatcher import dispatch_instruction
from src.app.services.instruction_service import route_instruction
from src.app.services.transcription_service import transcribe_audio
from src.app.utils.language import normalize_transcription_language

router = APIRouter(tags=["transcribe"])


@router.get("/")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/transcribe", response_model=TranscribeFlowResponse)
async def transcribe_and_run_flow(request: Request) -> TranscribeFlowResponse:
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()

    if content_type == "multipart/form-data":
        form = await request.form()
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'file' in multipart form data.",
            )
        language = normalize_transcription_language(form.get("language"))
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio file.",
            )
        transcription = await transcribe_audio(audio_bytes, file.filename or "command.webm", language)

    elif content_type == "application/json":
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON body.",
            ) from exc
        try:
            data = InstructionRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Field 'transcription' is required.",
            ) from exc
        transcription = data.transcription.strip()
        if not transcription:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Field 'transcription' cannot be empty.",
            )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported content type: expected multipart/form-data or application/json.",
        )

    instruction = await route_instruction(transcription)
    result = dispatch_instruction(instruction)
    return TranscribeFlowResponse(transcription=transcription, instruction=instruction, result=result)
