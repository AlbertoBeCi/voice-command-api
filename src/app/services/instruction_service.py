"""Turns a plain-text transcription into a validated `InstructionPayload` using
Groq as the sole decision-maker (no hardcoded keyword/intent matching).

This is the single routing implementation shared by `POST /instruction`
(routing only) and `POST /transcribe` (routing + execution).
"""

import json

from fastapi import HTTPException, status
from pydantic import ValidationError

from src.app.core.config import get_settings
from src.app.schemas.voice import InstructionPayload, Task
from src.app.services.actions import UnknownActionError, resolve_action, validate_params
from src.app.services.groq_client import get_groq_client, translate_groq_error
from src.app.services.task_store import TaskNotFoundError, task_store

_MAX_ATTEMPTS = 2

_SYSTEM_PROMPT_TEMPLATE = """You are the routing engine for a voice-controlled task manager API.
Given the user's transcribed voice command and the CURRENT TASKS list, output exactly
one JSON object describing the single REST action needed, and nothing else.

Supported actions:
- List tasks:              {{"endpoint": "/tasks", "method": "GET", "params": {{}}}}
- Create a task:            {{"endpoint": "/tasks", "method": "POST", "params": {{"title": string, "done"?: boolean}}}}
- Replace a task entirely:  {{"endpoint": "/tasks/<id>", "method": "PUT", "params": {{"title": string, "done": boolean}}}}
- Update a task partially:  {{"endpoint": "/tasks/<id>", "method": "PATCH", "params": {{"title"?: string, "done"?: boolean}}}}
- Delete a task:            {{"endpoint": "/tasks/<id>", "method": "DELETE", "params": {{}}}}

Rules:
- Replace <id> with the REAL numeric id of the task referenced by the user, matched
  against CURRENT TASKS by title. Never leave a placeholder like {{task_id}}.
- If the command is ambiguous or no matching task is found, respond with
  {{"endpoint": "/tasks", "method": "GET", "params": {{}}}}.
- Respond with ONLY the JSON object -- no markdown, no prose, no code fences.

CURRENT TASKS:
{current_tasks}
"""


def _build_system_prompt(tasks: list[Task]) -> str:
    current_tasks = json.dumps([t.model_dump() for t in tasks])
    return _SYSTEM_PROMPT_TEMPLATE.format(current_tasks=current_tasks)


async def _call_groq_chat(messages: list[dict[str, str]]) -> str:
    settings = get_settings()
    client = get_groq_client()
    try:
        response = await client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=300,
        )
    except Exception as exc:  # groq SDK errors + anything unexpected
        raise translate_groq_error(exc) from exc
    content = response.choices[0].message.content
    if not content:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Groq returned an empty routing response.",
        )
    return content


def _validate_structure(payload: InstructionPayload) -> None:
    """Raise on a malformed/unknown action. Retryable errors are plain
    exceptions; a resolved-but-missing task_id raises HTTPException(404)
    directly, since retrying the LLM cannot conjure a task into existence.
    """
    try:
        action = resolve_action(payload.method, payload.endpoint)
        validate_params(action, payload.params)
    except (UnknownActionError, ValidationError) as exc:
        raise ValueError(str(exc)) from exc

    if action.task_id is not None:
        try:
            task_store.get_task(action.task_id)
        except TaskNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {action.task_id} not found",
            ) from exc


async def route_instruction(transcription: str) -> InstructionPayload:
    tasks = task_store.list_tasks()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _build_system_prompt(tasks)},
        {"role": "user", "content": transcription},
    ]

    last_error: str | None = None
    for _ in range(_MAX_ATTEMPTS):
        if last_error:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid: {last_error}. "
                        "Reply again with ONLY valid JSON."
                    ),
                }
            )

        raw = await _call_groq_chat(messages)
        try:
            payload = InstructionPayload.model_validate_json(raw)
            _validate_structure(payload)
        except HTTPException:
            raise
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

        return payload

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Groq did not return a valid routing instruction after {_MAX_ATTEMPTS} attempts: {last_error}",
    )
