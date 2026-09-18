"""Executes a validated `InstructionPayload` against the in-memory task store.

This is technical HTTP dispatch (method + endpoint -> handler function), not
natural-language intent detection -- the actual intent resolution happens
entirely inside `instruction_service` via Groq.
"""

from typing import Any

from fastapi import HTTPException, status
from pydantic import ValidationError

from src.app.schemas.voice import InstructionPayload, TaskCreate, TaskReplace, TaskUpdate
from src.app.services.actions import UnknownActionError, resolve_action
from src.app.services.task_store import TaskNotFoundError, task_store


def dispatch_instruction(instruction: InstructionPayload) -> Any:
    try:
        action = resolve_action(instruction.method, instruction.endpoint)
    except UnknownActionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unsupported action returned by routing: {exc}",
        ) from exc

    try:
        if action.method == "GET":
            return [task.model_dump() for task in task_store.list_tasks()]

        if action.method == "POST":
            payload = TaskCreate.model_validate(instruction.params)
            return task_store.create_task(payload).model_dump()

        if action.method == "PUT":
            payload = TaskReplace.model_validate(instruction.params)
            return task_store.replace_task(action.task_id, payload).model_dump()

        if action.method == "PATCH":
            payload = TaskUpdate.model_validate(instruction.params)
            return task_store.update_task(action.task_id, payload).model_dump()

        if action.method == "DELETE":
            task_store.delete_task(action.task_id)
            return {"detail": f"Task {action.task_id} deleted"}

    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Routing params do not match the expected schema: {exc}",
        ) from exc
    except TaskNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {exc.task_id} not found",
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Unsupported method '{action.method}' returned by routing",
    )
