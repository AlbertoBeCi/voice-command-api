"""Single source of truth mapping (method, endpoint) -> supported task action.

Used by both `instruction_service` (to validate/retry the LLM's routing output)
and `dispatcher` (to execute it), so the set of supported actions and their
expected param shapes only exist in one place.
"""

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from src.app.schemas.voice import TaskCreate, TaskReplace, TaskUpdate

_TASK_ITEM_RE = re.compile(r"^/tasks/(?P<task_id>\d+)$")


@dataclass(frozen=True)
class ResolvedAction:
    method: str
    task_id: int | None
    param_model: type[BaseModel] | None


class UnknownActionError(Exception):
    """Raised when (method, endpoint) does not match any supported action."""


def resolve_action(method: str, endpoint: str) -> ResolvedAction:
    method = method.upper()

    if endpoint == "/tasks":
        if method == "GET":
            return ResolvedAction(method="GET", task_id=None, param_model=None)
        if method == "POST":
            return ResolvedAction(method="POST", task_id=None, param_model=TaskCreate)
        raise UnknownActionError(f"Unsupported method '{method}' for endpoint '{endpoint}'")

    match = _TASK_ITEM_RE.fullmatch(endpoint)
    if match:
        task_id = int(match.group("task_id"))
        if method == "PUT":
            return ResolvedAction(method="PUT", task_id=task_id, param_model=TaskReplace)
        if method == "PATCH":
            return ResolvedAction(method="PATCH", task_id=task_id, param_model=TaskUpdate)
        if method == "DELETE":
            return ResolvedAction(method="DELETE", task_id=task_id, param_model=None)
        raise UnknownActionError(f"Unsupported method '{method}' for endpoint '{endpoint}'")

    raise UnknownActionError(f"Unknown endpoint '{endpoint}'")


def validate_params(action: ResolvedAction, params: dict[str, Any]) -> BaseModel | None:
    """Validate `params` against the action's expected schema, if any."""
    if action.param_model is None:
        return None
    return action.param_model.model_validate(params)
