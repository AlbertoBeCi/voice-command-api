"""In-memory task storage. The single source of truth for task state, used by
both the direct REST endpoints (`api/routes/tasks.py`) and the voice dispatcher
(`services/dispatcher.py`) so CRUD logic is never duplicated.
"""

import threading

from src.app.schemas.voice import Task, TaskCreate, TaskReplace, TaskUpdate


class TaskNotFoundError(Exception):
    def __init__(self, task_id: int) -> None:
        self.task_id = task_id
        super().__init__(f"Task {task_id} not found")


class TaskStore:
    def __init__(self) -> None:
        self._tasks: dict[int, Task] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def list_tasks(self) -> list[Task]:
        with self._lock:
            return list(self._tasks.values())

    def get_task(self, task_id: int) -> Task:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise TaskNotFoundError(task_id)
            return task

    def create_task(self, payload: TaskCreate) -> Task:
        with self._lock:
            task = Task(id=self._next_id, title=payload.title, done=payload.done)
            self._tasks[task.id] = task
            self._next_id += 1
            return task

    def replace_task(self, task_id: int, payload: TaskReplace) -> Task:
        with self._lock:
            if task_id not in self._tasks:
                raise TaskNotFoundError(task_id)
            task = Task(id=task_id, title=payload.title, done=payload.done)
            self._tasks[task_id] = task
            return task

    def update_task(self, task_id: int, payload: TaskUpdate) -> Task:
        with self._lock:
            existing = self._tasks.get(task_id)
            if existing is None:
                raise TaskNotFoundError(task_id)
            changes = payload.model_dump(exclude_unset=True)
            updated = existing.model_copy(update=changes)
            self._tasks[task_id] = updated
            return updated

    def delete_task(self, task_id: int) -> None:
        with self._lock:
            if task_id not in self._tasks:
                raise TaskNotFoundError(task_id)
            del self._tasks[task_id]


task_store = TaskStore()
