import threading
from typing import Any, Optional

_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def _empty_state() -> dict[str, Any]:
    """Empty or reset the state of the job.

    Returns:
        dict[str, Any]: A job status.
    """
    return {
        "status": "idle",
        "progress": None,
        "message": "",
        "history": [],
        "error": None,
    }


def start_job(job_id: str, message: str = "") -> None:
    """Mark a job as running and reset its previous state.

    Args:
        job_id (str): The job ID.
        message (str): The message of the job.
    """
    with _LOCK:
        state = _empty_state()
        state["status"] = "running"
        state["progress"] = 0.0
        state["message"] = message
        _JOBS[job_id] = state


def update_job(
    job_id: str,
    progress: Optional[float] = None,
    message: Optional[str] = None,
) -> None:
    """Update the progress fraction and/or status message of a running job.

    Args:
        job_id (str): The job ID.
        progress (Optional[float]): The new progress fraction.
        message (Optional[str]): The new message.
    """
    with _LOCK:
        state = _JOBS.setdefault(job_id, _empty_state())
        if progress is not None:
            state["progress"] = max(0.0, min(1.0, float(progress)))
        if message is not None:
            state["message"] = message


def append_history(job_id: str, entry: dict[str, Any]) -> None:
    """Append a per-step metric record (used to draw live training curves).

    Args:
        job_id (str): The job ID.
        entry (dict[str, Any]): A per-step metric record.
    """
    with _LOCK:
        state = _JOBS.setdefault(job_id, _empty_state())
        state["history"].append(entry)


def complete_job(job_id: str, message: str = "") -> None:
    """Mark a job as completed successfully.

    Args:
        job_id (str): The job ID.
        message (str): The message of the job.
    """
    with _LOCK:
        state = _JOBS.setdefault(job_id, _empty_state())
        state["status"] = "completed"
        state["progress"] = 1.0
        if message:
            state["message"] = message


def fail_job(job_id: str, error: str) -> None:
    """Mark a job as failed and record the error message.

    Args:
        job_id (str): The job ID.
        error (str): The error message.
    """
    with _LOCK:
        state = _JOBS.setdefault(job_id, _empty_state())
        state["status"] = "failed"
        state["error"] = error
        state["message"] = "Job failed."


def get_job(job_id: str) -> dict[str, Any]:
    """Return a snapshot copy of a job's state.

    Args:
        job_id (str): The job ID.

    Returns:
        dict[str, Any]: A job status.
    """
    with _LOCK:
        state = _JOBS.get(job_id)
        if state is None:
            return _empty_state()
        snapshot = dict(state)
        snapshot["history"] = list(state["history"])
        return snapshot
