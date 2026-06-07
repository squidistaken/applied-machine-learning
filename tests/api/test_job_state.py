import pytest

from src.api import job_state


@pytest.fixture(autouse=True)
def clear_jobs() -> None:
    """Reset the in-memory job registry before each test."""
    job_state._JOBS.clear()


def test_get_unknown_job_returns_idle() -> None:
    """Test that an unknown job id reports a default idle state."""
    state = job_state.get_job("missing")

    assert state["status"] == "idle"
    assert state["progress"] is None
    assert state["history"] == []
    assert state["error"] is None


def test_start_job_initialises_running_state() -> None:
    """Test that starting a job marks it running with zero progress."""
    job_state.start_job("cnn", "Initialising...")

    state = job_state.get_job("cnn")

    assert state["status"] == "running"
    assert state["progress"] == 0.0
    assert state["message"] == "Initialising..."


def test_start_job_resets_previous_state() -> None:
    """Test that restarting a job clears its prior history and error."""
    job_state.start_job("cnn")
    job_state.append_history("cnn", {"epoch": 1})
    job_state.fail_job("cnn", "boom")

    job_state.start_job("cnn")

    state = job_state.get_job("cnn")
    assert state["status"] == "running"
    assert state["history"] == []
    assert state["error"] is None


def test_update_job_clamps_progress() -> None:
    """Test that out-of-range progress values are clamped to [0, 1]."""
    job_state.start_job("cnn")

    job_state.update_job("cnn", progress=1.5)
    assert job_state.get_job("cnn")["progress"] == 1.0

    job_state.update_job("cnn", progress=-0.5)
    assert job_state.get_job("cnn")["progress"] == 0.0


def test_update_job_message_only() -> None:
    """Test that updating the message leaves the progress untouched."""
    job_state.start_job("cnn")
    job_state.update_job("cnn", progress=0.5)

    job_state.update_job("cnn", message="Epoch 1/2")

    state = job_state.get_job("cnn")
    assert state["progress"] == 0.5
    assert state["message"] == "Epoch 1/2"


def test_append_history_accumulates() -> None:
    """Test that history entries are appended in order."""
    job_state.start_job("cnn")

    job_state.append_history("cnn", {"epoch": 1})
    job_state.append_history("cnn", {"epoch": 2})

    history = job_state.get_job("cnn")["history"]
    assert history == [{"epoch": 1}, {"epoch": 2}]


def test_complete_job_sets_full_progress() -> None:
    """Test that completing a job marks it done with full progress."""
    job_state.start_job("cnn")

    job_state.complete_job("cnn", "Done.")

    state = job_state.get_job("cnn")
    assert state["status"] == "completed"
    assert state["progress"] == 1.0
    assert state["message"] == "Done."


def test_fail_job_records_error() -> None:
    """Test that failing a job records its status and error message."""
    job_state.start_job("cnn")

    job_state.fail_job("cnn", "out of memory")

    state = job_state.get_job("cnn")
    assert state["status"] == "failed"
    assert state["error"] == "out of memory"


def test_get_job_returns_isolated_snapshot() -> None:
    """Test that a returned snapshot does not mutate the stored state."""
    job_state.start_job("cnn")
    job_state.append_history("cnn", {"epoch": 1})

    snapshot = job_state.get_job("cnn")
    snapshot["history"].append({"epoch": 99})
    snapshot["status"] = "tampered"

    state = job_state.get_job("cnn")
    assert state["history"] == [{"epoch": 1}]
    assert state["status"] == "running"
