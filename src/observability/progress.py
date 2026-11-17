"""Progress events from inside a graph node, sent through LangGraph's `custom` stream.

Each event carries the id of the node execution (LangGraph task id) so repeated or
parallel executions of the same node never get mixed up. Outside a graph run
(e.g. in tests) all calls are no-ops.

Background jobs of the furniture research (research.jobs) run outside any node execution. They
`capture` their events into a buffer; the research coordinator `replay`s them in its own execution.
"""
import time
from contextlib import contextmanager
from contextvars import ContextVar

from langgraph.config import get_config, get_stream_writer

_buffer: ContextVar[list | None] = ContextVar("progress_buffer", default=None)


def capture(buffer: list) -> None:
    """Collect this context's events in `buffer` instead of writing them to a stream."""
    _buffer.set(buffer)


def replay(events: list[dict]) -> None:
    """Emit buffered events as events of the current node execution."""
    writer, execution_id = _context()
    if writer:
        for event in events:
            writer({**event, "executionId": execution_id})


def _context():
    buffer = _buffer.get()
    if buffer is not None:
        return buffer.append, None
    try:
        writer = get_stream_writer()
        namespace = get_config()["metadata"].get("langgraph_checkpoint_ns", "")
    except RuntimeError:
        return None, None
    # "node:<task id>", nested graphs append "|child:<task id>"
    execution_id = namespace.split("|")[-1].partition(":")[2] or None
    return writer, execution_id


def emit(kind: str, **data):
    writer, execution_id = _context()
    if writer:
        writer({"kind": kind, "executionId": execution_id, **data})


def step(name: str, label: str, **data):
    """A milestone without duration."""
    emit("stage", stage=name, label=label, status="completed", durationMs=0, data=data)


@contextmanager
def stage(name: str, label: str):
    """Times a stage of the node. Put details for the inspector into the yielded dict."""
    details: dict = {}
    started = time.perf_counter()
    emit("stage", stage=name, label=label, status="running")
    try:
        yield details
    except Exception as exc:
        emit("stage", stage=name, label=label, status="failed",
             durationMs=round((time.perf_counter() - started) * 1000), error=str(exc), data=details)
        raise
    emit("stage", stage=name, label=label, status="completed",
         durationMs=round((time.perf_counter() - started) * 1000), data=details)
