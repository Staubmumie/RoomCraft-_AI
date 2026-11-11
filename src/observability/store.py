"""Workflow runs and their events. In memory for now; a persistent store only needs the same methods."""
import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone

from observability.serialize import to_jsonable

TERMINAL_EVENTS = {"workflow.completed", "workflow.failed", "workflow.cancelled"}
KEEPALIVE_SECONDS = 15


@dataclass
class Run:
    id: str
    workflow: str
    created_at: float
    status: str = "pending"
    events: list[dict] = field(default_factory=list)
    result: dict | None = None
    task: asyncio.Task | None = None
    listeners: set[asyncio.Queue] = field(default_factory=set)
    pending_approval: dict | None = None  # what the paused run waits for (human-in-the-loop)
    context: dict = field(default_factory=dict)  # workflow-specific bookkeeping across pause/resume

    @property
    def finished(self) -> bool:
        return bool(self.events) and self.events[-1]["type"] in TERMINAL_EVENTS

    def summary(self) -> dict:
        return {"runId": self.id, "workflow": self.workflow, "status": self.status,
                "createdAt": self.created_at, "eventCount": len(self.events),
                "pendingApproval": self.pending_approval}


class RunStore:
    def __init__(self, max_runs: int = 50):
        self.max_runs = max_runs
        self._runs: dict[str, Run] = {}

    def create(self, workflow: str) -> Run:
        finished = [r for r in self._runs.values() if r.finished]
        for old in sorted(finished, key=lambda r: r.created_at)[: max(len(self._runs) + 1 - self.max_runs, 0)]:
            del self._runs[old.id]
        run = Run(id=uuid.uuid4().hex, workflow=workflow, created_at=time.time() * 1000)
        self._runs[run.id] = run
        return run

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def list(self) -> list[Run]:
        return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def emit(self, run: Run, type: str, *, node: str | None = None, execution_id: str | None = None,
             status: str | None = None, payload: dict | None = None) -> dict:
        now = time.time()
        event = {
            "seq": len(run.events) + 1,
            "runId": run.id,
            "type": type,
            "node": node,
            "executionId": execution_id,
            "status": status,
            "ts": now * 1000,
            "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "payload": to_jsonable(payload or {}),
        }
        run.events.append(event)
        if type.startswith("workflow.") and status:
            run.status = status
        for queue in run.listeners:
            queue.put_nowait(event)
        return event

    async def subscribe(self, run: Run, after_seq: int = 0) -> AsyncIterator[dict | None]:
        """Replays events after `after_seq`, then follows live ones until the run ends.
        Yields None as a keep-alive when nothing happened for a while."""
        queue: asyncio.Queue = asyncio.Queue()
        run.listeners.add(queue)
        try:
            # No await between registering and copying, so no event is missed or sent twice.
            for event in list(run.events[after_seq:]):
                yield event
                if event["type"] in TERMINAL_EVENTS:
                    return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), KEEPALIVE_SECONDS)
                except TimeoutError:
                    yield None
                    continue
                if event["seq"] <= after_seq:
                    continue
                yield event
                if event["type"] in TERMINAL_EVENTS:
                    return
        finally:
            run.listeners.discard(queue)
