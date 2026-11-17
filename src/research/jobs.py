"""Run-scoped background jobs of the furniture research: per-item searches and asset downloads.

Why not plain asyncio tasks inside a node: human_approval pauses the run with interrupt(), which
ends the graph task; LangGraph re-runs the interrupted node on resume and knows nothing about
tasks the previous node execution left behind. So:

- The graph state (`research_items`) stays the source of truth. Each item records the key of the
  job it waits for (`job`, or a download entry's asset id); only graph nodes change items, one at a
  time, so there are no concurrent state updates.
- Jobs live here, keyed by run id (= LangGraph thread id) and job key. A key runs at most once at
  a time and its result is kept until the run ends, so a resumed coordinator finds results that
  arrived during the pause, in any order, and merging the same result twice is a no-op (the item
  has moved on). Downloads are keyed by asset id: two items choosing the same asset share one download.
- A job whose task disappeared (cancelled together with its event loop, or never started in this
  process) is "lost"; the coordinator starts it again under the same key. Searches have no side
  effects and downloads are idempotent (inventory.acquire_asset reuses an asset that is already there).
- Tasks run in an empty contextvars.Context, so they never write into the stream of a node
  execution that has ended; their progress events are buffered and replayed by the coordinator.
- close(run_id) (abort, end of the run) cancels every task and refuses new jobs: nothing starts
  after an abort, and late results are dropped together with the registry.
"""
import asyncio
import contextvars
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from langgraph.config import get_config

from design.routing import PARALLEL
from observability import progress

ADHOC_RUN = "_adhoc"  # nodes called outside a graph run (tests); job keys are unique per list anyway


@dataclass
class Job:
    task: asyncio.Task
    events: list[dict] = field(default_factory=list)


class RunJobs:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def _semaphore(self, kind: str) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if loop is not self._loop:  # semaphores belong to one event loop
            self._loop, self._semaphores = loop, {}
        if kind not in self._semaphores:
            self._semaphores[kind] = asyncio.Semaphore(PARALLEL[kind])
        return self._semaphores[kind]

    def state(self, key: str) -> str:
        """missing | running | done | lost"""
        job = self.jobs.get(key)
        if job is None:
            return "missing"
        if not job.task.done():
            return "running"
        return "lost" if job.task.cancelled() else "done"

    def start(self, key: str, kind: str, factory: Callable[[], Awaitable]) -> bool:
        """Starts `factory()` as job `key` (bounded by PARALLEL[kind]) unless it is running or done.
        Returns False when nothing was started (duplicate, or the run is closed)."""
        if self.closed or self.state(key) in ("running", "done"):
            return False
        events: list[dict] = []
        semaphore = self._semaphore(kind)

        async def body():
            async with semaphore:
                if self.closed:  # aborted while waiting for a free slot
                    raise asyncio.CancelledError()
                progress.capture(events)
                return await factory()

        task = asyncio.create_task(body(), name=f"research:{key}", context=contextvars.Context())
        self.jobs[key] = Job(task, events)
        return True

    def result(self, key: str):
        """(value, None) or (None, exception) of a finished job."""
        task = self.jobs[key].task
        error = task.exception()
        return (None, error) if error else (task.result(), None)

    def take_events(self) -> list[dict]:
        """Progress events buffered by all jobs since the last call."""
        events = []
        for job in self.jobs.values():
            events += job.events
            job.events.clear()
        return events

    async def wait(self, keys: list[str], timeout: float | None = None) -> None:
        """Until one of the running jobs `keys` has finished (or `timeout` seconds passed)."""
        tasks = [self.jobs[k].task for k in keys if self.state(k) == "running"]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)

    def close(self) -> None:
        self.closed = True
        for job in self.jobs.values():
            job.task.cancel()


_runs: dict[str, RunJobs] = {}


def current_run_id() -> str:
    """The LangGraph thread of the running node (= the API's run id)."""
    try:
        return get_config()["configurable"].get("thread_id") or ADHOC_RUN
    except RuntimeError:
        return ADHOC_RUN


def for_run(run_id: str | None = None) -> RunJobs:
    run_id = run_id or current_run_id()
    return _runs.setdefault(run_id, RunJobs())


def close(run_id: str) -> None:
    """Cancels the run's jobs and forgets them; a later for_run(run_id) starts empty."""
    jobs = _runs.pop(run_id, None)
    if jobs:
        jobs.close()


def active(run_id: str) -> bool:
    jobs = _runs.get(run_id)
    return bool(jobs) and any(jobs.state(k) == "running" for k in jobs.jobs)
