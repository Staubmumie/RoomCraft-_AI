import asyncio
import json
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from design.graph import design_graph
from design.runner import cancel_waiting_run, execute_design_run, resume_design_run, settle_cancelled_task
from design.routing import LIMITS
from design.state import Furniture, Room
from llm import AGENT_MODELS, AVAILABLE_MODELS, DEFAULT_MODEL, HELPER_MODEL
from observability.graph_view import describe_graph
from observability.store import RunStore
from research.human_approval import ApprovalError, validate_decision

router = APIRouter(prefix="/api/design", tags=["design"])
runs = RunStore()


class DesignOptions(BaseModel):
    allowNewFurniture: bool = Field(default=False, description="The designer may place inventory items")
    allowRemoval: bool = Field(default=False, description="The designer may remove furniture of the room")
    maxIterations: int = Field(default=LIMITS["designer_steps"], ge=1, le=15,
                               description="Max. designer steps (LLM calls of the interior designer)")
    maxResearch: int = Field(default=LIMITS["research_requests"], ge=0, le=5,
                             description="Max. furniture lists of the designer that go online (up to 6 pieces each)")
    finalReview: bool = Field(default=True, description="The critic reviews the validated layout once before finishing")
    searchOnline: bool = Field(default=True, description="Furniture may be searched online (MCP) and downloaded")
    reviewSuggestions: bool = Field(default=True, description="The user picks from up to four suggestions per piece; "
                                                              "off: the best suitable match is downloaded automatically")


class DesignRequest(BaseModel):
    request: str = Field(min_length=1, max_length=2000)
    model: str | None = Field(default=None, max_length=100, description="The interior designer's model")
    # Model per agent (keys of AGENT_MODELS); agents left out use their default (see agent_models).
    models: dict[str, str] = Field(default_factory=dict)
    room: Room
    furniture: list[Furniture] = Field(max_length=60)  # may be empty when new furniture is allowed
    options: DesignOptions = Field(default_factory=DesignOptions)

    @field_validator("models")
    @classmethod
    def _known_agents(cls, models: dict[str, str]) -> dict[str, str]:
        unknown = set(models) - set(AGENT_MODELS)
        if unknown:
            raise ValueError(f"Unknown agents: {', '.join(sorted(unknown))}")
        if any(len(m) > 100 for m in models.values()):
            raise ValueError("Model ids are limited to 100 characters")
        return {agent: m.strip() for agent, m in models.items() if m and m.strip()}


def default_model(agent: str) -> str:
    return DEFAULT_MODEL if agent == "interior_designer" else HELPER_MODEL


def agent_models(body: DesignRequest) -> dict[str, str]:
    """Every agent's model: the request's choice, else `model` for the designer and the small helper
    model for the other agents."""
    chosen = {"interior_designer": body.model, **body.models} if body.model else body.models
    return {agent: chosen.get(agent) or default_model(agent) for agent in AGENT_MODELS}


def _start_task(run, coroutine) -> None:
    """Runs a workflow coroutine as the run's task. A cancelled task always ends the run exactly once
    and its state is discarded, even when it was cancelled before it started."""
    run.task = asyncio.create_task(coroutine)

    def done(task: asyncio.Task):
        if task.cancelled():
            settle_cancelled_task(runs, run)

    run.task.add_done_callback(done)


def _get_run(run_id: str):
    run = runs.get(run_id)
    if not run:
        raise HTTPException(404, "Unknown workflow run.")
    return run


@router.get("/models")
async def list_models():
    return {"default": DEFAULT_MODEL, "models": AVAILABLE_MODELS,
            "agents": [{"id": agent, "label": label, "default": default_model(agent)}
                       for agent, label in AGENT_MODELS.items()]}


@router.get("/graph")
async def get_graph():
    return describe_graph(design_graph)


@router.post("/runs", status_code=202)
async def start_run(body: DesignRequest):
    """Starts the workflow in the background. Progress and result arrive via the events stream,
    so a dropped connection never starts or repeats a run."""
    if len({f.id for f in body.furniture}) != len(body.furniture):
        raise HTTPException(422, "Furniture ids must be unique.")
    run = runs.create("design")
    models = agent_models(body)
    inputs = {
        "request": body.request.strip(),
        "model": models["interior_designer"],
        "models": models,
        "room": body.room,
        "furniture": body.furniture,
        "options": {
            "allow_new_furniture": body.options.allowNewFurniture,
            "allow_removal": body.options.allowRemoval,
            "final_review": body.options.finalReview,
            "search_online": body.options.searchOnline,
            # Only meaningful with online search; stored explicitly, so automatic downloads always
            # trace back to this option.
            "review_suggestions": body.options.reviewSuggestions or not body.options.searchOnline,
            "limits": {"designer_steps": body.options.maxIterations,
                       "research_requests": body.options.maxResearch},
        },
    }
    _start_task(run, execute_design_run(runs, run, inputs))
    return {"runId": run.id}


@router.get("/runs")
async def list_runs():
    return [run.summary() for run in runs.list()]


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    run = _get_run(run_id)
    return {**run.summary(), "result": run.result, "events": run.events}


@router.get("/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request, after: int = 0):
    """Server-Sent Events. Reconnecting clients send Last-Event-ID and only get what they missed."""
    run = _get_run(run_id)
    last_id = request.headers.get("last-event-id")
    after_seq = int(last_id) if last_id and last_id.isdigit() else after

    async def event_stream():
        yield "retry: 2000\n\n"
        async for event in runs.subscribe(run, after_seq):
            if event is None:
                yield ": keep-alive\n\n"
            else:
                yield f"id: {event['seq']}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class ApprovalDecision(BaseModel):
    itemKey: str = Field(min_length=1, max_length=40, description="The furniture item the decision is for")
    run: int | None = Field(default=None, ge=1, description="Search run of the item the user saw (optional)")
    action: Literal["approve", "reject", "skip"]
    selected: list[str] = Field(default_factory=list, max_length=10)
    feedback: str = Field(default="", max_length=1000)


@router.post("/runs/{run_id}/approval", status_code=202)
async def answer_approval(run_id: str, body: ApprovalDecision):
    """Resumes a run paused for approval of one furniture item (same run and LangGraph thread).
    Decisions for another item or an earlier search run are refused (409); selected ids must be
    among this item's candidates (422); only those are downloaded."""
    run = _get_run(run_id)
    if run.status != "waiting_for_approval" or not run.pending_approval:
        raise HTTPException(409, "This run is not waiting for an approval.")
    if body.itemKey != run.pending_approval.get("itemKey") or body.run not in (None, run.pending_approval.get("run")):
        raise HTTPException(409, "This decision is outdated: another furniture item is waiting for approval.")
    try:
        decision = validate_decision(run.pending_approval, body.model_dump())
    except ApprovalError as exc:
        raise HTTPException(422, str(exc))
    # Claim the approval before anything awaits, so a second click cannot resume twice.
    run.pending_approval = None
    run.status = "resuming"
    _start_task(run, resume_design_run(runs, run, decision))
    return {"runId": run.id, "status": "resuming"}


@router.delete("/runs/{run_id}")
async def cancel_run(run_id: str):
    """Aborts a run for good (no pause): it is never resumed and nothing of it is applied.
    Safe to call twice, after the run ended and while it is resuming."""
    run = _get_run(run_id)
    if run.finished or run.status == "cancelling":
        return {"runId": run.id, "status": run.status}
    if run.status == "waiting_for_approval":
        cancel_waiting_run(runs, run)
    elif run.task and not run.task.done():
        run.status = "cancelling"  # the task emits workflow.cancelled once the cancellation arrives
        run.task.cancel()
    return {"runId": run.id, "status": run.status}
