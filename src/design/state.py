"""Room/furniture data and the LangGraph state shared by all nodes.

Coordinates match the frontend: the room is centred on the origin, X runs along the
width, Z along the depth, Y is up. Positions are metres, `rotation` is the Y rotation
in radians (the frontend's `instance.rotation[1]`). x/z are the centre of the footprint,
y is the height of the BOTTOM of the bounding box above the floor (0 = on the floor; the
frontend shifts every model so its lowest point sits at the instance position). Data
without y stays valid and means "on the floor".
"""
import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


class Room(BaseModel):
    width: float = Field(gt=0, description="Extent along X in metres")
    depth: float = Field(gt=0, description="Extent along Z in metres")
    height: float = Field(gt=0)
    walls: list[str] = Field(default_factory=list, description="Sides with a wall, e.g. 'back (z = -3)'")


class Furniture(BaseModel):
    """A placed piece of furniture: a scene instance plus its inventory metadata."""
    id: str = Field(description="Instance id from the scene")
    name: str
    assetId: str | None = None
    width: float | None = Field(default=None, description="Footprint along X at rotation 0, metres (scaled)")
    depth: float | None = Field(default=None, description="Footprint along Z at rotation 0, metres (scaled)")
    height: float | None = None
    scale: float | None = Field(default=None, description="Uniform model scale in the scene (for rendering)")
    x: float
    z: float
    y: float = Field(default=0.0, description="Height of the bottom of the piece above the floor, metres")
    rotation: float = 0.0


class Placement(BaseModel):
    id: str
    x: float
    z: float
    rotation: float
    y: float = 0.0


# --- reducers: nodes return only what they change -------------------------------------------

def add_counts(current: dict[str, int] | None, update: dict[str, int]) -> dict[str, int]:
    """Counters are incremented: a node returns {"validations": 1}, not the full dict."""
    merged = dict(current or {})
    for key, value in update.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def merge(current: dict | None, update: dict) -> dict:
    """Keys are set individually, so setting one return destination keeps the others."""
    return {**(current or {}), **update}


def model_for(state: dict, agent: str) -> str:
    """The model an agent (graph node name) runs on: its own choice, otherwise the run's main model."""
    return (state.get("models") or {}).get(agent) or state["model"]


class DesignState(TypedDict, total=False):
    # --- Input: set once by the API, never changed by nodes --------------------------------
    request: str                    # the user's original request, passed on unchanged
    model: str                      # the interior designer's model; fallback for agents missing in `models`
    models: dict[str, str]          # agent node name -> model id (see model_for)
    room: Room
    furniture: list[Furniture]      # the confirmed room; agents work on proposed_layout instead
    preferences: dict               # TODO: user preferences / budget from long-term memory
    # per request: allow_new_furniture, allow_removal, final_review, limits {"designer_steps": n, "research_requests": n}
    options: dict

    # --- Control flow ------------------------------------------------------------------------
    intent: str                     # DESIGN_ROOM | CRITIQUE_ROOM | SEARCH_FURNITURE | REJECT
    security_flags: list[str]       # why a request looked like prompt injection
    # Shared service nodes (render_scene, interior_critic, furniture_research) return to the node
    # stored here; a missing entry means "called by the orchestrator" -> END when finished.
    return_to: Annotated[dict[str, str], merge]
    counters: Annotated[dict[str, int], add_counts]  # loop protection, see design.routing.LIMITS
    workflow_log: Annotated[list[str], operator.add]  # short history the agents can read
    outcome: dict                   # final status: {"status": completed|failed|rejected, "kind", "message"}

    # --- Render Scene ------------------------------------------------------------------------
    # {"top_view": "renders/runs/<run_id>/<folder>/top.png", ...}: paths only, no image data
    scene_images: dict[str, str]
    render_trigger: str             # why render_scene runs next: initial | validation | critic (folder name)
    rendered_layout_version: int    # 0 = confirmed room, n = proposed layout version n
    rendered_layout: list[Furniture]  # the pieces the screenshots show (to report changes since then)

    # --- Interior Designer ---------------------------------------------------------------------
    designer_action: str            # validate | request_furniture | request_critique | finish
    finish_when_valid: bool         # designer's decision to finish as soon as its layout validates
    # Complete working layout: every piece in the proposed room (moved, kept, newly added from the
    # inventory); confirmed pieces that are missing were removed. Temporary, not applied to the room.
    proposed_layout: list[Furniture]
    layout_version: int             # bumped whenever proposed_layout changes
    explanation: str
    # Functional plan (requirements, zones, composition) from the designer's latest planning step;
    # shown to the designer in later steps and to the critic. Not applied anywhere.
    design_plan: dict
    # Completeness check of the designer's latest step: [{"item", "essential", "reason"}]. Essential
    # gaps trigger research when still possible, otherwise they become warnings of the result.
    furniture_gaps: list[dict]
    researched: Annotated[list[str], operator.add]  # descriptions of all furniture requests of this run

    # --- Layout Validator ----------------------------------------------------------------------
    validation: Any                 # design.validator.ValidationResult
    validated_layout_version: int
    # Set by every validation: render the validated layout before the designer continues
    # (design.working_layout.render_due); read by routing.after_validator.
    render_after_validation: bool
    # One entry per validation: {"version", "status", "strategy", "unresolved", "severity", "maxDepth",
    # "problems": {key: size}, "positions": {id: [x, y, z, rotation_deg]}}; see layout_validator.history_entry
    repair_history: Annotated[list[dict], operator.add]
    repair_strategy: str            # the designer's repair approach of its latest step (none, shift, ...)

    # --- Interior Critic -----------------------------------------------------------------------
    critic_action: str              # request_render | done
    critique: dict
    critiqued_layout_version: int
    # Set when the designer's validated layout went to the final review (options["final_review"]):
    # {"proposed_layout", "layout_version", "validation"}. It is the result if later changes fail.
    reviewed_layout: dict

    # --- Furniture Research ----------------------------------------------------------------------
    furniture_request: dict         # structured request from the designer; absent for direct searches
    research_round: int
    research_status: str            # awaiting_approval | done
    research_summary: str
    candidates: list[dict]          # external shortlist awaiting the user's choice: metadata and image URLs only
    local_candidates: list[dict]    # suitable items already in the furniture database (no download needed)
    local_assessment: str           # the agent's reason why local items were (not) enough
    rejected_asset_ids: list[str]   # not proposed again
    approval: dict | None           # {"action": approve|reject|retry, "selected": [...], "feedback": str}
    download_result: dict | None    # {"added": [...], "existing": [...], "failed": [...]}
    research_result: dict           # final result handed to the caller (designer or user)

    # --- Final result of a design run ----------------------------------------------------------
    layout: list[Furniture]         # validated layout the frontend applies (all pieces, incl. new ones)
    removed: list[str]              # confirmed pieces the validated layout no longer contains
    warnings: list[str]
    message: str
