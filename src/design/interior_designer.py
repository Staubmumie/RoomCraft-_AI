"""Interior Designer agent: creates and revises the working layout and decides what happens next.

Before changing the room for a redesign, the LLM writes a functional plan (requirements, zones,
anchors, piece relationships, composition) in the same structured answer, ahead of the placements.
The plan is kept in state["design_plan"] and shown again in later steps together with deterministic
plan-vs-layout hints (pieces in no zone, unplaced or unknown items, missing essentials).

Each invocation is one step: the LLM may move, add (from the inventory) and remove furniture,
which gives a new version of the temporary `proposed_layout` (never the confirmed room), and picks
a next action:

    validate | request_furniture | request_critique | finish

Screenshots are not an action: the validator sends a valid, substantially changed layout through
render_scene before the designer's next step (design.working_layout.render_due).

What may be added or removed is set per request (state["options"]): adding inventory pieces needs
allow_new_furniture, removing pieces that were already in the room needs allow_removal; pieces added
during the run can always be removed again. The agent is resumed after every other node it called
and reads their results from the state. Deterministic guards enforce these rules and the limits in
design.routing, and never let the run finish with an unvalidated layout.
"""
import json
import math
from typing import Literal

from pydantic import BaseModel, Field

import inventory
from design.layout_validator import correction_messages
from furniture.store import get_store
from design.routing import limit, remaining
from design.scene_images import image_content, load_scene_images
from design.state import DesignState, Furniture, add_counts, model_for
from design.working_layout import (STALE_RENDER_VERSIONS, added_pieces, changes_since_render, current_version,
                                   has_proposal, is_substantial, layout_furniture, removed_ids)
from llm import LLMError, structured_completion
from observability import progress

Action = Literal["validate", "request_furniture", "request_critique", "finish"]
RepairStrategy = Literal["none", "shift", "rotate", "relocate", "regroup", "remove"]


Y_DESCRIPTION = ("Height of the piece's bottom above the floor in metres: 0 on the floor (also on a rug), "
                 "the top of the supporting piece when standing on it, the mounting height when wall-mounted")


class PlacementProposal(BaseModel):
    id: str = Field(description="Id of a piece in the working layout, copied exactly")
    x: float = Field(description="Centre X in metres")
    y: float = Field(description=Y_DESCRIPTION)
    z: float = Field(description="Centre Z in metres")
    rotation_deg: float = Field(description="Rotation around the vertical axis in degrees")


class NewPiece(BaseModel):
    item_id: str = Field(description="item_id from the available inventory, copied exactly")
    x: float
    y: float = Field(description=Y_DESCRIPTION)
    z: float
    rotation_deg: float


class FurnitureRequest(BaseModel):
    description: str = Field(description="What is missing, e.g. 'small side table next to the sofa'")
    style: str | None
    size: str | None = Field(description="Size constraints in metres, if relevant")
    colors_materials: str | None
    budget: str | None


class Zone(BaseModel):
    name: str = Field(description="Short name, e.g. 'kitchen', 'dining', 'sofa corner', 'desk'")
    purpose: str = Field(description="Activities this zone supports and which requirement(s) it serves")
    area: str = Field(description="Where in the room, grounded in the room size, e.g. 'along the back wall, x -2.5..0'")
    anchor: str | None = Field(description="Id (or item_id for a piece still to add) of the piece the zone is arranged "
                                           "around, e.g. the dining table; null if the zone has none")
    pieces: list[str] = Field(description="Ids of working-layout pieces (or item_ids to add) belonging to this zone, incl. the anchor")
    arrangement: str = Field(description="How the pieces relate, e.g. 'four chairs around the table, two per long side, facing it'")
    missing: list[str] = Field(description="Essential pieces this zone still lacks (neither in the room nor in the inventory)")


class DesignPlan(BaseModel):
    requirements: list[str] = Field(description="Functions the room must support: explicit wishes of the request first, "
                                                "then existing uses of the room that should be kept")
    zones: list[Zone] = Field(description="Functional zones; only those the request and room actually need")
    composition: str = Field(description="How the zones sit relative to each other, main walkways, free areas and visual balance")


class FurnitureGap(BaseModel):
    item: str = Field(description="The missing piece, e.g. 'kitchen base cabinets with worktop', 'coffee table'")
    essential: bool = Field(description="true if an explicit wish or a core function of the room cannot work without it; "
                                        "false for an optional improvement")
    reason: str = Field(description="Which requirement or function it serves")


class DesignerDecision(BaseModel):
    plan: DesignPlan | None = Field(description="Your functional plan, made BEFORE placing anything. Required in your first "
                                                "step of a redesign and whenever the concept changes; null for small local "
                                                "edits (e.g. moving one piece) or when your stored plan still holds")
    placements: list[PlacementProposal] = Field(description="Pieces to move in this step; empty if nothing moves")
    add: list[NewPiece] = Field(description="Inventory items to place as new pieces; empty unless adding is allowed")
    remove: list[str] = Field(description="Ids of pieces to take out of the room; empty unless removing is allowed")
    furniture_gaps: list[FurnitureGap] = Field(description="Completeness check of the layout AFTER this step's changes: "
                                                           "pieces still missing that neither the room nor the available "
                                                           "inventory provides; empty if nothing is missing")
    repair_strategy: RepairStrategy = Field(
        description="How this step resolves the unresolved validation errors: none (no errors to repair), shift (move "
                    "pieces by at least the required distance), rotate, relocate (move pieces to another free spot), "
                    "regroup (rearrange a whole furniture group), remove (take out an optional piece)")
    next_action: Action
    finish_when_valid: bool = Field(description="With next_action=validate: finish right away if the layout passes")
    furniture_request: FurnitureRequest | None = Field(description="Required for next_action=request_furniture, else null")
    explanation: str = Field(description="2-4 sentences for the user on the design idea or on what changed, in the "
                                         "language of the request; name any wish that could not be fulfilled and why. "
                                         "Never offer to request furniture later (the user cannot answer during the run); "
                                         "do not mention validation, renders or other workflow steps")


SYSTEM_PROMPT = """You are an experienced interior designer. You arrange a room so the result is functional and fits the user's wishes. Bring your own taste and creativity: asymmetry and unusual compositions are welcome as long as the room stays usable.

Design principles:
- Function before style. Explicit wishes ("with a kitchen", "seating for four") are requirements; style words ("modern", "minimalist", "spacious", "cosy") guide the look, the amount of furniture and the feel of the space. They never justify removing pieces a required function needs.
- Think in functional groups, not single pieces. Each group has an anchor (dining table, sofa, bed, desk, kitchen run) and supporting pieces arranged around it: chairs around and facing their table, sofa and armchairs facing a shared focus with a coffee table in reach, cabinets in a continuous run along a wall forming one kitchen, a lamp or side table next to the seat it serves.
- Every piece needs a purpose and a place in a group or a deliberate standalone role (e.g. a plant balancing a corner). Avoid stray pieces in the middle of nowhere.
- Keep what a function needs. Remove a piece only if removing is allowed and it has no role in the plan, or a better replacement takes its role.
- Balance open space, circulation, access and looks: clear walkways between zones and to the doors, groups that read as distinct areas, no wall crammed while another stays empty without reason.
- Never invent items, and never silently drop a wish.

Missing furniture (check before every decision to finish, report in furniture_gaps):
- Essential: an explicit wish or a core function cannot work without it (a requested kitchen without kitchen furniture, a dining area without a table). Optional: the room works, the piece would only improve it (a coffee table for an otherwise complete sofa group, a plant, a lamp). A functional, coherent room does not need every furniture category.
- Resolve in this order: 1) a suitable piece already in the room; 2) a suitable item from the available inventory, added directly if adding is allowed; 3) if adding is allowed and research budget is left: request_furniture now. Furniture research starts without asking the user; the user still approves every download, so do not hold back for that.
- Request research for an essential gap. For an optional gap, request it only if it clearly improves the result for this request and research is still unused; otherwise skip it.
- Only when 1-3 are impossible (adding not allowed, no research left, or research found nothing) finish without it, and say in the explanation what is missing and why. Never finish with an offer like "I can request X if you want": the user cannot reply during the run, so either request it now or leave it.

Planning (the plan field, filled before placements):
- Before the first change of a redesign, plan: the requirements, the functional zones they need, the essential pieces and anchor of each zone, how the pieces of a zone relate, and where the zones sit in this room (use the real dimensions, walls and the pieces and inventory you actually have). Then derive placements from the plan, zone by zone, anchor first.
- Not every room needs predefined zones; plan only what the request and the room call for.
- Your plan is stored and shown to you in later steps. Repeat it only when you change the concept; otherwise send null.
- For small, local requests (move or turn one piece, a minor tweak) do not replan the room: send plan null and change only what was asked, keeping the piece's relationship to its group.

Coordinate system (metres):
- The room is a rectangle centred on (0, 0). X runs from -width/2 to +width/2, Z from -depth/2 to +depth/2. Y is up; the floor is y = 0, the ceiling y = room height.
- x and z are the centre of a piece's footprint; y is the height of its BOTTOM. A piece occupies the heights y to y + height ("top" in the layout lists).
- At rotation_deg = 0 a piece faces +Z. Rotation is counter-clockwise seen from above: 90 faces +X, 180 faces -Z, -90 faces -X. A piece against the back wall (z = -depth/2) facing into the room uses 0; against the left wall (x = -width/2) it uses 90.
- width is the footprint along X and depth along Z at rotation 0; at 90/-90 they swap.

Vertical placement:
- Furniture stands on the floor (y = 0) unless there is a reason for another height. Rugs lie on the floor and furniture may stand on them at y = 0.
- Small pieces may stand on a suitable larger one: y = that piece's top, centre over its footprint (TV on a sideboard or counter, lamp or decor on a table or shelf). Only put light, smaller things on top of sturdier, larger ones.
- Wall pieces (pictures, mirrors, dartboards, wall shelves, wall cabinets, wall-mounted TVs) touch a wall that exists and hang at a sensible height (pictures around eye level, wall cabinets well above the worktop). Never mount on a side without a wall. Their back touches the wall, so the centre sits half their depth in front of it: on the back wall z = -depth_room/2 + depth/2 with rotation 0, on the left wall x = -width_room/2 + depth/2 with rotation 90 (depth = the piece's depth). A centre exactly on the wall line sticks half into the wall.
- Nothing floats: a piece above the floor either stands on another piece or is mounted on a wall. Use height on purpose, not arbitrarily.

Hard constraints:
- Only use the piece ids of the working layout and the item_ids of the available inventory. Never invent furniture.
- Every piece must stay completely inside the room (also below the ceiling). Pieces must not intersect: footprints may only overlap where the height ranges do not (rug below, object on top, wall piece above). Furniture against a wall should touch it closely.
- Keep walkways of roughly 0.6 m and keep the fronts of cabinets, shelves and seats accessible.
- Respect the permissions below: add pieces only if adding is allowed; remove pieces that were in the room before only if removing is allowed. Pieces you added yourself (ids starting with "new-") may always be removed again. The same inventory item may be placed more than once.

Screenshots (when attached) show the layout version stated next to them: a top-down view (image up = back wall, image right = +x, 1 m grid, numbered markers = the "marker" numbers of the layout shown: the room list for v0, your working layout list otherwise; a piece on top of another covers it there, and markers of stacked pieces are drawn next to each other) and a perspective view from the front right. Use them for arrangement, groups, orientation, colours, materials and visual balance; use the structured numbers for exact geometry. Never add furniture you think you see in an image. When they show an older version than your working layout, the section "Screenshots vs. working layout" lists what has changed since. You cannot request screenshots: after a validation that passes, they are refreshed automatically when the layout changed substantially (pieces added or removed, several moved or turned) or the screenshots are a few versions old. Invalid layouts are not rendered; repair them from the numbers.

You work in steps. In each step you may move pieces (placements; pieces you leave out keep their position), add pieces (add) and remove pieces (remove), and choose next_action:
- validate: a deterministic check of boundaries, 3D collisions and support (no floating pieces); it may correct small errors. Set finish_when_valid=true when you are done as soon as it passes (usual case); you then get no further step, so do the completeness check first.
- request_critique: independent feedback from the interior critic on your working layout, useful once for bigger redesigns. When a final review is enabled and no critique happened yet, the critic reviews your validated layout automatically when you finish, and you get one more step to act on it.
- request_furniture: ask the furniture research agent for ONE missing piece per request (fill furniture_request with what it is for, size limits that fit the free space, style, colours). Possible when adding is allowed and research budget is left. You may place and arrange the rest of the room in the same step. Afterwards you get the next step: found or downloaded items then appear in the available inventory; place them and validate.
- finish: done. Only possible when your latest layout passed validation, or when no change is needed.
Respect the remaining budget given below and avoid repeating requests without a reason.

Repairing validation errors (when the latest validation lists unresolved errors):
- Read every error first. Each names the pieces by id, the overlap or violation in metres, its severity and repair options: moves of single pieces with direction, distance, target position and what that move alone would cause ("clear" or e.g. "hits 'Sofa'"). Options are geometric suggestions, not guaranteed solutions; you may use one, combine them or do something else.
- Move far enough. A move shorter than the stated overlap cannot resolve it; nudging a piece by a few centimetres against a 0.3 m overlap only wastes a step. Prefer an option marked clear; when moving one piece would hit another, move the other one too or choose a different direction.
- Choose the strategy by the situation (repair_strategy): shift for single small to medium overlaps with a clear option; rotate when a piece fits in its place only turned; relocate when its spot is too tight; regroup when several pieces of one group collide (e.g. chairs crowding a table): rearrange the whole group, usually with more space or fewer seats, possibly elsewhere in the room; remove only an optional piece in an overcrowded room and only if removing is allowed (pieces you added yourself may always go). Never remove a piece an explicit wish or a core function needs just to pass validation.
- Substantial changes are allowed whenever local corrections cannot produce a functional arrangement; keep the relationships and requirements of your plan.
- The repair progress section shows whether your previous attempts improved the layout. Never repeat an attempt that did not help; escalate (shift -> rotate/relocate -> regroup -> remove an optional piece)."""


def options(state: DesignState) -> dict:
    return {"allow_new_furniture": False, "allow_removal": False, "final_review": False, **(state.get("options") or {})}


def available_items() -> dict[str, dict]:
    """Inventory items that can be placed (with measured dimensions), sizes in metres."""
    items = {}
    for item in inventory.load_index().values():
        dims = item.get("dimensions")
        if not dims or (inventory.ASSETS_DIR / item["id"] / item["modelPath"]).is_file() is False:
            continue
        scale = item.get("unitScale") or 1.0
        items[item["id"]] = {"item_id": item["id"], "name": item["name"], "scale": scale,
                             "width": round(dims["width"] * scale, 3), "depth": round(dims["depth"] * scale, 3),
                             "height": round(dims["height"] * scale, 3)}
    return items


async def item_categories() -> dict[str, str]:
    """Inventory item id -> category from the furniture database (e.g. 'rug', 'lamp', 'shelf').
    Helps the designer judge what can go on top, below or on a wall; optional context only."""
    try:
        return await get_store().categories()
    except Exception as exc:  # the database is an aid here: a design must not fail without it
        progress.step("categories", f"Furniture categories unavailable ({type(exc).__name__})")
        return {}


# --- context for the LLM -----------------------------------------------------------------------

def _top(piece: Furniture) -> float | None:
    return round(piece.y + piece.height, 3) if piece.height is not None else None


def _layout_json(pieces: list[Furniture], confirmed: set[str], categories: dict[str, str]) -> list[dict]:
    return [{"marker": index + 1, "id": p.id, "name": p.name,
             **({"category": categories[p.assetId]} if p.assetId in categories else {}),
             "x": round(p.x, 3), "y": round(p.y, 3), "z": round(p.z, 3), "rotation_deg": round(math.degrees(p.rotation)),
             "width": p.width, "depth": p.depth, "height": p.height, "top": _top(p),
             **({} if p.id in confirmed else {"added_by_you": True})} for index, p in enumerate(pieces)]


MAX_LISTED_ERRORS = 12


def _option_text(o) -> str:
    t = o.target
    return (f"move {o.furnitureId} {o.move} by {o.distance:.2f} m -> (x {t['x']:.2f}, y {t['y']:.2f}, z {t['z']:.2f}): "
            f"{o.outcome}")


def _validation_text(state: DesignState) -> str | None:
    validation = state.get("validation")
    if validation is None:
        return None
    lines = [f"Validation of layout v{state.get('validated_layout_version')}: {validation.status}."]
    if validation.corrections:
        lines.append("Automatic corrections (already applied to your working layout):")
        lines += [f"- '{c.name}' ({c.furnitureId}) moved {c.distance:.3f} m ({c.reason})" for c in validation.corrections]
    if validation.errors:
        lines.append(f"Unresolved errors ({len(validation.errors)}):")
    for e in sorted(validation.errors, key=lambda e: -e.size)[:MAX_LISTED_ERRORS]:  # most severe first
        ids = ", ".join(e.furnitureIds)
        lines.append(f"- [{e.type}{', ' + e.severity if e.severity else ''}] ({ids}) {e.message}")
        lines += [f"    option: {_option_text(o)}" for o in e.options]
    if len(validation.errors) > MAX_LISTED_ERRORS:
        lines.append(f"- ... {len(validation.errors) - MAX_LISTED_ERRORS} more")
    for cluster in collision_clusters(validation.errors):
        lines.append(f"Crowded cluster: {', '.join(sorted(cluster))} collide with each other "
                     f"({sum(1 for e in validation.errors if e.type == 'collision' and set(e.furnitureIds) <= cluster)} "
                     "collisions). Moving single pieces cannot resolve it: regroup these pieces with more space, "
                     "relocate some of them, or use fewer.")
    return "\n".join(lines)


def collision_clusters(errors, min_size: int = 3) -> list[set[str]]:
    """Groups of at least `min_size` pieces connected by collisions (union-find over the pairs)."""
    parent: dict[str, str] = {}

    def root(i: str) -> str:
        while parent.setdefault(i, i) != i:
            i = parent[i]
        return i

    for e in errors:
        if e.type == "collision":
            a, b = e.furnitureIds
            parent[root(a)] = root(b)
    groups: dict[str, set[str]] = {}
    for i in parent:
        groups.setdefault(root(i), set()).add(i)
    return [g for g in groups.values() if len(g) >= min_size]


def _improved(before: dict, after: dict) -> bool:
    """Measurably better: fewer unresolved problems without more severity, or clearly less severity."""
    return (after["unresolved"] < before["unresolved"] and after["severity"] <= before["severity"] + 1e-6) \
        or after["severity"] < 0.9 * before["severity"]


def failed_repairs(history: list[dict]) -> int:
    """Consecutive validations at the end that did not improve on the one before."""
    count = 0
    for before, after in zip(reversed(history[:-1]), reversed(history[1:])):
        if after["status"] != "invalid" or _improved(before, after):
            break
        count += 1
    return count


def _moved(a: list, b: list) -> float:
    return math.dist(a[:3], b[:3])


def repair_progress_text(state: DesignState) -> str | None:
    """Deterministic comparison of the validations of this run, shown while errors remain."""
    history = state.get("repair_history") or []
    validation = state.get("validation")
    if not history or validation is None or validation.status != "invalid":
        return None
    names = {p.id: p.name for p in layout_furniture(state)} | {f.id: f.name for f in state["furniture"]}
    lines = ["Repair progress (one line per validation; severity = sum of overlaps/violations in metres):"]
    for previous, entry in zip([None, *history[:-1]], history):
        trend = "" if previous is None else (" -> improved" if _improved(previous, entry) else " -> NOT improved")
        lines.append(f"- v{entry['version']} (strategy {entry['strategy']}): {entry['status']}, {entry['unresolved']} "
                     f"unresolved, severity {entry['severity']:.2f} m, max {entry['maxDepth']:.2f} m{trend}")
    if len(history) >= 2:
        before, after = history[-2], history[-1]
        label = lambda key: " / ".join(f"'{names.get(i, i)}' ({i})" for i in key.split(":", 1)[1].split("|"))
        persistent = [k for k in after["problems"] if k in before["problems"]]
        new = [k for k in after["problems"] if k not in before["problems"]]
        for key in persistent:
            moves = ", ".join(f"'{names.get(i, i)}' {_moved(before['positions'][i], after['positions'][i]):.2f} m"
                              for i in key.split(":", 1)[1].split("|")
                              if i in before["positions"] and i in after["positions"])
            lines.append(f"- still unresolved: {key.split(':')[0]} {label(key)}: {before['problems'][key]:.2f} m before, "
                         f"{after['problems'][key]:.2f} m now" + (f"; you moved {moves} in between" if moves else ""))
        if new:
            lines.append("- new since the previous validation (caused by your last changes): "
                         + "; ".join(f"{k.split(':')[0]} {label(k)} {after['problems'][k]:.2f} m" for k in new))
    failed = failed_repairs(history)
    opts = options(state)
    if failed == 0:
        advice = ("Resolve the errors with deliberate moves of at least the stated distance; use the clear options "
                  "or move the neighbours as well.")
    elif failed == 1:
        advice = ("Your last repair did not improve the layout: do not repeat it. Rotate or relocate the affected "
                  "pieces, or rearrange their whole group with more space.")
    else:
        removable = opts["allow_removal"] or bool(added_pieces(state))
        advice = (f"{failed} repairs in a row did not improve the layout. Rethink the affected area: regroup it from "
                  "scratch with more space or fewer pieces"
                  + ("; if the room is overcrowded, remove an optional piece (never one a wish or core function needs)"
                     if removable else "") + ".")
    lines.append(f"Unsuccessful repair attempts in a row: {failed}. {advice}")
    if remaining(state, "validations") <= 1 or remaining(state, "designer_steps") <= 1:
        lines.append("Budget almost used up: prefer a robust arrangement with generous gaps over fine-tuning.")
    return "\n".join(lines)


def _critique_text(state: DesignState) -> str | None:
    critique = state.get("critique")
    if not critique:
        return None
    lines = [f"Critique of layout v{state.get('critiqued_layout_version')}: {critique.get('summary', '')}"]
    lines += [f"- issue ({i['severity']}): {i['description']}" for i in critique.get("issues", [])]
    lines += [f"- suggestion: {s}" for s in critique.get("suggestions", [])]
    reviewed = state.get("reviewed_layout")
    if reviewed and state.get("critiqued_layout_version") == reviewed["layout_version"]:
        v = reviewed["layout_version"]
        lines.append(
            f"This is the final review of your validated layout v{v}, requested automatically when you finished. "
            "Weigh its points against the request: apply the improvements that clearly make the room better (then "
            "validate with finish_when_valid=true), or finish if the layout already serves the request well. "
            f"There is no further critique. If your changes cannot be validated within the remaining budget, "
            f"layout v{v} is kept.")
    return "\n".join(lines)


def _research_text(state: DesignState) -> str | None:
    result = state.get("research_result")
    if not result or not state.get("furniture_request"):
        return None
    added = ", ".join(f"{a['name']} (item_id {a['itemId']})" for a in result.get("added", [])) or "none"
    local = ", ".join(f"{c['name']} (item_id {c['itemId']})" for c in result.get("local", [])) or "none"
    return (f"Furniture research for '{state['furniture_request'].get('description')}': {result.get('status')}. "
            f"Suitable items already in the inventory: {local}. Newly downloaded: {added}. {result.get('summary', '')}")


def plan_notes(plan: dict, pieces: list[Furniture], items: dict[str, dict]) -> list[str]:
    """Deterministic consistency hints on the stored plan against the working layout. They only point
    at gaps between plan and layout; judging the design itself stays with the designer and critic."""
    layout_ids = {p.id for p in pieces}
    placed_assets = {p.assetId for p in pieces if p.assetId}
    referenced: set[str] = set()
    notes = []
    for zone in plan.get("zones", []):
        refs = list(dict.fromkeys([*([zone["anchor"]] if zone.get("anchor") else []), *zone.get("pieces", [])]))
        referenced.update(refs)
        unknown = [r for r in refs if r not in layout_ids and r not in items]
        pending = [r for r in refs if r not in layout_ids and r in items and r not in placed_assets]
        if unknown:
            notes.append(f"zone '{zone['name']}' refers to ids that are neither in the working layout nor in the "
                         f"available inventory: {', '.join(unknown)}")
        if pending:
            notes.append(f"zone '{zone['name']}' plans inventory items that are not placed yet: {', '.join(pending)}")
        if zone.get("missing"):
            notes.append(f"zone '{zone['name']}' still lacks: {', '.join(zone['missing'])} "
                         "(add from the inventory or request research; tell the user only if neither is possible)")
    if plan.get("zones"):
        loose = [f"{p.name} ({p.id})" for p in pieces if p.id not in referenced and p.assetId not in referenced]
        if loose:
            notes.append("pieces not part of any zone (give them a role, or remove them if that is allowed): " + ", ".join(loose))
    return notes


def _plan_text(state: DesignState, items: dict[str, dict]) -> str:
    plan = state.get("design_plan")
    if not plan:
        return ("No design plan yet. Unless the request is a small local edit, fill plan first and derive "
                "your placements from it.")
    text = f"Your design plan (keep following it unless you send a new one):\n{json.dumps(plan, indent=1)}"
    notes = plan_notes(plan, layout_furniture(state), items)
    return text + ("\nPlan vs. working layout:\n" + "\n".join(f"- {n}" for n in notes) if notes else "")


def _render_text(state: DesignState) -> str | None:
    """How far the attached screenshots are behind the working layout (deterministic, no images)."""
    changes = changes_since_render(state)
    if changes is None:
        return None
    shown = state.get("rendered_layout_version", 0)
    if not any(changes.values()):
        return f"Screenshots vs. working layout: the screenshots (v{shown}) show your current working layout."
    parts = [f"{label} {len(names)} ({', '.join(names[:6])}{', ...' if len(names) > 6 else ''})"
             for label, names in (("moved/turned", changes["moved"]), ("added", changes["added"]),
                                  ("removed", changes["removed"])) if names]
    stale = current_version(state) - shown >= STALE_RENDER_VERSIONS
    verdict = ("substantial change: new screenshots follow automatically once this layout passes validation"
               if is_substantial(changes) or stale else "small change: the screenshots are still a usable reference")
    return (f"Screenshots vs. working layout: the screenshots show v{shown}, your working layout is "
            f"v{current_version(state)}. Since then: {'; '.join(parts)}. {verdict.capitalize()}.")


def research_blocker(state: DesignState) -> str | None:
    """Why the designer cannot start furniture research now (None if it can). `state` must already
    count the current designer step: a step has to be left to place what research brings back."""
    if not options(state)["allow_new_furniture"]:
        return "adding furniture is not allowed"
    if remaining(state, "research_requests") <= 0:
        return "furniture research limit reached"
    if remaining(state, "designer_steps") <= 0:
        return "no designer step left to place researched furniture"
    return None


def _research_availability(state: DesignState) -> str:
    # The prompt is built before this step is counted, so one step must remain beyond it.
    blocker = research_blocker({**state, "counters": add_counts(state.get("counters"), {"designer_steps": 1})})
    if blocker:
        return f"Furniture research: NOT available ({blocker})."
    return ("Furniture research: AVAILABLE. Use request_furniture for an important missing piece; it starts without "
            "asking the user, who approves downloads later.")


def build_messages(state: DesignState, images: list[tuple[str, str]] | None = None,
                   items: dict[str, dict] | None = None, categories: dict[str, str] | None = None) -> list[dict]:
    room = state["room"]
    opts = options(state)
    categories = categories or {}
    confirmed = {f.id for f in state["furniture"]}
    furniture = [
        {"marker": index + 1, "id": f.id, "name": f.name,
         **({"category": categories[f.assetId]} if f.assetId in categories else {}),
         "width": f.width, "depth": f.depth, "height": f.height,
         "confirmed": {"x": round(f.x, 2), "y": round(f.y, 2), "z": round(f.z, 2),
                       "rotation_deg": round(math.degrees(f.rotation))}}
        for index, f in enumerate(state["furniture"])
    ]
    context = {"room": {"width": room.width, "depth": room.depth, "height": room.height,
                        "walls": room.walls or ["all four sides"]},
               "furniture_in_the_room": furniture}
    sections = [
        f"Design request: {state['request']}",
        f"Room and furniture:\n{json.dumps(context, indent=1)}",
        "Permissions: adding furniture from the inventory is "
        + ("ALLOWED" if opts["allow_new_furniture"] else "NOT allowed")
        + "; removing furniture that was in the room is "
        + ("ALLOWED" if opts["allow_removal"] else "NOT allowed (every piece of the room must stay)") + ".",
    ]
    if opts["allow_new_furniture"]:
        listed = [{**{k: v for k, v in i.items() if k != "scale"},
                   **({"category": categories[i["item_id"]]} if i["item_id"] in categories else {})}
                  for i in (items or {}).values()]
        sections.append(f"Available inventory (metres):\n{json.dumps(listed, indent=1)}" if listed
                        else "The inventory has no placeable items.")
    if has_proposal(state):
        sections.append(f"Your working layout v{current_version(state)} (all pieces currently in the room):\n"
                        f"{json.dumps(_layout_json(layout_furniture(state), confirmed, categories), indent=1)}")
        if removed_ids(state):
            sections.append("Removed by you: " + ", ".join(removed_ids(state)))
    else:
        sections.append("You have not proposed a layout yet; the working layout is the room as it is.")
    sections.append(_plan_text(state, items or {}))
    for text in (_render_text(state), _validation_text(state), repair_progress_text(state), _critique_text(state),
                 _research_text(state)):
        if text:
            sections.append(text)
    if state.get("workflow_log"):
        sections.append("Workflow so far:\n" + "\n".join(f"- {entry}" for entry in state["workflow_log"][-12:]))
    sections.append(_research_availability(state))
    sections.append("Remaining budget: " + ", ".join(
        f"{name} {remaining(state, name)}" for name in ("designer_steps", "validations", "critiques", "research_requests")))
    content = image_content("\n\n".join(sections), images or [], state.get("rendered_layout_version", 0))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


# --- applying a decision -----------------------------------------------------------------------

def apply_changes(state: DesignState, decision: DesignerDecision, items: dict[str, dict]) -> tuple[dict, list[str]]:
    """New working layout version from the decision. Changes that break the permissions or refer to
    unknown pieces are skipped and reported back (they appear in the workflow log)."""
    opts = options(state)
    confirmed = {f.id for f in state["furniture"]}
    pieces = {p.id: p for p in layout_furniture(state)}
    version = state.get("layout_version", 0) + 1
    notes, changed = [], False

    for piece_id in decision.remove:
        if piece_id not in pieces:
            notes.append(f"remove {piece_id}: not in the working layout")
        elif piece_id in confirmed and not opts["allow_removal"]:
            notes.append(f"remove {pieces[piece_id].name}: removing furniture of the room is not allowed")
        else:
            del pieces[piece_id]
            changed = True
    for move in decision.placements:
        if move.id not in pieces:
            notes.append(f"move {move.id}: not in the working layout")
            continue
        pieces[move.id] = pieces[move.id].model_copy(update={"x": move.x, "y": move.y, "z": move.z,
                                                             "rotation": math.radians(move.rotation_deg)})
        changed = True
    for index, new in enumerate(decision.add):
        item = items.get(new.item_id)
        if not opts["allow_new_furniture"]:
            notes.append(f"add {new.item_id}: adding furniture is not allowed")
        elif item is None:
            notes.append(f"add {new.item_id}: not a placeable inventory item")
        else:
            piece_id = f"new-{version}-{index + 1}"
            pieces[piece_id] = Furniture(id=piece_id, name=item["name"], assetId=item["item_id"], width=item["width"],
                                         depth=item["depth"], height=item["height"], scale=item["scale"],
                                         x=new.x, y=new.y, z=new.z, rotation=math.radians(new.rotation_deg))
            changed = True

    if not changed:
        return {}, notes
    return {"proposed_layout": list(pieces.values()), "layout_version": version}, notes


# --- guards and endings ------------------------------------------------------------------------

def _validated(state: DesignState) -> bool:
    validation = state.get("validation")
    return (validation is not None and validation.status != "invalid"
            and state.get("validated_layout_version") == current_version(state))


def _failure_message(state: DesignState, reason: str) -> str:
    validation = state.get("validation")
    if validation is None or validation.status != "invalid":
        return f"The design could not be completed: {reason}."
    errors = "; ".join(e.message for e in validation.errors[:3])
    more = f" (+{len(validation.errors) - 3} more)" if len(validation.errors) > 3 else ""
    return f"The layout could not be validated ({reason}): {errors}{more}"


def _gap_warnings(state: DesignState) -> list[str]:
    """Essential pieces the designer reported as still missing: never dropped silently."""
    return [f"Not fulfilled: {g['item']} is missing ({g['reason']})."
            for g in state.get("furniture_gaps") or [] if g.get("essential")]


def _finish(state: DesignState, note: str) -> dict:
    """Ends the design run: with the validated working layout, or without changes."""
    if not has_proposal(state):
        return {"designer_action": "finish", "layout": [], "removed": [], "warnings": _gap_warnings(state),
                "outcome": {"status": "completed", "kind": "design", "message": "No layout changes were needed."},
                "workflow_log": [f"interior_designer: finished without changes ({note})"]}
    validation = state.get("validation")
    return {"designer_action": "finish", "layout": layout_furniture(state), "removed": removed_ids(state),
            "warnings": (correction_messages(validation) if validation else []) + _gap_warnings(state),
            "outcome": {"status": "completed", "kind": "design", "message": None},
            "workflow_log": [f"interior_designer: finished with layout v{current_version(state)} "
                             f"({len(added_pieces(state))} added, {len(removed_ids(state))} removed; {note})"]}


def _fail(state: DesignState, reason: str) -> dict:
    reviewed = state.get("reviewed_layout")
    if reviewed:
        # Changes after the final review did not work out: the reviewed layout was validated, keep it.
        restored = {**state, **reviewed, "validated_layout_version": reviewed["layout_version"]}
        update = _finish(restored, f"changes after the critique failed ({reason}); kept the reviewed layout "
                                   f"v{reviewed['layout_version']}")
        update["warnings"] = [f"The critic's suggestions could not be applied ({reason}); the validated layout "
                              "from before the critique was kept.", *update["warnings"]]
        update["validation"] = reviewed["validation"]
        return update
    return {"designer_action": "finish",
            "outcome": {"status": "failed", "kind": "design", "message": _failure_message(state, reason)},
            "workflow_log": [f"interior_designer: failed ({reason})"]}


def _safe_fallback(state: DesignState, reason: str) -> tuple[dict, str]:
    """What to do when the chosen action is not allowed: validate an unvalidated layout,
    otherwise finish. Never accepts an unvalidated layout."""
    if has_proposal(state) and not _validated(state):
        if remaining(state, "validations") > 0:
            return {"designer_action": "validate", "finish_when_valid": True}, f"{reason}; validating instead"
        return _fail(state, f"{reason}; no validations left"), reason
    return _conclude(state, reason), reason


def final_review(state: DesignState) -> dict | None:
    """Sends a validated layout to the critic once before the run ends (option final_review). Only
    when the run had no critique yet and a designer step and a validation are left to act on it, so
    it can never loop. The reviewed layout is kept as the fallback result (see _fail)."""
    if not options(state)["final_review"] or state.get("reviewed_layout"):
        return None
    if not has_proposal(state) or not _validated(state) or (state.get("counters") or {}).get("critiques", 0):
        return None
    if min(remaining(state, name) for name in ("critiques", "designer_steps", "validations")) <= 0:
        return None
    return {"designer_action": "request_critique", "return_to": {"interior_critic": "interior_designer"},
            "reviewed_layout": {"proposed_layout": layout_furniture(state), "layout_version": current_version(state),
                                "validation": state.get("validation")}}


def _conclude(state: DesignState, note: str) -> dict:
    """Ends the run, unless the final review is still due."""
    return final_review(state) or _finish(state, note)


def _start_research(request: dict) -> dict:
    # Fresh research context; the previous search (if any) is finished.
    return {"designer_action": "request_furniture", "furniture_request": {**request, "origin": "interior_designer"},
            "return_to": {"furniture_research": "interior_designer"}, "counters": {"research_requests": 1},
            "researched": [request["description"]],
            "research_round": 0, "candidates": [], "approval": None, "download_result": None, "research_result": {},
            "research_status": "", "local_candidates": [], "local_assessment": ""}


def _was_researched(state: DesignState, item: str) -> bool:
    return any(item.strip().lower() in done.lower() for done in state.get("researched") or [])


def _unresearched_essential(state: DesignState, decision: DesignerDecision) -> FurnitureGap | None:
    """An essential gap the designer wants to finish with although research could still close it.
    Gaps that were already researched in this run are not searched again (one request per gap)."""
    finishing = decision.next_action == "finish" or (decision.next_action == "validate" and decision.finish_when_valid)
    if not finishing or research_blocker(state) is not None:
        return None
    return next((g for g in decision.furniture_gaps if g.essential and not _was_researched(state, g.item)), None)


def guard_action(state: DesignState, decision: DesignerDecision) -> tuple[dict, str | None]:
    """Turns the LLM's decision into the actual next step. `state` already contains this step's
    layout change. Returns (state update, note if the decision was overridden)."""
    action = decision.next_action
    version = current_version(state)
    if gap := _unresearched_essential(state, decision):
        request = {"description": f"{gap.item} ({gap.reason})", "style": None, "size": None,
                   "colors_materials": None, "budget": None}
        if decision.furniture_request and gap.item.lower() in decision.furniture_request.description.lower():
            request = decision.furniture_request.model_dump()  # the designer's own, more detailed request
        return _start_research(request), f"essential furniture missing: {gap.item}"
    if action == "validate":
        if not has_proposal(state):
            return _finish(state, "nothing to validate"), "validate without a layout"
        if remaining(state, "validations") <= 0:
            return _fail(state, "validation limit reached"), "validation limit reached"
        return {"designer_action": "validate", "finish_when_valid": decision.finish_when_valid}, None
    if action == "request_critique":
        if remaining(state, "critiques") <= 0:
            return _safe_fallback(state, "critique limit reached")
        if state.get("critiqued_layout_version") == version:
            return _safe_fallback(state, f"layout v{version} was already critiqued")
        return {"designer_action": action, "return_to": {"interior_critic": "interior_designer"}}, None
    if action == "request_furniture":
        if blocker := research_blocker(state):
            return _safe_fallback(state, blocker)
        if decision.furniture_request is None:
            return _safe_fallback(state, "furniture request without details")
        return _start_research(decision.furniture_request.model_dump()), None
    # finish
    if has_proposal(state) and not _validated(state):
        return _safe_fallback(state, "finish with an unvalidated layout")
    if review := final_review(state):
        return review, "final critique before finishing"
    return _finish(state, "designer finished"), None


# --- node --------------------------------------------------------------------------------------

async def interior_designer(state: DesignState) -> dict:
    # Committed earlier: "finish as soon as it validates". No LLM call needed for that.
    if state.get("designer_action") == "validate" and state.get("finish_when_valid") and _validated(state):
        if review := final_review(state):
            progress.step("review", f"Validated layout v{current_version(state)} goes to the critic before finishing")
            return {**review, "workflow_log": [f"interior_designer: layout v{current_version(state)} validated, "
                                               "final critique before finishing"]}
        progress.step("accept", f"Validated layout v{current_version(state)} accepted")
        return _finish(state, "validated layout accepted")
    if remaining(state, "designer_steps") <= 0:
        progress.step("limit", "Designer step limit reached", limit=limit(state, "designer_steps"))
        return _finish(state, "step limit reached") if not has_proposal(state) or _validated(state) \
            else _fail(state, "step limit reached")

    items = available_items() if options(state)["allow_new_furniture"] else {}
    categories = await item_categories()
    model = model_for(state, "interior_designer")
    images = await load_scene_images(state, "Interior Designer", model)
    with progress.stage("prepare", "Prepare design context") as details:
        messages = build_messages(state, images, items, categories)
        details.update(pieces=len(layout_furniture(state)), layoutVersion=current_version(state),
                       inventoryItems=len(items), options=options(state), images=[view for view, _ in images],
                       hasValidation=state.get("validation") is not None, hasCritique=bool(state.get("critique")))

    progress.step("send", "Send request to OpenRouter", model=model)
    with progress.stage("llm", "Wait for LLM response") as details:
        try:
            response = await structured_completion(model, messages, DesignerDecision)
        except LLMError as exc:
            if exc.info:
                progress.emit("llm", **exc.info.as_dict())
            raise
        progress.emit("llm", **response.info.as_dict())
        details.update(tokens=(response.info.usage or {}).get("totalTokens"), cost=response.info.cost)
    decision = response.parsed

    layout_update, notes = apply_changes(state, decision, items)
    update: dict = {"counters": {"designer_steps": 1}, "explanation": decision.explanation.strip(),
                    "furniture_gaps": [g.model_dump() for g in decision.furniture_gaps],
                    "repair_strategy": decision.repair_strategy, **layout_update}
    if decision.plan is not None:
        update["design_plan"] = decision.plan.model_dump()
    # State as it will be after this step (counters through their reducer, like LangGraph does).
    after = {**state, **update, "counters": add_counts(state.get("counters"), update["counters"])}
    action_update, override = guard_action(after, decision)
    update.update({k: v for k, v in action_update.items() if k not in ("counters", "workflow_log")})
    if "counters" in action_update:
        update["counters"] = {**update["counters"], **action_update["counters"]}

    chosen = update["designer_action"]
    counts = [f"{n} {label}" for n, label in ((len(decision.placements), "moved"), (len(decision.add), "added"),
                                              (len(decision.remove), "removed")) if n]
    if decision.plan is not None:
        counts.insert(0, f"planned {len(decision.plan.zones)} zone(s)")
    entry = (f"interior_designer: {', '.join(counts) + ', ' if counts else ''}"
             f"layout v{current_version(after)}, chose {decision.next_action}"
             + (f" (repair: {decision.repair_strategy})" if decision.repair_strategy != "none" else ""))
    if override:
        entry += f" -> {chosen} ({override})"
        progress.step("guard", f"Action {decision.next_action} replaced by {chosen}", reason=override)
    for note in notes:
        progress.step("skipped", f"Change skipped: {note}")
    update["workflow_log"] = [entry, *(f"interior_designer: skipped {note}" for note in notes),
                              *action_update.get("workflow_log", [])]
    progress.step("decision", f"Next: {chosen}", moved=len(decision.placements), added=len(decision.add),
                  removed=len(decision.remove), finishWhenValid=decision.finish_when_valid,
                  layoutVersion=current_version(after))
    return update
