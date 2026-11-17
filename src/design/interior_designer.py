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
from design.routing import MAX_ITEMS_PER_REQUEST, limit, remaining
from design.scene_images import image_content, load_scene_images
from design.state import DesignState, Furniture, add_counts, model_for
from design.working_layout import (STALE_RENDER_VERSIONS, added_pieces, changes_since_render, current_version,
                                   has_proposal, is_substantial, layout_furniture, removed_ids)
from furniture.store import get_store
from llm import LLMError, structured_completion
from observability import progress
from prompts import load_prompt

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


class FurnitureItemRequest(BaseModel):
    description: str = Field(description="One missing piece, e.g. 'small side table next to the sofa'")
    purpose: str | None = Field(description="Zone / function it serves, e.g. 'sofa corner: put down drinks'")
    size: str | None = Field(description="Size limits in metres that fit the free space, if relevant")
    style: str | None
    colors_materials: str | None
    essential: bool = Field(description="true if a wish or core function of the room cannot work without it")


class FurnitureRequest(BaseModel):
    items: list[FurnitureItemRequest] = Field(description="Every piece still missing, essential ones first")
    style: str | None = Field(description="Style for the whole list, if any")
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
    zones: list[Zone] = Field(description="Functional zones the request and the room call for; in a spacious "
                                                 "room also a zone that suits it and uses free space")
    composition: str = Field(description="How the zones are distributed over the room and sit relative to each other, "
                                         "main walkways, the larger open areas and what each is for, visual balance")


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


SYSTEM_PROMPT = load_prompt("interior_designer_system")


def options(state: DesignState) -> dict:
    return {"allow_new_furniture": False, "allow_removal": False, "final_review": False, "search_online": True,
            "review_suggestions": True, **(state.get("options") or {})}


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
    if scores := critique.get("scores"):
        lines.append("Scores (1-5): " + ", ".join(f"{name} {value}" for name, value in scores.items()))
    lines += [f"- issue ({', '.join(filter(None, (i['severity'], i.get('aspect'))))}): {i['description']}"
              for i in critique.get("issues", [])]
    lines += [f"- suggestion: {s}" for s in critique.get("suggestions", [])]
    if state.get("critic_research_bonus"):
        lines.append("Extra furniture research was granted for this critique: use it for pieces that resolve one of "
                     "its concrete points and that the inventory does not have.")
    reviewed = state.get("reviewed_layout")
    if reviewed and state.get("critiqued_layout_version") == reviewed["layout_version"]:
        v = reviewed["layout_version"]
        lines.append(
            f"This is the final review of your validated layout v{v}, requested automatically when you finished. "
            "Weigh its points against the request: apply the improvements that clearly make the room better (then "
            "validate with finish_when_valid=true), or finish if the layout already serves the request well. "
            "For points that need furniture the inventory lacks, request_furniture is an option while furniture "
            "research is available (see below); place the found pieces in the following step. "
            f"There is no further critique. If your changes cannot be validated within the remaining budget, "
            f"layout v{v} is kept.")
    return "\n".join(lines)


def _research_item_text(entry: dict) -> str:
    ids = lambda items: ", ".join(f"{i['name']} (item_id {i['itemId']})" for i in items)
    status = entry["status"]
    if status == "found_locally":
        text = f"already in the inventory: {ids(entry['local'])}"
    elif status in ("completed", "failed"):
        parts = []
        if entry["added"]:
            parts.append(("chosen automatically and downloaded: " if entry.get("auto") else "found online and downloaded: ")
                         + ids(entry["added"]))
        if entry["existing"]:
            parts.append(f"already in the inventory: {ids(entry['existing'])}")
        if entry["failed"]:
            parts.append("download failed: " + ", ".join(f["name"] for f in entry["failed"]))
        text = "; ".join(parts) or (entry.get("note") or "failed")
    else:
        text = {"skipped": "skipped by the user (do not request it again)",
                "rejected": f"all suggestions rejected by the user after {entry['runs']} search runs (do not request it again)",
                "not_found": f"not found ({entry.get('note') or 'nothing suitable'})"}.get(status, status)
    return f"- {entry['description']}{' [essential]' if entry.get('essential') else ''}: {text}"


def _research_text(state: DesignState) -> str | None:
    result = state.get("research_result")
    if not result or not result.get("items") or not state.get("furniture_request"):
        return None
    lines = ["Result of your latest furniture list (place the available pieces now, all in this step):"]
    lines += [_research_item_text(entry) for entry in result["items"]]
    return "\n".join(lines)


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
    if not options(state)["search_online"]:
        return "online furniture search is switched off; use the inventory"
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
    lists = remaining(state, "research_requests")
    return (f"Furniture research: AVAILABLE for {lists} more list(s) of up to {MAX_ITEMS_PER_REQUEST} pieces each. "
            "Use request_furniture with one list of all missing pieces; it starts without asking the user.")


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


def _start_research(items: list[dict], shared: dict | None = None) -> dict:
    """A fresh furniture list; the previous list (if any) is finished. The research counts the
    budget itself once the list really goes online (research_requests)."""
    return {"designer_action": "request_furniture",
            "furniture_request": {"items": items, **(shared or {}), "origin": "interior_designer"},
            "return_to": {"furniture_research": "interior_designer"},
            "researched": [item["description"] for item in items],
            "research_items": [], "research_status": "", "research_result": {}, "approval": None,
            "approval_item": None, "download_result": None}


def _was_researched(state: DesignState, item: str) -> bool:
    return any(item.strip().lower() in done.lower() for done in state.get("researched") or [])


def _gap_item(gap: FurnitureGap) -> dict:
    return {"description": f"{gap.item} ({gap.reason})", "purpose": gap.reason, "size": None, "style": None,
            "colors_materials": None, "essential": True}


def _unresearched_essential(state: DesignState, decision: DesignerDecision) -> list[FurnitureGap]:
    """Essential gaps that research could still close: gaps the designer wants to finish with, and
    gaps missing from its own list. Gaps researched before in this run (also skipped or rejected
    ones) are never searched again automatically. Only while research is possible (online search on)."""
    finishing = decision.next_action == "finish" or (decision.next_action == "validate" and decision.finish_when_valid)
    if not (finishing or decision.next_action == "request_furniture") or research_blocker(state) is not None:
        return []
    return [g for g in decision.furniture_gaps if g.essential and not _was_researched(state, g.item)]


def _research_list(decision: DesignerDecision, gaps: list[FurnitureGap]) -> tuple[list[dict], dict]:
    """The designer's own list plus the essential gaps it does not cover, essential pieces first."""
    request = decision.furniture_request
    items = [i.model_dump() for i in request.items] if request else []
    covered = lambda gap: any(gap.item.strip().lower() in i["description"].lower() for i in items)
    items += [_gap_item(g) for g in gaps if not covered(g)]
    shared = {"style": request.style, "budget": request.budget} if request else {}
    return sorted(items, key=lambda i: not i["essential"]), shared


def guard_action(state: DesignState, decision: DesignerDecision) -> tuple[dict, str | None]:
    """Turns the LLM's decision into the actual next step. `state` already contains this step's
    layout change. Returns (state update, note if the decision was overridden)."""
    action = decision.next_action
    version = current_version(state)
    gaps = _unresearched_essential(state, decision)
    if gaps and action != "request_furniture":
        items, shared = _research_list(decision, gaps)
        return _start_research(items, shared), f"essential furniture missing: {', '.join(g.item for g in gaps)}"
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
        items, shared = _research_list(decision, gaps)
        if not items:
            return _safe_fallback(state, "furniture request without items")
        merged = [g.item for g in gaps if any(i["description"] == _gap_item(g)["description"] for i in items)]
        return _start_research(items, shared), (f"added essential gaps to the list: {', '.join(merged)}" if merged else None)
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
