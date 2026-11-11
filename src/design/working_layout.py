"""The layout the agents are currently working on: the designer's proposal if there is one,
otherwise the confirmed room. Version 0 always means the confirmed room.

A proposal is a complete list of pieces: confirmed pieces (possibly moved), pieces added from the
inventory (ids starting with "new-") and, by their absence, removed pieces.
"""
import math

from design.state import DesignState, Furniture, Placement


def has_proposal(state: DesignState) -> bool:
    return state.get("proposed_layout") is not None


def current_version(state: DesignState) -> int:
    return state.get("layout_version", 0) if has_proposal(state) else 0


def layout_furniture(state: DesignState) -> list[Furniture]:
    """Every piece of the working layout at its working position."""
    return list(state["proposed_layout"]) if has_proposal(state) else list(state["furniture"])


def as_placements(furniture: list[Furniture]) -> list[Placement]:
    return [Placement(id=f.id, x=f.x, y=f.y, z=f.z, rotation=f.rotation) for f in furniture]


def _confirmed(state: DesignState) -> dict[str, Furniture]:
    return {f.id: f for f in state["furniture"]}


def proposed_changes(state: DesignState) -> list[Placement]:
    """Working pieces that are new or differ from the confirmed room (what the designer placed)."""
    confirmed = _confirmed(state)
    return [Placement(id=p.id, x=p.x, y=p.y, z=p.z, rotation=p.rotation) for p in layout_furniture(state)
            if (f := confirmed.get(p.id)) is None or (f.x, f.y, f.z, f.rotation) != (p.x, p.y, p.z, p.rotation)]


def validation_baseline(state: DesignState) -> list[Furniture]:
    """The working pieces for the validator: confirmed pieces at their confirmed position (so only what
    the designer moved counts as a change), new pieces as placed. Removed pieces are not included."""
    confirmed = _confirmed(state)
    return [confirmed[p.id] if p.id in confirmed else p for p in layout_furniture(state)]


def removed_ids(state: DesignState) -> list[str]:
    working = {p.id for p in layout_furniture(state)}
    return [f.id for f in state["furniture"] if f.id not in working]


MOVE_TOLERANCE_M = 0.05
TURN_TOLERANCE_RAD = math.radians(5)


def _turned(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def changes_since_render(state: DesignState) -> dict[str, list[str]] | None:
    """What differs between the latest screenshots and the working layout: names of moved
    (incl. turned / lifted), added and removed pieces. None if there are no screenshots."""
    if not state.get("scene_images"):
        return None
    rendered = {p.id: p for p in state.get("rendered_layout") or
                (state["furniture"] if state.get("rendered_layout_version", 0) == 0 else [])}
    working = {p.id: p for p in layout_furniture(state)}
    moved = [p.name for pid, p in working.items() if (r := rendered.get(pid)) is not None and (
        math.dist((p.x, p.y, p.z), (r.x, r.y, r.z)) > MOVE_TOLERANCE_M or _turned(p.rotation, r.rotation) > TURN_TOLERANCE_RAD)]
    return {"moved": moved,
            "added": [p.name for pid, p in working.items() if pid not in rendered],
            "removed": [p.name for pid, p in rendered.items() if pid not in working]}


def is_substantial(changes: dict[str, list[str]]) -> bool:
    """A change worth new screenshots: pieces added or removed, or at least two moved/turned."""
    return bool(changes["added"] or changes["removed"]) or len(changes["moved"]) >= 2


STALE_RENDER_VERSIONS = 2  # screenshots this many layout versions behind count as outdated


def render_due(state: DesignState) -> str | None:
    """Why the working layout should be rendered now (None if the screenshots are good enough):
    no screenshots, a substantial change since the last render, or screenshots that are outdated."""
    version = current_version(state)
    if state.get("scene_images") and state.get("rendered_layout_version") == version:
        return None
    changes = changes_since_render(state)
    if changes is None:
        return "no screenshots yet"
    if is_substantial(changes):
        return "substantial change since the last screenshots"
    if version - state.get("rendered_layout_version", 0) >= STALE_RENDER_VERSIONS:
        return f"screenshots are {version - state.get('rendered_layout_version', 0)} versions old"
    return None


def added_pieces(state: DesignState) -> list[Furniture]:
    confirmed = _confirmed(state)
    return [p for p in layout_furniture(state) if p.id not in confirmed]
