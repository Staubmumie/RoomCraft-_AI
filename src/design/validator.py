"""Deterministic layout validation: data checks, room boundaries, collisions, vertical support and
small automatic corrections. No LLM, no LangGraph: usable from the graph, an API route or tests.

It is a safety net against physically implausible layouts, not an interior design rule engine:
- Collision = the footprints overlap AND the height ranges overlap (each beyond a small tolerance).
  A chair on a rug, a lamp on a table or a wall cabinet above a counter therefore never collides;
  two cabinets in the same volume do. Small overlaps from imperfect model bounding boxes are tolerated.
- Support: a piece whose bottom is clearly above the floor must either stand on another piece
  (its centre on that piece's footprint, its bottom near that piece's top, and the supporting piece
  not much smaller) or touch a wall (wall-mounted). Anything else is "unsupported" (floating).
- Pieces without a known height keep the former 2D behaviour (they block their whole footprint).

Correction procedure (bounded, deterministic):
1. Detect all boundary violations, collisions and unsupported pieces in the complete layout.
2. For the first problem (fixed order) build candidate moves: back inside the room, apart along
   each SAT axis of a colliding pair (either piece), or down onto the surface below / the floor.
3. A candidate is accepted only if the moved piece ends up inside the room, collides with nothing,
   is supported, and stays within `max_correction` of where the designer put it. For a collision,
   the smaller piece of the pair may be pushed up to `max_collision_push` (e.g. a chair out of a
   table), as long as nothing stands on it. Among valid candidates the smallest move wins.
4. Apply it and start over; anything left after `max_iterations` is reported as unresolved.
Rotations are never changed.

Overlap classes: intentional (rug below, object on top, wall piece above: never reported), numerical
(within the tolerances: ignored), automatically repairable (see 3.) and design decisions (reported
to the designer). Every unresolved geometry problem carries `options`: concrete separating moves of
each involved piece (minimum translation along each SAT axis, or back into the room / onto a
surface / to a wall) with their target position and what that move would cause on its own. They are
geometric options, not guaranteed solutions.
"""
import math
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from design.geometry import SIDES, Footprint, RoomBounds, overlap_depth, separation_axes, vertical_overlap
from design.state import Furniture, Placement, Room


@dataclass(frozen=True)
class ValidatorConfig:
    max_correction: float = 0.15       # metres a piece may be moved automatically (from the designer's position)
    max_collision_push: float = 0.30   # larger limit for pushing the smaller piece of a colliding pair apart
    collision_tolerance: float = 0.01  # floor overlaps up to this depth count as touching (imperfect bounding boxes)
    vertical_tolerance: float = 0.03   # height overlaps up to this count as resting on / under (rugs, sunk-in bases)
    support_tolerance: float = 0.05    # gap between a piece's bottom and the surface (or floor) it stands on
    wall_tolerance: float = 0.05       # max. gap between a wall-mounted piece and its wall
    min_support_ratio: float = 0.5     # a supporting piece needs at least this share of the supported footprint
    boundary_tolerance: float = 0.0005
    separation_margin: float = 0.002   # gap left between pieces after separating them
    max_iterations: int = 50


DEFAULT_CONFIG = ValidatorConfig()

Status = Literal["valid", "auto_corrected", "invalid"]
GEOMETRY_TYPES = ("out_of_bounds", "collision", "unsupported")


class RepairOption(BaseModel):
    """One geometric way to resolve a problem by moving a single piece (rotation unchanged)."""
    furnitureId: str
    name: str
    move: str = Field(description="Direction label, e.g. '+x', '-z', '(+0.71x, -0.71z)', 'down', 'to the back wall'")
    distance: float = Field(description="Length of the move in metres")
    target: dict[str, float] = Field(description="Resulting position {x, y, z}")
    outcome: str = Field(description="'clear', or what this move alone would cause, e.g. \"hits 'Sofa' (id)\"")


class Problem(BaseModel):
    type: str = Field(description="empty_layout, unknown_furniture, duplicate_furniture, invalid_value, "
                                  "invalid_dimensions, too_large, out_of_bounds, collision or unsupported")
    furnitureIds: list[str]
    message: str
    measurements: dict[str, float] = Field(default_factory=dict)
    retryable: bool = True  # can the designer fix it by changing the layout?
    severity: Literal["small", "medium", "large"] | None = None  # by the move needed, see _severity
    options: list[RepairOption] = Field(default_factory=list)

    @property
    def key(self) -> str:
        """Identifies the same problem across validations (type + pieces involved)."""
        return f"{self.type}:{'|'.join(sorted(self.furnitureIds))}"

    @property
    def size(self) -> float:
        """Penetration depth / violation / needed move in metres (0 for data problems)."""
        m = self.measurements
        return float(m.get("overlapDepth") or m.get("violation") or m.get("requiredCorrection") or 0.0)


class Correction(BaseModel):
    furnitureId: str
    name: str
    before: dict[str, float]
    after: dict[str, float]
    distance: float
    reason: str


class ValidationStats(BaseModel):
    checked: int
    collisions: int
    boundaryViolations: int
    corrections: int
    unresolved: int
    unsupported: int = 0


class ValidationResult(BaseModel):
    status: Status
    layout: list[Placement] = Field(description="Complete layout incl. unchanged furniture, after corrections")
    problems: list[Problem] = Field(description="Everything found in the proposed layout")
    corrections: list[Correction]
    errors: list[Problem] = Field(description="Problems that remain unresolved")
    stats: ValidationStats

    @property
    def retryable(self) -> bool:
        return bool(self.errors) and all(e.retryable for e in self.errors)


StageFactory = Callable[[str, str], AbstractContextManager[dict]]


def _finite(*values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _pos(fp: Footprint) -> dict[str, float]:
    return {"x": round(fp.x, 4), "y": round(fp.y, 4), "z": round(fp.z, 4)}


def _distance(a: Footprint, b: tuple[float, float, float]) -> float:
    return math.dist((a.x, a.y, a.z), b)


def _short(name: str, length: int = 28) -> str:
    return name if len(name) <= length else name[:length - 1] + "…"


def _direction(dx: float, dz: float) -> str:
    for label, (ux, uz) in (("+x", (1, 0)), ("-x", (-1, 0)), ("+z", (0, 1)), ("-z", (0, -1))):
        if dx * ux + dz * uz > 0.98:
            return label
    return f"({dx:+.2f}x, {dz:+.2f}z)"


def _severity(size: float, config: ValidatorConfig) -> str:
    if size <= config.max_correction:
        return "small"
    return "medium" if size <= max(config.max_collision_push, config.max_correction) else "large"


def room_bounds(room: Room) -> RoomBounds:
    """Walls are given as e.g. 'back (z = -3)'; without any, all four sides count as walls."""
    sides = tuple(s for s in SIDES if any(w.strip().lower().startswith(s) for w in room.walls))
    return RoomBounds(room.width, room.depth, room.height, sides or SIDES)


# --- data checks -----------------------------------------------------------

def check_data(room: Room, furniture: list[Furniture], proposed: list[Placement]) -> list[Problem]:
    problems: list[Problem] = []
    known = {f.id for f in furniture}
    if not proposed:
        problems.append(Problem(type="empty_layout", furnitureIds=[], message="The layout contains no placements."))

    unknown = sorted({p.id for p in proposed} - known)
    if unknown:
        problems.append(Problem(type="unknown_furniture", furnitureIds=unknown,
                                message=f"Unknown furniture ids (do not exist in the room): {', '.join(unknown)}"))
    ids = [p.id for p in proposed]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        problems.append(Problem(type="duplicate_furniture", furnitureIds=duplicates,
                                message=f"Furniture placed more than once: {', '.join(duplicates)}"))
    for p in proposed:
        if not _finite(p.x, p.y, p.z):
            problems.append(Problem(type="invalid_value", furnitureIds=[p.id],
                                    message=f"Position of '{p.id}' is not a valid number."))
        if not _finite(p.rotation):
            problems.append(Problem(type="invalid_value", furnitureIds=[p.id],
                                    message=f"Rotation of '{p.id}' is not a valid number."))

    bounds = room_bounds(room)
    for f in furniture:
        if not _finite(f.width, f.depth) or f.width <= 0 or f.depth <= 0:
            problems.append(Problem(type="invalid_dimensions", furnitureIds=[f.id], retryable=False,
                                    message=f"'{f.name}' has no valid width/depth, so it cannot be validated."))
        elif not any(bounds.fits(Footprint(0, 0, f.width, f.depth, r)) for r in (0.0, math.pi / 2)):
            problems.append(Problem(type="too_large", furnitureIds=[f.id], retryable=False,
                                    message=f"'{f.name}' ({f.width:.2f} × {f.depth:.2f} m) does not fit into the room.",
                                    measurements={"width": f.width, "depth": f.depth}))
        elif _finite(f.height) and f.height > room.height:
            problems.append(Problem(type="too_large", furnitureIds=[f.id], retryable=False,
                                    message=f"'{f.name}' ({f.height:.2f} m high) is taller than the room.",
                                    measurements={"height": f.height}))
    return problems


def merge_layout(furniture: list[Furniture], proposed: list[Placement]) -> list[Placement]:
    """Complete layout in room order: proposed placements, everything else keeps its current position."""
    by_id = {p.id: p for p in proposed}
    layout = []
    for f in furniture:
        p = by_id.get(f.id) or Placement(id=f.id, x=f.x, y=f.y, z=f.z, rotation=f.rotation)
        layout.append(Placement(id=f.id, x=p.x, y=p.y, z=p.z, rotation=math.remainder(p.rotation, 2 * math.pi)))
    return layout


# --- geometry checks -------------------------------------------------------

class _Layout:
    """Working copy of the layout's pieces during correction."""

    def __init__(self, room: Room, furniture: list[Furniture], layout: list[Placement], config: ValidatorConfig,
                 movable: set[str]):
        self.bounds = room_bounds(room)
        self.config = config
        self.movable = movable  # only pieces the designer placed; unchanged furniture keeps its position
        self.names = {f.id: f.name for f in furniture}
        self.order = [p.id for p in layout]
        dims = {f.id: f for f in furniture}
        self.items = {p.id: Footprint(p.x, p.z, dims[p.id].width, dims[p.id].depth, p.rotation,
                                      dims[p.id].height if _finite(dims[p.id].height) else None, p.y)
                      for p in layout}
        self.original = {i: (fp.x, fp.y, fp.z) for i, fp in self.items.items()}

    # -- relations
    def collides(self, a: Footprint, b: Footprint) -> float:
        """Floor overlap depth of two pieces that intersect in 3D, else 0."""
        if vertical_overlap(a, b) <= self.config.vertical_tolerance:
            return 0.0
        depth = overlap_depth(a, b)
        return depth if depth > self.config.collision_tolerance else 0.0

    def supports(self, base: Footprint, piece: Footprint) -> bool:
        """Can `piece` stand on `base`: bottom near base's top, centre on it, base not much smaller."""
        return (base.top is not None and abs(piece.y - base.top) <= self.config.support_tolerance
                and base.contains(piece.x, piece.z, self.config.support_tolerance)
                and base.area >= self.config.min_support_ratio * piece.area)

    def is_supported(self, item_id: str, fp: Footprint) -> bool:
        if fp.y <= self.config.support_tolerance:
            return True  # on the floor (or on a rug)
        if self.bounds.wall_gap(fp) <= self.config.wall_tolerance:
            return True  # wall-mounted
        return any(self.supports(other, fp) for other_id, other in self.items.items() if other_id != item_id)

    def resting_height(self, item_id: str, fp: Footprint) -> float:
        """Height of the highest surface below the piece that could carry it, or the floor."""
        tops = [other.top for other_id, other in self.items.items()
                if other_id != item_id and other.top is not None and other.top <= fp.y + self.config.support_tolerance
                and other.contains(fp.x, fp.z) and other.area >= self.config.min_support_ratio * fp.area]
        return max(tops, default=0.0)

    # -- detection
    def problems(self) -> list[Problem]:
        found = []
        for i in self.order:
            fp = self.items[i]
            if not self.bounds.fits(fp):
                found.append(Problem(type="too_large", furnitureIds=[i],
                                     message=f"'{self.names[i]}' is too large for the room at this rotation."))
                continue
            violation = self.bounds.violation(fp)
            if violation > self.config.boundary_tolerance:
                found.append(Problem(type="out_of_bounds", furnitureIds=[i], measurements={"violation": round(violation, 4)},
                                     message=f"'{self.names[i]}' extends {violation:.3f} m beyond the room boundary."))
            vertical = self.bounds.vertical_violation(fp)
            if vertical > self.config.vertical_tolerance:
                where = "below the floor" if fp.y < 0 else "above the ceiling"
                found.append(Problem(type="out_of_bounds", furnitureIds=[i], measurements={"violation": round(vertical, 4)},
                                     message=f"'{self.names[i]}' reaches {vertical:.3f} m {where}."))
        for n, a in enumerate(self.order):
            for b in self.order[n + 1:]:
                depth = self.collides(self.items[a], self.items[b])
                if depth:
                    found.append(Problem(type="collision", furnitureIds=[a, b], measurements={"overlapDepth": round(depth, 4)},
                                         message=f"'{self.names[a]}' and '{self.names[b]}' overlap by {depth:.3f} m."))
        for i in self.order:
            fp = self.items[i]
            if self.bounds.vertical_violation(fp) <= self.config.vertical_tolerance and not self.is_supported(i, fp):
                found.append(Problem(type="unsupported", furnitureIds=[i], measurements={"y": round(fp.y, 4)},
                                     message=f"'{self.names[i]}' floats {fp.y:.2f} m above the floor: it neither stands "
                                             "on a piece below it nor touches a wall."))
        return found

    def is_clear(self, item_id: str, fp: Footprint) -> bool:
        if self.bounds.violation(fp) > self.config.boundary_tolerance:
            return False
        if self.bounds.vertical_violation(fp) > self.config.vertical_tolerance:
            return False
        if any(self.collides(fp, other) for other_id, other in self.items.items() if other_id != item_id):
            return False
        return self.is_supported(item_id, fp)

    # -- correction candidates
    def candidates(self, problem: Problem) -> list[tuple[str, Footprint, str]]:
        return [c for c in self._candidates(problem) if c[0] in self.movable]

    def _candidates(self, problem: Problem) -> list[tuple[str, Footprint, str]]:
        if problem.type == "out_of_bounds":
            (item_id,) = problem.furnitureIds
            fp = self.items[item_id]
            x, z = self.bounds.nearest_inside(fp)
            top_limit = self.bounds.height - (fp.height or 0.0)
            y = min(max(fp.y, 0.0), max(top_limit, 0.0))
            return [(item_id, fp.moved(x, z, y), f"was {problem.measurements['violation']:.3f} m outside the room")]
        if problem.type == "collision":
            a, b = problem.furnitureIds
            reason = lambda other: f"overlapped '{self.names[other]}' by {problem.measurements['overlapDepth']:.3f} m"
            result = []
            for (ax, az), overlap in separation_axes(self.items[a], self.items[b]):
                step = overlap + self.config.separation_margin
                fa, fb = self.items[a], self.items[b]
                result.append((b, fb.moved(fb.x + ax * step, fb.z + az * step), reason(a)))
                result.append((a, fa.moved(fa.x - ax * step, fa.z - az * step), reason(b)))
            return result
        if problem.type == "unsupported":
            (item_id,) = problem.furnitureIds
            fp = self.items[item_id]
            return [(item_id, fp.moved(fp.x, fp.z, self.resting_height(item_id, fp)),
                     f"floated {problem.measurements['y']:.2f} m above the floor; lowered onto the surface below")]
        return []

    def carries(self, item_id: str) -> bool:
        """Does anything stand on this piece (moving it would leave that piece floating)?"""
        base = self.items[item_id]
        return any(other.y > self.config.support_tolerance and self.supports(base, other)
                   for other_id, other in self.items.items() if other_id != item_id)

    def correction_limit(self, problem: Problem, item_id: str) -> float:
        """How far `item_id` may be moved automatically for `problem`: the larger collision push
        only for the smaller piece of a colliding pair that carries nothing."""
        limit = self.config.max_correction
        if problem.type == "collision" and self.config.max_collision_push > limit:
            (partner,) = [i for i in problem.furnitureIds if i != item_id] or [item_id]
            if self.items[item_id].area <= self.items[partner].area + 1e-9 and not self.carries(item_id):
                return self.config.max_collision_push
        return limit

    def outcome(self, item_id: str, fp: Footprint, ignore: str | None = None) -> str:
        """What moving `item_id` to `fp` alone would cause ('clear' if nothing)."""
        violation = self.bounds.violation(fp)
        if violation > self.config.boundary_tolerance:
            return f"leaves the room by {violation:.2f} m"
        if self.bounds.vertical_violation(fp) > self.config.vertical_tolerance:
            return "leaves the room vertically"
        hits = [f"'{_short(self.names[o])}' ({o}) by {d:.2f} m" for o, other in self.items.items()
                if o not in (item_id, ignore) and (d := self.collides(fp, other))]
        if hits:
            return "hits " + ", ".join(hits[:3]) + (f" (+{len(hits) - 3} more)" if len(hits) > 3 else "")
        if not self.is_supported(item_id, fp):
            return "leaves it floating (no support, no wall)"
        return "clear"

    def option(self, item_id: str, fp: Footprint, move: str, ignore: str | None = None) -> RepairOption:
        current = self.items[item_id]
        return RepairOption(furnitureId=item_id, name=self.names[item_id], move=move,
                            distance=round(_distance(fp, (current.x, current.y, current.z)), 3),
                            target={k: round(v, 3) for k, v in _pos(fp).items()},
                            outcome=self.outcome(item_id, fp, ignore))

    def wall_positions(self, fp: Footprint) -> list[tuple[str, Footprint]]:
        """The piece moved straight onto each existing wall (same height, same rotation)."""
        ex, ez = fp.extent
        w, d = self.bounds.width / 2, self.bounds.depth / 2
        targets = {"back": (fp.x, -d + ez), "front": (fp.x, d - ez), "left": (-w + ex, fp.z), "right": (w - ex, fp.z)}
        return [(f"to the {side} wall", fp.moved(*targets[side])) for side in self.bounds.walls]

    def repair_options(self, problem: Problem, per_piece: int = 2, limit: int = 3) -> list[RepairOption]:
        """Separating moves of the involved pieces, at most `per_piece` each and `limit` in total:
        clear ones first, then the shortest."""
        options: list[RepairOption] = []
        if problem.type == "collision":
            a, b = problem.furnitureIds
            seen = set()
            for (ax, az), overlap in separation_axes(self.items[a], self.items[b]):
                if overlap <= 0:
                    continue
                step = overlap + self.config.separation_margin
                for item_id, sign, other in ((b, 1.0, a), (a, -1.0, b)):
                    dx, dz = ax * sign, az * sign
                    key = (item_id, round(dx, 3), round(dz, 3))
                    if key in seen:
                        continue
                    seen.add(key)
                    fp = self.items[item_id]
                    options.append(self.option(item_id, fp.moved(fp.x + dx * step, fp.z + dz * step), _direction(dx, dz), other))
        elif problem.type == "out_of_bounds":
            (item_id,) = problem.furnitureIds
            fp = self.items[item_id]
            if (inside := self.bounds.nearest_inside(fp)) is not None:
                y = min(max(fp.y, 0.0), max(self.bounds.height - (fp.height or 0.0), 0.0))
                options.append(self.option(item_id, fp.moved(*inside, y), "back inside the room"))
        elif problem.type == "unsupported":
            (item_id,) = problem.furnitureIds
            fp = self.items[item_id]
            options.append(self.option(item_id, fp.moved(fp.x, fp.z, self.resting_height(item_id, fp)), "down"))
            options += [self.option(item_id, moved, label) for label, moved in self.wall_positions(fp)]
        # The shortest move always (it shows why the obvious fix fails), then clear ones, then the rest.
        shortest = sorted(options, key=lambda o: o.distance)[:1]
        ranked = shortest + sorted((o for o in options if o not in shortest), key=lambda o: (o.outcome != "clear", o.distance))
        chosen, per = [], {}
        for o in ranked:
            if per.get(o.furnitureId, 0) < per_piece and len(chosen) < limit:
                chosen.append(o)
                per[o.furnitureId] = per.get(o.furnitureId, 0) + 1
        return chosen

    def placements(self, rotations: dict[str, float]) -> list[Placement]:
        return [Placement(id=i, x=round(self.items[i].x, 4), y=round(self.items[i].y, 4), z=round(self.items[i].z, 4),
                          rotation=rotations[i]) for i in self.order]


def _explain_unresolved(work: _Layout, problems: list[Problem]) -> list[Problem]:
    """Adds why automatic correction did not work (e.g. the move it would have needed), how severe
    the problem is and the geometric repair options for the designer."""
    explained = []
    for problem in problems:
        update: dict = {}
        if problem.type in GEOMETRY_TYPES:
            update["options"] = work.repair_options(problem)
        candidates = work.candidates(problem)
        if candidates:
            needed, item_id = min((_distance(fp, work.original[i]), i) for i, fp, _ in candidates)
            limit = work.correction_limit(problem, item_id)
            if needed > limit:
                note = f" Automatic correction would need {needed:.2f} m (limit {limit:.2f} m)."
            else:
                note = " No small move resolves it without creating another collision or leaving the room."
            update["message"] = problem.message + note
            update["measurements"] = {**problem.measurements, "requiredCorrection": round(needed, 4)}
        merged = problem.model_copy(update=update)
        if problem.type in GEOMETRY_TYPES:
            size = merged.size or min((o.distance for o in merged.options), default=0.0)
            merged = merged.model_copy(update={"severity": _severity(size, work.config)})
        explained.append(merged)
    return explained


# --- entry point -------------------------------------------------------------

def validate_layout(room: Room, furniture: list[Furniture], proposed: list[Placement],
                    config: ValidatorConfig = DEFAULT_CONFIG, stage: StageFactory | None = None) -> ValidationResult:
    """Validates a (possibly partial) proposed layout against the room and the existing furniture.
    `stage` optionally wraps the steps for progress reporting (see observability.progress.stage)."""
    stage = stage or (lambda name, label: nullcontext({}))

    with stage("data", "Check data") as details:
        data_problems = check_data(room, furniture, proposed)
        layout = merge_layout(furniture, proposed) if not data_problems else []
        details.update(furniture=len(furniture), proposed=len(proposed), problems=len(data_problems))
    if data_problems:
        return ValidationResult(
            status="invalid", layout=layout, problems=data_problems, corrections=[], errors=data_problems,
            stats=ValidationStats(checked=len(furniture), collisions=0, boundaryViolations=0, corrections=0,
                                  unresolved=len(data_problems)))

    rotations = {p.id: p.rotation for p in layout}
    work = _Layout(room, furniture, layout, config, movable={p.id for p in proposed})
    with stage("detect", "Check room boundaries, collisions and support") as details:
        found = work.problems()
        collisions = sum(p.type == "collision" for p in found)
        boundary = sum(p.type == "out_of_bounds" for p in found)
        unsupported = sum(p.type == "unsupported" for p in found)
        details.update(collisions=collisions, boundaryViolations=boundary, unsupported=unsupported,
                       other=len(found) - collisions - boundary - unsupported)

    corrections: list[Correction] = []
    with stage("correct", "Compute small corrections") as details:
        problems = found
        for _ in range(config.max_iterations):
            fixable = [p for p in problems if p.type in GEOMETRY_TYPES]
            applied = False
            for problem in fixable:
                options = []
                for index, (item_id, fp, reason) in enumerate(work.candidates(problem)):
                    current = work.items[item_id]
                    step = _distance(fp, (current.x, current.y, current.z))
                    total = _distance(fp, work.original[item_id])
                    limit = work.correction_limit(problem, item_id)
                    if step > 1e-9 and total <= limit + 1e-9 and work.is_clear(item_id, fp):
                        if total > config.max_correction + 1e-9:
                            reason += "; larger push of the smaller piece"
                        options.append((step, total, work.order.index(item_id), index, item_id, fp, reason))
                if options:
                    step, _, _, _, item_id, fp, reason = min(options, key=lambda o: o[:4])
                    current = work.items[item_id]
                    corrections.append(Correction(furnitureId=item_id, name=work.names[item_id],
                                                  before=_pos(current), after=_pos(fp),
                                                  distance=round(step, 4), reason=reason))
                    work.items[item_id] = fp
                    applied = True
                    break
            if not applied:
                break
            problems = work.problems()
        details.update(corrections=len(corrections))

    with stage("revalidate", "Re-check the complete layout") as details:
        remaining = work.problems()
        errors = _explain_unresolved(work, remaining)
        details.update(unresolved=len(errors))

    if errors:
        status: Status = "invalid"
    elif corrections:
        status = "auto_corrected"
    else:
        status = "valid"
    return ValidationResult(
        status=status,
        layout=work.placements(rotations),
        problems=found,
        corrections=corrections,
        errors=errors,
        stats=ValidationStats(checked=len(furniture), collisions=collisions, boundaryViolations=boundary,
                              corrections=len(corrections), unresolved=len(errors), unsupported=unsupported),
    )
