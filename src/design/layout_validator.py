"""Layout Validator node: deterministic checks of the designer's working layout.

Always returns to the interior designer, valid or not; the designer decides what to do. When a valid
layout changed substantially since the last screenshots (or they are outdated), it goes through
render_scene first (render_after_validation), so the designer sees its latest valid room.
Automatic corrections become a new layout version, so it is clear which version was validated. They
are kept even when other problems remain: every correction was only accepted if the moved piece ends
up clear, so the designer continues from the corrected layout, which is also the one the remaining
errors (and their repair options) refer to.

Each validation appends one entry to state["repair_history"] (problem count, total and maximum
severity, which problems, positions of the pieces involved). The designer compares the entries to
see whether its repairs work; see interior_designer.repair_progress_text.
"""
import math

from design.state import DesignState
from design.validator import DEFAULT_CONFIG, ValidationResult, ValidationStats, ValidatorConfig, validate_layout
from design.working_layout import (
    as_placements, current_version, layout_furniture, proposed_changes, render_due, validation_baseline,
)
from observability import progress


def correction_messages(result) -> list[str]:
    return [f"'{c.name}' was moved {c.distance * 100:.1f} cm ({c.reason})." for c in result.corrections]


def _empty_room() -> ValidationResult:
    return ValidationResult(status="valid", layout=[], problems=[], corrections=[], errors=[],
                            stats=ValidationStats(checked=0, collisions=0, boundaryViolations=0, corrections=0, unresolved=0))


def history_entry(state: DesignState, version: int, result: ValidationResult, pieces) -> dict:
    """Measurable state of one validation, for comparing repair attempts."""
    errors = result.errors
    involved = {i for e in errors for i in e.furnitureIds}
    positions = {p.id: [round(p.x, 3), round(p.y, 3), round(p.z, 3), round(math.degrees(p.rotation))]
                 for p in pieces if p.id in involved}
    return {"version": version, "status": result.status, "strategy": state.get("repair_strategy") or "none",
            "unresolved": len(errors), "collisions": sum(e.type == "collision" for e in errors),
            "severity": round(sum(e.size for e in errors), 3), "maxDepth": round(max((e.size for e in errors), default=0.0), 3),
            "problems": {e.key: round(e.size, 3) for e in errors}, "positions": positions}


def render_reason(state: DesignState, result: ValidationResult, pieces, version: int) -> str | None:
    """Why the validated layout should be rendered before the designer continues (None: no render).
    Invalid layouts are not rendered (the repair works from the numbers), and neither is a layout
    that is accepted right away without another designer step (a final review renders on its own)."""
    if result.status == "invalid" or state.get("finish_when_valid"):
        return None
    return render_due({**state, "proposed_layout": pieces, "layout_version": version})


def layout_validator(state: DesignState, validator_config: ValidatorConfig = DEFAULT_CONFIG) -> dict:
    # Not named `config`: LangGraph injects its RunnableConfig into a parameter with that name.
    version = current_version(state)
    pieces = layout_furniture(state)
    if not pieces:
        result = _empty_room()  # everything was removed: nothing can collide
    else:
        # Confirmed pieces the designer did not move keep their position and are not moved by corrections;
        # new and moved pieces are the proposal.
        proposed = proposed_changes(state) or as_placements(pieces)
        result = validate_layout(state["room"], validation_baseline(state), proposed, config=validator_config,
                                 stage=progress.stage)

    update: dict = {"validation": result, "counters": {"validations": 1}}
    if result.corrections:
        version += 1
        corrected = {p.id: p for p in result.layout}
        pieces = [p.model_copy(update={"x": corrected[p.id].x, "y": corrected[p.id].y, "z": corrected[p.id].z})
                  if p.id in corrected else p for p in pieces]
        update.update(layout_version=version, proposed_layout=pieces)
    update["validated_layout_version"] = version
    reason = render_reason(state, result, pieces, version)
    update["render_after_validation"] = reason is not None
    if reason:
        update.update(return_to={"render_scene": "interior_designer"}, render_trigger="validation")
        progress.step("render", f"New screenshots of v{version} before the designer continues ({reason})")
    entry = history_entry(state, version, result, pieces)
    update["repair_history"] = [entry]
    progress.step("result", f"Result: {result.status}", layoutVersion=version, **result.stats.model_dump(),
                  severity=entry["severity"], maxDepth=entry["maxDepth"])
    update["workflow_log"] = [
        f"layout_validator: layout v{version} {result.status}"
        + (f", {len(result.corrections)} correction(s)" if result.corrections else "")
        + (f", {len(result.errors)} error(s), total severity {entry['severity']:.2f} m" if result.errors else "")
    ]
    return update
