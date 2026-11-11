"""Render Scene: one shared node for every agent that needs to see the room.

It renders the working layout (the designer's proposal, or the confirmed room when there is none),
records which layout version the screenshots show and returns to the node stored in
return_to["render_scene"]. Who asks sets render_trigger: "initial" (orchestrator, first look at the
room), "validation" (layout_validator, after a validated substantial change) or "critic" (the critic's
screenshots were outdated). The designer cannot request renders itself.

Every render gets its own folder inside the run's folder (the run id is the LangGraph thread id), see
rendering.service. Rendering itself lives there too; a failure raises RenderError and stops the workflow.
"""
from langchain_core.runnables import RunnableConfig

from design.routing import caller
from design.state import DesignState
from design.working_layout import current_version, layout_furniture
from observability import progress
from rendering.service import render_room


async def render_scene(state: DesignState, config: RunnableConfig) -> dict:
    # The parameter must be called `config`: LangGraph injects its RunnableConfig into it.
    version = current_version(state)
    target = caller(state, "render_scene")
    trigger = state.get("render_trigger") or "initial"
    run_id = ((config or {}).get("configurable") or {}).get("thread_id")
    pieces = layout_furniture(state)
    result = await render_room(state["room"], pieces, stage=progress.stage,
                               run_id=run_id, layout_version=version, trigger=trigger)
    progress.step("done", f"Screenshots of layout v{version} ready for {target}"
                  + (" (cached)" if result.cached else ""),
                  cached=result.cached, durationMs=result.duration_ms, images=result.images, layoutVersion=version,
                  folder=result.folder, trigger=trigger)
    return {
        "scene_images": result.images,
        "rendered_layout_version": version,
        "rendered_layout": pieces,
        "render_after_validation": False,
        "workflow_log": [f"render_scene: rendered layout v{version} for {target} ({trigger})"],
    }
