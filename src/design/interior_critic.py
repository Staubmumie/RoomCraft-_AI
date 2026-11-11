"""Interior Critic agent: evaluates a layout and suggests improvements without changing it.

Called by the orchestrator (critique of the confirmed room -> END) or by the interior designer
(critique of the designer's working layout -> back to the designer). If the latest screenshots do
not show the layout to critique, it first asks render_scene for new ones (render_scene returns here).
"""
import json
import math
from typing import Literal

from pydantic import BaseModel, Field

from design.routing import caller
from design.scene_images import image_content, load_scene_images
from design.state import DesignState, model_for
from design.working_layout import current_version, layout_furniture
from llm import LLMError, structured_completion
from observability import progress


class CritiqueIssue(BaseModel):
    severity: Literal["low", "medium", "high"]
    description: str
    furniture_ids: list[str] = Field(description="Ids of the furniture involved, may be empty")


class CritiqueScores(BaseModel):
    functionality: int = Field(description="1 (poor) to 5 (excellent)")
    aesthetics: int = Field(description="1 (poor) to 5 (excellent)")
    space_utilization: int = Field(description="1 (poor) to 5 (excellent)")
    preference_fit: int = Field(description="How well it fits the user's request and preferences, 1 to 5")


class Critique(BaseModel):
    summary: str = Field(description="2-3 sentences, in the language of the user's request")
    scores: CritiqueScores
    strengths: list[str]
    issues: list[CritiqueIssue]
    suggestions: list[str] = Field(description="Concrete, actionable improvements (no coordinates needed)")


SYSTEM_PROMPT = """You are an independent interior design critic. You evaluate a room layout honestly and constructively: functionality (walkways, access to fronts, sensible groups), aesthetics (balance, style, colours and materials), space utilisation, and how well it fits the user's request and preferences.

Check in particular that every explicit wish of the request is actually served by furniture in the layout (a requested kitchen needs kitchen furniture, not just free space), that related pieces form coherent groups (chairs around and facing their table, seating around a shared focus) and that no piece stands isolated without a purpose. If the designer's plan is given, judge the layout itself, but point out where it misses the plan's requirements or where the plan itself is weak.

You only give feedback; you never change the layout yourself. Base geometric statements on the structured data (metres, room centred on (0, 0), y = height of a piece's bottom above the floor: pieces may stand on rugs, on other pieces or hang on walls, which is not a collision; top-down image: up = back wall, right = +x, 1 m grid, numbered markers = furniture "marker" numbers). Do not invent furniture you think you see in an image. Write in the language of the user's request."""


def build_messages(state: DesignState, images: list[tuple[str, str]]) -> list[dict]:
    room = state["room"]
    version = current_version(state)
    furniture = [
        {"marker": index + 1, "id": f.id, "name": f.name, "width": f.width, "depth": f.depth, "height": f.height,
         "x": round(f.x, 2), "y": round(f.y, 2), "z": round(f.z, 2), "rotation_deg": round(math.degrees(f.rotation))}
        for index, f in enumerate(layout_furniture(state))
    ]
    for_designer = caller(state, "interior_critic") == "interior_designer"
    subject = ("the designer's working layout v{0} for this request".format(version)
               if for_designer else "the user's current room")
    sections = [
        f"User request: {state['request']}",
        f"Evaluate {subject}.",
        f"Preferences: {json.dumps(state['preferences'])}" if state.get("preferences") else "No stored preferences.",
        "Room and furniture:\n" + json.dumps({"room": {"width": room.width, "depth": room.depth, "height": room.height,
                                                        "walls": room.walls}, "furniture": furniture}, indent=1),
    ]
    if for_designer and state.get("design_plan"):
        sections.append(f"The designer's plan:\n{json.dumps(state['design_plan'], indent=1)}")
    text = "\n\n".join(sections)
    content = image_content(text, images, state.get("rendered_layout_version", 0))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def _clamp_scores(critique: Critique) -> dict:
    data = critique.model_dump()
    data["scores"] = {k: min(max(int(v), 1), 5) for k, v in data["scores"].items()}
    return data


def critique_message(critique: dict) -> str:
    suggestions = " ".join(f"• {s}" for s in critique.get("suggestions", [])[:3])
    return f"{critique['summary']} {suggestions}".strip()


async def interior_critic(state: DesignState) -> dict:
    version = current_version(state)
    requester = caller(state, "interior_critic")

    # Never critique outdated screenshots; re-render only when the layout actually changed. After the
    # render the versions match, so this cannot loop.
    if state.get("rendered_layout_version") != version or not state.get("scene_images"):
        progress.step("render", f"Screenshots show v{state.get('rendered_layout_version')}, requesting v{version}")
        return {"critic_action": "request_render", "return_to": {"render_scene": "interior_critic"},
                "render_trigger": "critic",
                "workflow_log": [f"interior_critic: requested a render of layout v{version}"]}

    model = model_for(state, "interior_critic")
    images = await load_scene_images(state, "Interior Critic", model)
    with progress.stage("prepare", "Prepare critique context") as details:
        messages = build_messages(state, images)
        details.update(layoutVersion=version, requester=requester, images=[view for view, _ in images])
    with progress.stage("llm", "Wait for LLM response") as details:
        try:
            response = await structured_completion(model, messages, Critique)
        except LLMError as exc:
            if exc.info:
                progress.emit("llm", **exc.info.as_dict())
            raise
        progress.emit("llm", **response.info.as_dict())
        details.update(tokens=(response.info.usage or {}).get("totalTokens"), cost=response.info.cost)

    critique = _clamp_scores(response.parsed)
    update = {
        "critique": critique,
        "critiqued_layout_version": version,
        "critic_action": "done",
        "counters": {"critiques": 1},
        "workflow_log": [f"interior_critic: critiqued layout v{version} ({len(critique['issues'])} issue(s), "
                         f"{len(critique['suggestions'])} suggestion(s))"],
    }
    if requester == "end":  # standalone critique: this is the final answer
        update["outcome"] = {"status": "completed", "kind": "critique", "message": critique_message(critique)}
    return update
