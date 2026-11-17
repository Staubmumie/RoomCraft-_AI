"""Interior Critic agent: evaluates a layout and suggests improvements without changing it.

Called by the orchestrator (critique of the confirmed room -> END) or by the interior designer
(critique of the designer's working layout -> back to the designer). If the latest screenshots do
not show the layout to critique, it first asks render_scene for new ones (render_scene returns here).
"""
import json
import math
from typing import Literal

from pydantic import BaseModel, Field

from design.routing import CRITIC_BONUS, caller
from design.scene_images import image_content, load_scene_images
from design.state import DesignState, model_for
from design.working_layout import current_version, layout_furniture
from llm import LLMError, structured_completion
from observability import progress
from prompts import load_prompt


Aspect = Literal["functionality", "aesthetics", "space_utilization", "preference_fit"]


class CritiqueIssue(BaseModel):
    severity: Literal["low", "medium", "high"]
    aspect: Aspect = Field(description="The score this issue lowers")
    description: str = Field(description="The concrete problem in this layout: where, which pieces or area, and why "
                                         "it hurts the room")
    furniture_ids: list[str] = Field(description="Ids of the furniture involved, may be empty")


class CritiqueScores(BaseModel):
    functionality: int = Field(description="1 (poor) to 5 (excellent)")
    aesthetics: int = Field(description="1 (poor) to 5 (excellent)")
    space_utilization: int = Field(description="1 (poor) to 5 (excellent)")
    preference_fit: int = Field(description="How well it fits the user's request and preferences, 1 to 5")


class Critique(BaseModel):
    summary: str = Field(description="2-3 sentences, in the language of the user's request")
    strengths: list[str]
    issues: list[CritiqueIssue]
    # After the observations: structured output is generated in field order, so the scores follow from them.
    scores: CritiqueScores
    suggestions: list[str] = Field(description="Concrete, actionable improvements (no coordinates needed)")


SYSTEM_PROMPT = load_prompt("interior_critic_system")


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
    elif research_bonus_due(state, critique):
        update["critic_research_bonus"] = True
        update["workflow_log"].append(f"interior_critic: furniture research bonus granted (+{CRITIC_BONUS['research_requests']} "
                                      "list) for furniture that resolves the critique")
    return update


def research_bonus_due(state: DesignState, critique: dict) -> bool:
    """Once per run: a critique with issues or suggestions earns the designer one more furniture list
    (routing.CRITIC_BONUS), only when it may add furniture and search online at all."""
    options = state.get("options") or {}
    return (not state.get("critic_research_bonus") and bool(critique["issues"] or critique["suggestions"])
            and bool(options.get("allow_new_furniture")) and options.get("search_online", True) is not False)
