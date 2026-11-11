"""Conditional routing of the RoomCraft graph.

Every decision is read from explicit state values: the orchestrator's intent, the agents'
structured actions and the `return_to` entries of shared nodes. Agents propose an action;
the guards in the agents (see LIMITS) make sure the action is allowed before routing here.
"""
from langgraph.graph import END

from design.state import DesignState

# Loop protection per run. Reaching a limit never accepts unfinished work: the designer then
# has to finish with a validated layout or the run fails. Renders have no limit: they cost time but
# no LLM call, and they only happen after validations or for an outdated critique (both bounded).
LIMITS = {
    "designer_steps": 8,     # interior_designer invocations
    "validations": 4,        # layout_validator runs
    "critiques": 2,          # interior_critic runs
    "research_requests": 2,  # furniture requests by the designer (one missing piece each)
    "research_rounds": 3,    # research + approval rounds per request
}

ORCHESTRATOR_ROUTES = {
    "DESIGN_ROOM": "render_scene",
    "CRITIQUE_ROOM": "render_scene",
    "SEARCH_FURNITURE": "furniture_research",
    "REJECT": END,
}
RENDER_ROUTES = {"interior_designer": "interior_designer", "interior_critic": "interior_critic"}
DESIGNER_ROUTES = {
    "validate": "layout_validator",
    "request_furniture": "furniture_research",
    "request_critique": "interior_critic",
    "finish": END,
}
VALIDATOR_ROUTES = {"render": "render_scene", "interior_designer": "interior_designer"}
CRITIC_ROUTES = {"request_render": "render_scene", "interior_designer": "interior_designer", "end": END}
RESEARCH_ROUTES = {"awaiting_approval": "human_approval", "interior_designer": "interior_designer", "end": END}
APPROVAL_ROUTES = {"approve": "asset_download", "reject": "furniture_research", "retry": "furniture_research"}


def limit(state: DesignState, counter: str) -> int:
    """LIMITS, with per-request overrides from options["limits"] (e.g. the user's max. iterations)."""
    return ((state.get("options") or {}).get("limits") or {}).get(counter, LIMITS[counter])


def remaining(state: DesignState, counter: str) -> int:
    return limit(state, counter) - (state.get("counters") or {}).get(counter, 0)


def caller(state: DesignState, node: str) -> str:
    """Who called a shared node; 'end' when it was started by the orchestrator."""
    return (state.get("return_to") or {}).get(node, "end")


def after_orchestrator(state: DesignState) -> str:
    intent = state.get("intent")
    return intent if intent in ORCHESTRATOR_ROUTES else "REJECT"


def after_render(state: DesignState) -> str:
    # Whoever routes to render_scene sets return_to["render_scene"]; a missing entry is a bug.
    target = caller(state, "render_scene")
    if target not in RENDER_ROUTES:
        raise ValueError("render_scene was reached without a return destination")
    return target


def after_designer(state: DesignState) -> str:
    action = state.get("designer_action")
    return action if action in DESIGNER_ROUTES else "finish"


def after_validator(state: DesignState) -> str:
    # The validator decided whether the designer should see new screenshots first (render_scene
    # then returns to the designer via return_to).
    return "render" if state.get("render_after_validation") else "interior_designer"


def after_critic(state: DesignState) -> str:
    if state.get("critic_action") == "request_render":
        return "request_render"
    return caller(state, "interior_critic")


def after_research(state: DesignState) -> str:
    if state.get("research_status") == "awaiting_approval" and state.get("candidates"):
        return "awaiting_approval"
    return caller(state, "furniture_research")


def after_approval(state: DesignState) -> str:
    return state["approval"]["action"]


# node -> (router, path map); used by the graph and to report the chosen next node.
# asset_download -> furniture_research is a fixed edge.
ROUTERS = {
    "orchestrator": (after_orchestrator, ORCHESTRATOR_ROUTES),
    "render_scene": (after_render, RENDER_ROUTES),
    "interior_designer": (after_designer, DESIGNER_ROUTES),
    "layout_validator": (after_validator, VALIDATOR_ROUTES),
    "interior_critic": (after_critic, CRITIC_ROUTES),
    "furniture_research": (after_research, RESEARCH_ROUTES),
    "human_approval": (after_approval, APPROVAL_ROUTES),
}
FIXED_EDGES = {"asset_download": "furniture_research"}


def next_node(node: str, state: DesignState) -> str | None:
    """The node LangGraph runs after `node` for this state."""
    if node in FIXED_EDGES:
        return FIXED_EDGES[node]
    if node not in ROUTERS:
        return None
    router, routes = ROUTERS[node]
    return routes.get(router(state))
