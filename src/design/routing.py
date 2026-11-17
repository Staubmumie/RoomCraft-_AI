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
    # furniture LISTS of the designer that went online (fully local lists do not count); a list holds
    # at most MAX_ITEMS_PER_REQUEST (6) pieces, so this also bounds the pieces searched online
    "research_requests": 2,
}
# A critique with issues or suggestions earns the designer one more furniture list, once per run
# (state["critic_research_bonus"], see limit()).
CRITIC_BONUS = {"research_requests": 1}
# Pieces per furniture list; the furniture research cuts longer lists.
MAX_ITEMS_PER_REQUEST = 6
# Technical concurrency limits of the furniture research (not user settings): per-item searches
# (local check + online search) and asset downloads that run at the same time within one run.
PARALLEL = {"research": 3, "downloads": 3}

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
RESEARCH_ROUTES = {"awaiting_approval": "human_approval", "download": "asset_download",
                   "interior_designer": "interior_designer", "end": END}
APPROVAL_ROUTES = {"approve": "asset_download", "reject": "furniture_research", "skip": "furniture_research"}


def limit(state: DesignState, counter: str) -> int:
    """LIMITS, with per-request overrides from options["limits"] (e.g. the user's max. iterations),
    plus the critic bonus once it was granted in this run."""
    base = ((state.get("options") or {}).get("limits") or {}).get(counter, LIMITS[counter])
    return base + (CRITIC_BONUS.get(counter, 0) if state.get("critic_research_bonus") else 0)


def research_mode(state: DesignState) -> str:
    """How furniture is searched, from the user's options:
    local_only (no online search) | guided (the user picks from suggestions) | automatic (best fit is downloaded)."""
    options = state.get("options") or {}
    if not options.get("search_online", True):
        return "local_only"
    return "guided" if options.get("review_suggestions", True) else "automatic"


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
    # The research coordinator shows one list item for approval, hands approved or automatically
    # chosen assets to asset_download, or returns the finished list to its caller.
    status = state.get("research_status")
    if status == "awaiting_approval" and state.get("approval_item"):
        return "awaiting_approval"
    if status == "download":
        return "download"
    return caller(state, "furniture_research")


def after_approval(state: DesignState) -> str:
    return state["approval"]["action"]


# node -> (router, path map); used by the graph and to report the chosen next node.
# asset_download -> furniture_research is a fixed edge: downloads start in the background, the
# research coordinator collects their results.
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
