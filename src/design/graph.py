"""The RoomCraft LangGraph workflow. Every node exists once; shared nodes return to their caller.

    START -> orchestrator
    orchestrator       -> REJECT: END | DESIGN_ROOM / CRITIQUE_ROOM: render_scene | SEARCH_FURNITURE: furniture_research
    render_scene       -> return_to["render_scene"]: interior_designer | interior_critic
    interior_designer  -> layout_validator | furniture_research | interior_critic | END
    layout_validator   -> render_scene (valid and substantially changed / outdated screenshots) | interior_designer
    interior_critic    -> render_scene (outdated screenshots) | interior_designer (if it asked) | END (standalone critique)
    furniture_research -> human_approval (one item ready) | asset_download (downloads queued)
                          | interior_designer (if it asked) | END (standalone search)
    human_approval     -> approve: asset_download | reject / skip: furniture_research   (interrupt())
    asset_download     -> furniture_research (always)

Routing functions and loop limits live in design.routing. The three nodes of a furniture list
(furniture_research, human_approval, asset_download) live in the research package. human_approval
pauses the run with interrupt(); it resumes in the same LangGraph thread with Command(resume=decision).
The per-item searches and downloads of a furniture list run as background jobs (research.jobs) that
outlive such a pause; furniture_research coordinates them through the explicit state in `research_items`.
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph

from design.interior_critic import interior_critic
from design.interior_designer import interior_designer
from design.layout_validator import layout_validator
from design.orchestrator import orchestrator
from design.render_scene import render_scene
from design.routing import FIXED_EDGES, ROUTERS
from design.state import DesignState
from research.agent import furniture_research
from research.asset_download import asset_download
from research.human_approval import human_approval

NODES = {
    "orchestrator": (orchestrator, "Request Router", "router",
                     "Checks the request (blocks manipulation attempts) and routes it: design, critique, search or reject."),
    "render_scene": (render_scene, "Render Scene", "tool",
                     "Screenshots of the working layout: first look, after validated changes, for the critic."),
    "interior_designer": (interior_designer, "Interior Designer", "agent",
                          "Creates and revises the working layout and decides the next step."),
    "layout_validator": (layout_validator, "Layout Validator", "validator",
                         "Deterministic boundary and collision checks with small corrections."),
    "interior_critic": (interior_critic, "Interior Critic", "agent",
                        "Evaluates a layout and suggests improvements."),
    "furniture_research": (furniture_research, "Furniture Research", "agent",
                           "Works through a furniture list: local check, parallel online searches per item, results."),
    "human_approval": (human_approval, "Human Approval", "human",
                       "Waits for the user's decision on one furniture item (approve, nothing fits, skip)."),
    "asset_download": (asset_download, "Asset Download", "tool",
                       "Starts background downloads of approved or automatically chosen assets."),
}


def build_design_graph(checkpointer=None):
    graph = StateGraph(DesignState)
    for name, (node, label, kind, description) in NODES.items():
        graph.add_node(name, node, metadata={"label": label, "kind": kind, "description": description})
    graph.add_edge(START, "orchestrator")
    for source, target in FIXED_EDGES.items():
        graph.add_edge(source, target)
    for node, (router, routes) in ROUTERS.items():
        graph.add_conditional_edges(node, router, routes)
    # interrupt() needs a checkpointer. TODO: a persistent saver (e.g. SQLite) together with a
    # persistent run store, so paused runs survive a backend restart.
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


design_graph = build_design_graph()


def mermaid() -> str:
    """Mermaid diagram of the compiled graph (each node once, all routing keys on the edges)."""
    from observability.graph_view import describe_graph

    graph = describe_graph(design_graph)
    lines = ["flowchart TD"]
    for node in graph["nodes"]:
        shape = "([{}])" if node["kind"] in ("start", "end") else "[{}]"
        lines.append(f"    {node['id']}{shape.format(node['label'])}")
    for edge in graph["edges"]:
        arrow = f" -. {edge['label']} .-> " if edge["conditional"] and edge["label"] else " --> "
        lines.append(f"    {edge['source']}{arrow}{edge['target']}")
    return "\n".join(lines)
