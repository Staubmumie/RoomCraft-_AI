from langgraph.graph import END, START


def describe_graph(compiled_graph) -> dict:
    """Nodes and edges of a compiled LangGraph for the frontend, taken from the graph itself.
    Labels and kinds come from the `metadata` passed to `add_node`."""
    graph = compiled_graph.get_graph()
    nodes = []
    for node in graph.nodes.values():
        metadata = node.metadata or {}
        if node.id == START:
            kind, label = "start", "Start"
        elif node.id == END:
            kind, label = "end", "End"
        else:
            kind, label = metadata.get("kind", "node"), metadata.get("label", node.name)
        nodes.append({"id": node.id, "label": label, "kind": kind, "description": metadata.get("description")})
    edges = [
        {
            "id": f"{edge.source}->{edge.target}",
            "source": edge.source,
            "target": edge.target,
            "conditional": edge.conditional,
            "label": _route_label(compiled_graph, edge),
        }
        for edge in graph.edges
    ]
    return {"nodes": nodes, "edges": edges}


def _route_label(compiled_graph, edge) -> str | None:
    """All routing keys of a conditional edge (several keys can lead to the same node)."""
    if not edge.conditional:
        return None
    keys = [key for branch in compiled_graph.builder.branches.get(edge.source, {}).values()
            for key, target in (branch.ends or {}).items() if target == edge.target]
    return " / ".join(keys) or (edge.data if isinstance(edge.data, str) else None)
