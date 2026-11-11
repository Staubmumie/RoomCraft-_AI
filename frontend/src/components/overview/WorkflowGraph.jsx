import { useCallback, useEffect, useMemo, useState } from 'react';
import { applyNodeChanges, Handle, MarkerType, Position, ReactFlow } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { formatCost, formatDuration, formatTokens, STATUS_LABELS } from '../../observability/runState.js';

const RANK_GAP = 105;
const COLUMN_GAP = 200;

function AgentNode({ data }) {
  return (
    <div className={`flow-node status-${data.status} ${data.highlighted ? 'highlighted' : ''}`}>
      <Handle type="target" position={Position.Top} id="in" />
      {/* Side handles keep loops (right) and edges that skip a row (left) apart from the main flow. */}
      <Handle type="target" position={Position.Right} id="back-in" />
      <Handle type="source" position={Position.Right} id="back-out" />
      <Handle type="source" position={Position.Left} id="skip-out" />
      <Handle type="source" position={Position.Right} id="skip-out-r" />
      <div className="flow-node-title">
        <span className="status-dot" />
        <strong>{data.label}</strong>
      </div>
      <div className="flow-node-meta">
        <span>{STATUS_LABELS[data.status]}</span>
        {data.durationMs != null && <span className="mono">{formatDuration(data.durationMs)}</span>}
        {data.runs > 1 && <span>{data.runs}×</span>}
      </div>
      {data.waitMs > 0 && (
        <div className="flow-node-meta mono"><span>⏸ {formatDuration(data.waitMs)} gewartet</span></div>
      )}
      {data.tokens > 0 && (
        <div className="flow-node-meta mono">
          <span>{formatTokens(data.tokens)} Tok.</span>
          {data.cost != null && <span>{formatCost(data.cost)}</span>}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} id="out" />
    </div>
  );
}

function TerminalNode({ data }) {
  return (
    <div className={`flow-terminal status-${data.status}`}>
      {data.kind === 'end' && <Handle type="target" position={Position.Top} id="in" />}
      {data.kind === 'end' && <Handle type="target" position={Position.Left} id="skip-in" />}
      {data.kind === 'end' && <Handle type="target" position={Position.Right} id="skip-in-r" />}
      {data.label}
      {data.kind === 'start' && <Handle type="source" position={Position.Bottom} id="out" />}
    </div>
  );
}

const nodeTypes = { agent: AgentNode, terminal: TerminalNode };

// Places nodes in rows by their distance from START, so new nodes and edges need no manual layout.
function layoutPositions(graph) {
  const rank = new Map([['__start__', 0]]);
  const queue = ['__start__'];
  while (queue.length) {
    const id = queue.shift();
    for (const edge of graph.edges.filter((e) => e.source === id)) {
      if (!rank.has(edge.target)) {
        rank.set(edge.target, rank.get(id) + 1);
        queue.push(edge.target);
      }
    }
  }
  const maxRank = Math.max(0, ...rank.values());
  graph.nodes.forEach((n) => { if (!rank.has(n.id)) rank.set(n.id, maxRank + 1); });
  // END always goes last
  if (rank.has('__end__')) rank.set('__end__', Math.max(...[...rank.entries()].filter(([id]) => id !== '__end__').map(([, r]) => r)) + 1);

  const rows = new Map();
  graph.nodes.forEach((n) => rows.set(rank.get(n.id), [...(rows.get(rank.get(n.id)) || []), n.id]));
  const positions = new Map();
  rows.forEach((ids, r) => ids.forEach((id, i) => {
    positions.set(id, { x: (i - (ids.length - 1) / 2) * COLUMN_GAP, y: r * RANK_GAP });
  }));
  return { positions, rank };
}

// Which handles an edge uses: down to the next row, back up (loop), or down past a row
// (on the side of the source node's column, so it doesn't cross the main flow).
function edgeHandles(edge, rank, positions) {
  const from = rank.get(edge.source);
  const to = rank.get(edge.target);
  if (to <= from) return { sourceHandle: 'back-out', targetHandle: 'back-in', type: 'smoothstep' };
  if (to > from + 1) {
    const right = positions.get(edge.source).x > 0;
    const targetHandle = edge.target === '__end__' ? (right ? 'skip-in-r' : 'skip-in') : 'in';
    return { sourceHandle: right ? 'skip-out-r' : 'skip-out', targetHandle, type: 'smoothstep' };
  }
  return { sourceHandle: 'out', targetHandle: 'in' };
}

// Several routing keys can share one edge; show the first and how many more, to keep labels apart.
function shortLabel(label) {
  if (!label) return undefined;
  const keys = label.split(' / ');
  return keys.length > 2 || (keys.length > 1 && label.length > 16) ? `${keys[0]} +${keys.length - 1}` : label;
}

export default function WorkflowGraph({ graph, nodeStates, runStatus, highlightedNode, onNodeClick }) {
  const { positions, rank } = useMemo(() => layoutPositions(graph), [graph]);
  const [nodes, setNodes] = useState([]);
  // React Flow stores measured sizes on the nodes; keep them by applying its changes.
  const onNodesChange = useCallback((changes) => setNodes((list) => applyNodeChanges(changes, list)), []);

  const computed = graph.nodes.map((n) => {
    const terminal = n.kind === 'start' || n.kind === 'end';
    const state = nodeStates.get(n.id);
    let status = state?.status ?? 'idle';
    if (n.kind === 'start' && runStatus !== 'idle') status = 'completed';
    if (n.kind === 'end' && runStatus === 'completed') status = 'completed';
    const position = positions.get(n.id);
    return {
      id: n.id,
      type: terminal ? 'terminal' : 'agent',
      position: terminal ? { x: position.x + 55, y: position.y + (n.kind === 'end' ? 0 : 20) } : position,
      data: {
        ...n, status, durationMs: state?.durationMs, waitMs: state?.waitMs, runs: state?.runs,
        tokens: state?.tokens, cost: state?.cost, highlighted: highlightedNode === n.id
      },
      draggable: false,
      connectable: false
    };
  });

  // Merge live data into the existing nodes instead of replacing them (keeps `measured`).
  const computedKey = JSON.stringify(computed.map((n) => [n.id, n.type, n.position, n.data]));
  useEffect(() => {
    setNodes((previous) => computed.map((n) => {
      const old = previous.find((o) => o.id === n.id);
      return old ? { ...old, type: n.type, position: n.position, data: n.data } : n;
    }));
  }, [computedKey]);

  const edges = graph.edges.map((e) => {
    const active = nodeStates.get(e.target)?.status === 'running';
    return {
      id: e.id,
      source: e.source,
      target: e.target,
      ...edgeHandles(e, rank, positions),
      label: shortLabel(e.label),
      labelStyle: { fontSize: 10, fill: 'var(--muted)' },
      labelBgPadding: [3, 1],
      animated: active,
      className: active ? 'edge-active' : '',
      style: e.conditional ? { strokeDasharray: '5 4' } : undefined,
      markerEnd: { type: MarkerType.ArrowClosed }
    };
  });

  return (
    <div className="flow-wrapper">
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange}
        fitView fitViewOptions={{ padding: 0.2 }}
        nodesDraggable={false} nodesConnectable={false} panOnScroll={false} zoomOnScroll={false}
        minZoom={0.4} maxZoom={1.5}
        onNodeClick={(_, node) => node.type === 'agent' && onNodeClick(node.id)} />
    </div>
  );
}
