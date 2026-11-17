import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  applyNodeChanges, BaseEdge, EdgeLabelRenderer, Handle, MarkerType, Position, ReactFlow, useInternalNode
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { formatCost, formatDuration, formatTokens, STATUS_LABELS } from '../../observability/runState.js';

const NODE_WIDTH = 150;   // keep in sync with .flow-node in style.css
const NODE_HEIGHT = 64;   // estimate; edges use the measured size
const TERMINAL_WIDTH = 60;
const TERMINAL_HEIGHT = 22;

// Hand-made layout (centres) so the loops can be read at a glance:
//   designer -> validator -> render -> designer   (left triangle: change, check, look)
//   designer <-> critic, render <-> critic        (right triangle: review)
//   designer <-> research, research <-> approval / download below
const COL = 200;
const LAYOUT = {
  __start__: [0, 0],
  orchestrator: [0, 90],
  layout_validator: [-COL, 220],
  render_scene: [0, 220],
  interior_critic: [COL, 220],
  interior_designer: [0, 370],
  furniture_research: [0, 520],
  human_approval: [-COL, 650],
  asset_download: [COL, 650],
  __end__: [0, 770]
};

// Long edges run in vertical lanes outside the nodes instead of across them. `sy` / `ty` shift the
// start / end vertically so edges that share a node side do not overlap.
const LANES = {
  'orchestrator>furniture_research': { side: 'left', x: -COL - 110, sy: 8 },
  'orchestrator>__end__': { side: 'left', x: -COL - 145, sy: -8 },
  'interior_designer>__end__': { side: 'right', x: COL + 110, ty: -5 },
  'interior_critic>__end__': { side: 'right', x: COL + 145, ty: 5 }
};

// Routing keys as short edge labels; edges whose target already says it stay unlabelled.
const EDGE_LABELS = {
  REJECT: 'reject', SEARCH_FURNITURE: 'search', 'DESIGN_ROOM / CRITIQUE_ROOM': 'design / critique',
  validate: 'validate', render: 'render', request_render: 'render', request_critique: 'critique',
  request_furniture: 'furniture', finish: 'finish', end: 'done', awaiting_approval: 'approval',
  download: 'download', approve: 'approve', 'reject / skip': 'reject / skip',
  interior_designer: '', interior_critic: ''
};

function AgentNode({ data }) {
  return (
    <div className={`flow-node kind-${data.kind} status-${data.status} ${data.highlighted ? 'highlighted' : ''}`}
      title={data.description}>
      <Handle type="target" position={Position.Top} id="in" />
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
        <div className="flow-node-meta mono"><span>⏸ waited {formatDuration(data.waitMs)}</span></div>
      )}
      {data.tokens > 0 && (
        <div className="flow-node-meta mono">
          <span>{formatTokens(data.tokens)} tokens</span>
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
      {data.label}
      {data.kind === 'start' && <Handle type="source" position={Position.Bottom} id="out" />}
    </div>
  );
}

function rectOf(node) {
  const { x, y } = node.internals.positionAbsolute;
  const w = node.measured?.width ?? 0;
  const h = node.measured?.height ?? 0;
  return { cx: x + w / 2, cy: y + h / 2, hw: w / 2, hh: h / 2 };
}

// Where a ray from p (inside the rectangle) in direction d leaves the rectangle.
function exitPoint(r, p, d) {
  const tx = d.x > 0 ? (r.cx + r.hw - p.x) / d.x : d.x < 0 ? (r.cx - r.hw - p.x) / d.x : Infinity;
  const ty = d.y > 0 ? (r.cy + r.hh - p.y) / d.y : d.y < 0 ? (r.cy - r.hh - p.y) / d.y : Infinity;
  const t = Math.min(tx, ty);
  return { x: p.x + d.x * t, y: p.y + d.y * t };
}

// Edges go straight from border to border (or through a lane). Edges with a reverse partner are
// shifted to their own side, so "there and back" shows as two parallel arrows.
function FlowEdge({ source, target, markerEnd, style, label, data }) {
  const s = useInternalNode(source);
  const t = useInternalNode(target);
  if (!s?.measured?.width || !t?.measured?.width) return null;
  const a = rectOf(s);
  const b = rectOf(t);
  let path;
  let labelAt;
  if (data.lane) {
    const { side, x: laneX, sy = 0, ty = 0 } = data.lane;
    const sign = side === 'left' ? -1 : 1;
    const x1 = a.cx + sign * a.hw;
    const y1 = a.cy + sy;
    const x2 = b.cx + sign * b.hw;
    const y2 = b.cy + ty;
    path = `M ${x1} ${y1} H ${laneX} V ${y2} H ${x2}`;
    labelAt = { x: laneX, y: (y1 + y2) / 2 };
  } else {
    const dx = b.cx - a.cx;
    const dy = b.cy - a.cy;
    const length = Math.hypot(dx, dy) || 1;
    const d = { x: dx / length, y: dy / length };
    const n = { x: -d.y, y: d.x };
    const off = data.paired ? 7 : 0;
    const start = exitPoint(a, { x: a.cx + n.x * off, y: a.cy + n.y * off }, d);
    const end = exitPoint(b, { x: b.cx + n.x * off, y: b.cy + n.y * off }, { x: -d.x, y: -d.y });
    path = `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
    const shift = data.paired ? 13 : 0;
    labelAt = { x: (start.x + end.x) / 2 + n.x * shift, y: (start.y + end.y) / 2 + n.y * shift };
  }
  return (<>
    <BaseEdge path={path} markerEnd={markerEnd} style={style} />
    {label && (
      <EdgeLabelRenderer>
        <div className="flow-edge-label" style={{ transform: `translate(-50%, -50%) translate(${labelAt.x}px, ${labelAt.y}px)` }}>
          {label}
        </div>
      </EdgeLabelRenderer>
    )}
  </>);
}

const nodeTypes = { agent: AgentNode, terminal: TerminalNode };
const edgeTypes = { flow: FlowEdge };

// Top-left positions from the hand-made layout; nodes added later get a row of their own at the bottom.
function layoutPositions(graph) {
  const positions = new Map();
  const extra = graph.nodes.filter((n) => !LAYOUT[n.id]);
  const bottom = Math.max(...Object.values(LAYOUT).map(([, y]) => y)) + 120;
  graph.nodes.forEach((n) => {
    const terminal = n.kind === 'start' || n.kind === 'end';
    const index = extra.indexOf(n);
    const [cx, cy] = LAYOUT[n.id] ?? [(index - (extra.length - 1) / 2) * COL, bottom];
    const [w, h] = terminal ? [TERMINAL_WIDTH, TERMINAL_HEIGHT] : [NODE_WIDTH, NODE_HEIGHT];
    positions.set(n.id, { x: cx - w / 2, y: cy - h / 2 });
  });
  return positions;
}

export default function WorkflowGraph({ graph, nodeStates, runStatus, highlightedNode, onNodeClick, large = false }) {
  const positions = useMemo(() => layoutPositions(graph), [graph]);
  const [nodes, setNodes] = useState([]);
  // React Flow stores measured sizes on the nodes; keep them by applying its changes.
  const onNodesChange = useCallback((changes) => setNodes((list) => applyNodeChanges(changes, list)), []);

  const computed = graph.nodes.map((n) => {
    const terminal = n.kind === 'start' || n.kind === 'end';
    const state = nodeStates.get(n.id);
    let status = state?.status ?? 'idle';
    if (n.kind === 'start' && runStatus !== 'idle') status = 'completed';
    if (n.kind === 'end' && runStatus === 'completed') status = 'completed';
    return {
      id: n.id,
      type: terminal ? 'terminal' : 'agent',
      position: positions.get(n.id),
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

  const pairs = new Set(graph.edges.map((e) => `${e.source}>${e.target}`));
  const edges = graph.edges.map((e) => {
    const key = `${e.source}>${e.target}`;
    const active = nodeStates.get(e.target)?.status === 'running';
    return {
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: 'out',
      targetHandle: 'in',
      type: 'flow',
      label: EDGE_LABELS[e.label] ?? e.label ?? undefined,
      data: { lane: LANES[key], paired: pairs.has(`${e.target}>${e.source}`) },
      animated: active,
      className: active ? 'edge-active' : '',
      style: e.conditional ? { strokeDasharray: '5 4' } : undefined,
      markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 }
    };
  });

  return (
    <div className={`flow-wrapper ${large ? 'large' : ''}`}>
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes} onNodesChange={onNodesChange}
        fitView fitViewOptions={{ padding: 0.08 }}
        nodesDraggable={false} nodesConnectable={false} panOnScroll={false} zoomOnScroll={large}
        minZoom={0.3} maxZoom={1.8}
        onNodeClick={(_, node) => node.type === 'agent' && onNodeClick(node.id)} />
    </div>
  );
}
