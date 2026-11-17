// Derives everything the Overview shows from the raw event list of one workflow run.
// Events are the single source of truth; nothing here is stored separately.

export const TERMINAL_EVENTS = new Set(['workflow.completed', 'workflow.failed', 'workflow.cancelled']);

function upsertStage(stages, payload, ts) {
  const open = [...stages].reverse().find((s) => s.stage === payload.stage && s.status === 'running');
  if (open && payload.status !== 'running') {
    Object.assign(open, { status: payload.status, endedAt: ts, durationMs: payload.durationMs, data: payload.data, error: payload.error });
  } else {
    stages.push({ ...payload, startedAt: ts, endedAt: payload.status === 'running' ? null : ts });
  }
}

const emptyUsage = () => ({ promptTokens: 0, completionTokens: 0, reasoningTokens: 0, totalTokens: 0, cost: 0, calls: 0, hasCost: false });

// Adds one LLM call's token usage and cost to a running total.
function addUsage(total, call) {
  const usage = call.usage || {};
  total.calls += 1;
  total.promptTokens += usage.promptTokens || 0;
  total.completionTokens += usage.completionTokens || 0;
  total.reasoningTokens += usage.reasoningTokens || 0;
  total.totalTokens += usage.totalTokens || 0;
  if (call.cost != null) {
    total.cost += call.cost;
    total.hasCost = true;
  }
}

export function deriveRun(events) {
  const run = {
    status: 'idle',
    startedAt: null,
    endedAt: null,
    durationMs: null,
    model: null,
    models: {},           // agent node name -> model id
    request: null,
    intent: null,
    result: null,
    error: null,
    executions: [],
    latestByNode: new Map(),
    counts: { completed: 0, failed: 0, running: 0, waiting: 0 },
    usage: emptyUsage(),
    approval: null,       // pending human-in-the-loop request (one furniture item) while status is 'waiting'
    research: null,       // latest furniture list progress {counts, total}
    waitMs: 0,            // finished waiting periods; the current one is added live
    waitingSince: null
  };
  const executions = new Map();

  for (const e of events) {
    const p = e.payload || {};
    const ex = e.executionId ? executions.get(e.executionId) : null;
    switch (e.type) {
      case 'workflow.started':
        Object.assign(run, { status: 'running', startedAt: e.ts, model: p.model, models: p.models || {}, request: p.request });
        break;
      case 'workflow.completed':
        Object.assign(run, { status: 'completed', endedAt: e.ts, durationMs: p.durationMs, result: p.result });
        break;
      case 'workflow.failed':
        Object.assign(run, { status: 'failed', endedAt: e.ts, durationMs: p.durationMs, error: p.error });
        break;
      case 'workflow.cancelled':
        Object.assign(run, { status: 'cancelled', endedAt: e.ts, durationMs: p.durationMs, error: 'Cancelled', approval: null });
        if (run.waitingSince != null) run.waitMs = p.waitMs ?? run.waitMs + (e.ts - run.waitingSince);
        run.waitingSince = null;
        break;
      case 'workflow.waiting':
        Object.assign(run, { status: 'waiting', approval: p.approval, waitingSince: e.ts });
        break;
      case 'workflow.resumed':
        run.waitMs += p.waitMs ?? (e.ts - run.waitingSince);
        Object.assign(run, { status: 'running', approval: null, waitingSince: null, lastDecision: p.decision });
        break;
      case 'node.waiting':
        if (ex) Object.assign(ex, { status: 'waiting', waitingSince: e.ts, activeMs: p.activeMs, request: p.request?.[0] });
        break;
      case 'node.resumed':
        if (ex) {
          ex.waitMs += p.waitMs ?? (e.ts - ex.waitingSince);
          Object.assign(ex, { status: 'running', waitingSince: null, resumedAt: e.ts });
        }
        break;
      case 'node.tool':
        if (ex) ex.tools.push({ ...p, ts: e.ts });
        break;
      case 'node.started':
        executions.set(e.executionId, {
          executionId: e.executionId,
          node: e.node,
          order: executions.size + 1,
          status: 'running',
          startedAt: e.ts,
          endedAt: null,
          durationMs: null,
          input: p.input,
          triggers: p.triggers,
          output: null,
          changes: null,
          error: null,
          stages: [],
          llmCalls: [],
          tools: [],
          usage: emptyUsage(),
          activeMs: 0,       // active time before a pause
          waitMs: 0,
          waitingSince: null,
          resumedAt: null
        });
        break;
      case 'node.completed':
      case 'node.failed':
        if (ex) {
          Object.assign(ex, {
            status: e.status, endedAt: e.ts, durationMs: p.durationMs,
            output: p.output ?? null, changes: p.changes ?? null, error: p.error ?? null,
            summary: p.summary ?? null, next: p.next ?? null, waitMs: p.waitMs ?? ex.waitMs, waitingSince: null
          });
          if (p.output?.intent) run.intent = p.output.intent;
          if (e.node === 'furniture_research' && p.output?.research_status === 'done') run.research = null;
          // Stages still open when the node ended were interrupted.
          ex.stages.forEach((s) => {
            if (s.status === 'running') Object.assign(s, { status: 'failed', endedAt: e.ts, durationMs: e.ts - s.startedAt });
          });
        }
        break;
      case 'node.progress':
        if (ex) upsertStage(ex.stages, p, e.ts);
        // Latest state of the furniture list (counts per item status, online budget used).
        if (p.stage === 'research_status') run.research = p.data;
        break;
      case 'node.llm':
        if (ex) {
          ex.llmCalls.push({ ...p, receivedAt: e.ts });
          addUsage(ex.usage, p);
        }
        addUsage(run.usage, p);
        break;
      default:
        break;
    }
  }

  run.executions = [...executions.values()];
  for (const ex of run.executions) {
    run.latestByNode.set(ex.node, ex);
    if (ex.status in run.counts) run.counts[ex.status] += 1;
  }
  return run;
}

// Nodes that work on one furniture list together. Downloads run in the background, so the graph
// alternates between them (research -> download -> research -> approval ...) for the same list.
const RESEARCH_NODES = new Set(['furniture_research', 'asset_download', 'human_approval']);
const GROUP_STATUS = ['failed', 'waiting', 'running', 'cancelled'];

// Timeline rows: single executions, or one group per furniture list (consecutive research nodes).
export function groupTimeline(executions) {
  const rows = [];
  for (const ex of executions) {
    const last = rows.at(-1);
    if (RESEARCH_NODES.has(ex.node) && last?.executions && RESEARCH_NODES.has(last.executions[0].node)) {
      last.executions.push(ex);
    } else {
      rows.push(RESEARCH_NODES.has(ex.node) ? { id: `group:${ex.executionId}`, executions: [ex] } : ex);
    }
  }
  return rows.map((row) => {
    if (!row.executions) return row;
    if (row.executions.length === 1) return row.executions[0];
    const list = row.executions;
    const statuses = new Set(list.map((e) => e.status));
    const usage = emptyUsage();
    for (const e of list) {
      for (const key of ['promptTokens', 'completionTokens', 'reasoningTokens', 'totalTokens', 'cost', 'calls']) usage[key] += e.usage[key];
      usage.hasCost ||= e.usage.hasCost;
    }
    const lastResearch = [...list].reverse().find((e) => e.node === 'furniture_research' && e.summary);
    return {
      ...row,
      status: GROUP_STATUS.find((s) => statuses.has(s)) ?? 'completed',
      startedAt: list[0].startedAt,
      endedAt: list.every((e) => e.endedAt != null) ? list.at(-1).endedAt : null,
      summary: lastResearch?.summary ?? null,
      usage
    };
  });
}

// Duration of something that may still be running, using server timestamps corrected by the clock offset.
export function liveDuration(startedAt, endedAt, durationMs, now, clockOffset) {
  if (durationMs != null) return durationMs;
  if (startedAt == null) return null;
  return Math.max((endedAt ?? now - clockOffset) - startedAt, 0);
}

// Active time of a node execution, without time spent waiting for the user.
export function activeDuration(ex, now, clockOffset) {
  if (ex.durationMs != null) return ex.durationMs;
  if (ex.status === 'waiting') return ex.activeMs;
  return ex.activeMs + Math.max(now - clockOffset - (ex.resumedAt ?? ex.startedAt), 0);
}

// Time a node execution or run has spent waiting for the user, including a pause still going on.
export function waitingDuration(item, now, clockOffset) {
  return item.waitMs + (item.waitingSince != null ? Math.max(now - clockOffset - item.waitingSince, 0) : 0);
}

export function formatDuration(ms) {
  if (ms == null) return '–';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`;
  const seconds = Math.round(ms / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')} min`;
}

export function formatTokens(n) {
  if (!n) return '–';
  return n >= 10000 ? `${(n / 1000).toFixed(1)}k` : n.toLocaleString('en-US');
}

export function formatCost(usd) {
  if (usd == null) return '–';
  return `$${usd < 0.01 ? usd.toFixed(4) : usd.toFixed(3)}`;
}

export function formatTime(ts) {
  if (ts == null) return '–';
  const d = new Date(ts);
  return `${d.toLocaleTimeString('en-GB', { hour12: false })}.${String(d.getMilliseconds()).padStart(3, '0')}`;
}

export const STATUS_LABELS = {
  idle: 'Ready',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
  waiting: 'Waiting for approval'
};
