import { useEffect, useMemo, useState } from 'react';
import { api } from '../../api.js';
import { useNow } from '../../observability/useNow.js';
import {
  activeDuration, formatCost, formatDuration, formatTime, formatTokens, liveDuration, STATUS_LABELS, waitingDuration
} from '../../observability/runState.js';
import WorkflowGraph from './WorkflowGraph.jsx';
import ExecutionInspector from './ExecutionInspector.jsx';
import ApprovalPanel from './ApprovalPanel.jsx';

export default function OverviewPanel({ designRun }) {
  const { run, runId, clockOffset, connectionError, respond, cancel, cancelling, active } = designRun;
  const [graph, setGraph] = useState(null);
  const [graphError, setGraphError] = useState('');
  const [selectedId, setSelectedId] = useState(null);
  const now = useNow(run.status === 'running' || run.status === 'waiting');

  useEffect(() => {
    api.designGraph().then(setGraph).catch((err) => setGraphError(err.message));
  }, []);

  // A new run starts with "follow the latest execution" again.
  useEffect(() => setSelectedId(null), [runId]);

  const labels = useMemo(() => new Map((graph?.nodes ?? []).map((n) => [n.id, n.label])), [graph]);
  const selected = run.executions.find((e) => e.executionId === selectedId) ?? run.executions.at(-1) ?? null;

  const nodeStates = new Map();
  for (const [node, ex] of run.latestByNode) {
    const nodeExecutions = run.executions.filter((e) => e.node === node);
    nodeStates.set(node, {
      status: ex.status,
      durationMs: activeDuration(ex, now, clockOffset),
      waitMs: waitingDuration(ex, now, clockOffset),
      runs: nodeExecutions.length,
      // Summed over all executions of this node, so loops show their full cost.
      tokens: nodeExecutions.reduce((sum, e) => sum + e.usage.totalTokens, 0),
      cost: nodeExecutions.some((e) => e.usage.hasCost)
        ? nodeExecutions.reduce((sum, e) => sum + e.usage.cost, 0) : null
    });
  }
  const total = liveDuration(run.startedAt, run.endedAt, run.durationMs, now, clockOffset);
  const waited = waitingDuration(run, now, clockOffset);
  const { usage } = run;
  const downloads = run.result?.downloads;
  // Models of the other agents (orchestrator, critic, research), shown below the designer's model.
  const runModels = run.models || {};
  const otherModels = [...new Set(Object.entries(runModels)
    .filter(([agent]) => agent !== 'interior_designer').map(([, m]) => m))];

  const selectNode = (nodeId) => {
    const latest = run.latestByNode.get(nodeId);
    if (latest) setSelectedId(latest.executionId);
  };

  return (
    <div className="panel overview">
      {run.status === 'waiting' && run.approval && (
        <ApprovalPanel key={`${runId}:${run.approval.round}`} approval={run.approval} onRespond={respond}
          onAbort={cancel} aborting={cancelling} />
      )}
      {active && (
        <div className="abort-bar">
          <span className="small muted">Prompt sent too early, wrong settings or stuck in a loop?</span>
          <button type="button" className="danger" onClick={cancel} disabled={cancelling}
            title="Stop the run for good; nothing of it is applied">
            {cancelling ? 'Aborting…' : 'Abort run'}
          </button>
        </div>
      )}

      <section className="run-summary">
        <div className="summary-grid">
          <div><span className="muted small">Status</span><strong className={`status-text status-${run.status}`}>
            {run.status === 'running' && <span className="spinner" />} {STATUS_LABELS[run.status]}</strong></div>
          <div>
            <span className="muted small">Total duration</span><strong className="mono">{formatDuration(total)}</strong>
            {waited > 0 && (
              <span className="muted small mono">active {formatDuration(Math.max((total ?? 0) - waited, 0))} · waiting {formatDuration(waited)}</span>
            )}
          </div>
          <div>
            <span className="muted small">Model (designer)</span>
            <strong className="truncate" title={Object.entries(runModels).map(([agent, m]) => `${agent}: ${m}`).join('\n') || (run.model ?? '')}>
              {run.model ?? '–'}
            </strong>
            {otherModels.length > 0 && (
              <span className="muted small truncate" title={otherModels.join(', ')}>others: {otherModels.join(', ')}</span>
            )}
          </div>
          <div><span className="muted small">Nodes</span><strong>{run.counts.completed} ✓ · {run.counts.failed} ✗</strong></div>
          <div>
            <span className="muted small">Tokens</span>
            <strong className="mono" title={usage.calls ? `${usage.promptTokens} in · ${usage.completionTokens} out · ${usage.reasoningTokens} reasoning` : ''}>
              {formatTokens(usage.totalTokens)}
            </strong>
            {usage.calls > 0 && (
              <span className="muted small">
                {formatTokens(usage.promptTokens)} in · {formatTokens(usage.completionTokens)} out
              </span>
            )}
          </div>
          <div>
            <span className="muted small">Cost</span>
            <strong className="mono">{usage.hasCost ? formatCost(usage.cost) : '–'}</strong>
            {usage.calls > 0 && <span className="muted small">{usage.calls} LLM call{usage.calls === 1 ? '' : 's'}</span>}
          </div>
        </div>
        {run.request && <p className="small request">“{run.request}”</p>}
        {run.intent && <p className="small">Intent: <span className="badge ok">{run.intent}</span></p>}
        {connectionError && <p className="error small">{connectionError}</p>}
        {run.error && <p className="error small">{run.error}</p>}
        {run.status === 'completed' && run.result?.message && (
          <p className={`small ${downloads?.added?.length ? 'result-line' : 'notice-inline'}`}>{run.result.message}</p>
        )}
        {run.status === 'completed' && downloads?.added?.length > 0 && (
          <p className="small muted">The new furniture is ready in the inventory.</p>
        )}
        {run.status === 'completed' && run.result?.kind === 'critique' && run.result.critique && (
          <p className="small notice-inline">{run.result.critique.summary}</p>
        )}
        {run.status === 'completed' && run.result?.kind === 'design' && !run.result.message && (
          <p className="small result-line">Layout with {run.result.placements.length} pieces applied.
            {run.result.warnings.length > 0 && ` ${run.result.warnings.length} warning(s).`}</p>
        )}
        {run.status === 'idle' && <p className="muted small">Start a request in the “Design” tab to follow its execution here live.</p>}
      </section>

      <section>
        <h3 className="section-title">Workflow</h3>
        {graphError && <p className="error small">{graphError}</p>}
        {graph && (
          <WorkflowGraph graph={graph} nodeStates={nodeStates} runStatus={run.status}
            highlightedNode={selected?.node} onNodeClick={selectNode} />
        )}
      </section>

      <section>
        <h3 className="section-title">Timeline</h3>
        {run.executions.length === 0 && <p className="muted small">No executions yet.</p>}
        <ol className="timeline">
          {run.executions.map((ex) => {
            const wait = waitingDuration(ex, now, clockOffset);
            return (
              <li key={ex.executionId}>
                <button className={`timeline-entry status-${ex.status} ${selected?.executionId === ex.executionId ? 'selected' : ''}`}
                  onClick={() => setSelectedId(ex.executionId)}>
                  <span className="order">{ex.order}</span>
                  <span className="status-dot" />
                  <span className="timeline-name">{labels.get(ex.node) ?? ex.node}</span>
                  <span className="mono muted small">{formatTime(ex.startedAt)}</span>
                  <span className="mono small">{formatDuration(activeDuration(ex, now, clockOffset))}</span>
                  {ex.status === 'waiting' && <span className="timeline-usage small status-text status-waiting">Waiting for selection · {formatDuration(wait)}</span>}
                  {ex.status !== 'waiting' && wait > 0 && <span className="timeline-usage mono muted small">Waited {formatDuration(wait)}</span>}
                  {ex.summary && <span className="timeline-usage small">{ex.summary}</span>}
                  {ex.usage.calls > 0 && (
                    <span className="timeline-usage mono muted small">
                      {formatTokens(ex.usage.totalTokens)} tokens{ex.usage.hasCost ? ` · ${formatCost(ex.usage.cost)}` : ''}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ol>
      </section>

      <section>
        <h3 className="section-title">Details</h3>
        <ExecutionInspector execution={selected} labels={labels}
          now={now} clockOffset={clockOffset} />
      </section>
    </div>
  );
}
