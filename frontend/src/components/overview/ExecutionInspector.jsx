import JsonView from './JsonView.jsx';
import CritiqueView from '../CritiqueView.jsx';
import {
  activeDuration, formatDuration, formatTime, liveDuration, STATUS_LABELS, waitingDuration
} from '../../observability/runState.js';

const fmt = (v) => (typeof v === 'number' ? v.toFixed(2) : '–');
// Positions as (x, y, z); y is the height above the floor (0 for older events without it).
const fmtPos = (p) => `(${fmt(p.x)}, ${fmt(p.y ?? 0)}, ${fmt(p.z)})`;

function LayoutChanges({ changes }) {
  const layout = changes?.layout;
  if (!layout) return changes?.keys?.length ? <p className="small">Changed fields: {changes.keys.join(', ')}</p> : null;
  const changed = layout.filter((c) => c.moved || c.rotated);
  return (
    <div className="changes">
      {changed.length === 0 && <p className="muted small">No furniture changed.</p>}
      {changed.map((c) => (
        <div key={c.id} className="change">
          <strong>{c.name}</strong>
          {c.moved && (
            <div className="small">Position (x, y, z): {fmtPos(c.before)} → {fmtPos(c.after)}</div>
          )}
          {c.rotated && <div className="small">Rotation: {c.before.rotationDeg}° → {c.after.rotationDeg}°</div>}
        </div>
      ))}
      {layout.length > changed.length && (
        <p className="muted small">{layout.length - changed.length} piece(s) unchanged.</p>
      )}
    </div>
  );
}

const VIEW_NAMES = { top_view: 'Top view', perspective_view: 'Perspective' };
const VALIDATION_LABELS = { valid: 'Valid', auto_corrected: 'Auto-corrected', invalid: 'Invalid' };
const VALIDATION_CLASS = { valid: 'completed', auto_corrected: 'completed', invalid: 'failed' };

function ProblemList({ problems }) {
  return (
    <ul className="problem-list small">
      {problems.map((p, i) => (
        <li key={i}><span className="badge">{p.type}</span> {p.message}</li>
      ))}
    </ul>
  );
}

function ValidationDetails({ validation, durationMs }) {
  const { stats } = validation;
  return (
    <div className="validation">
      <div className="kv small">
        <span>Result</span>
        <span className={`status-text status-${VALIDATION_CLASS[validation.status]}`}>{VALIDATION_LABELS[validation.status]}</span>
        <span>Checked</span><span>{stats.checked} pieces</span>
        <span>Collisions</span><span>{stats.collisions}</span>
        <span>Out of bounds</span><span>{stats.boundaryViolations}</span>
        <span>Corrections</span><span>{stats.corrections}</span>
        <span>Unresolved</span><span>{stats.unresolved}</span>
        <span>Duration</span><span className="mono">{formatDuration(durationMs)}</span>
      </div>
      {validation.corrections.length > 0 && (<>
        <h5>Automatic corrections</h5>
        <ul className="problem-list small">
          {validation.corrections.map((c, i) => (
            <li key={i}>
              <strong>{c.name}</strong>: {fmtPos(c.before)} → {fmtPos(c.after)},
              {' '}{(c.distance * 100).toFixed(1)} cm – {c.reason}
            </li>
          ))}
        </ul>
      </>)}
      {validation.errors.length > 0 && (<>
        <h5>Unresolved errors</h5>
        <ProblemList problems={validation.errors} />
      </>)}
      {validation.problems.length > 0 && validation.status !== 'invalid' && (<>
        <h5>Problems found</h5>
        <ProblemList problems={validation.problems} />
      </>)}
    </div>
  );
}

function LlmCall({ call }) {
  const usage = call.usage || {};
  return (
    <div className="llm-call small">
      <div className="kv">
        <span>Model</span><span>{call.resolvedModel || call.model}{call.provider ? ` · ${call.provider}` : ''}</span>
        <span>Duration</span><span>{formatDuration(call.durationMs)}</span>
        {call.usage && (<>
          <span>Tokens</span>
          <span>
            {usage.promptTokens ?? '–'} in · {usage.completionTokens ?? '–'} out
            {usage.reasoningTokens ? ` (incl. ${usage.reasoningTokens} reasoning)` : ''}
          </span>
        </>)}
        {call.cost != null && (<><span>Cost</span><span>${call.cost.toFixed(5)}</span></>)}
        <span>Attempts</span><span>{call.attempts?.length ?? 1}{call.retries ? ` (${call.retries} ${call.retries === 1 ? 'retry' : 'retries'})` : ''}</span>
      </div>
      {call.attempts?.filter((a) => a.error).map((a) => (
        <p key={a.attempt} className="error small">Attempt {a.attempt}: {a.error}</p>
      ))}
      {call.error && <p className="error small">{call.error}</p>}
    </div>
  );
}

const TOOL_NAMES = {
  search_assets: 'MCP search', get_asset: 'MCP details', view_preview: 'View preview', search_local_furniture: 'Local search'
};
const ACTION_LABELS = { approve: 'Approved', reject: 'Nothing fits', skip: 'Skipped' };
const ITEM_STATUS = {
  pending: 'pending', searching_local: 'local search', searching_online: 'online search', retrying: 'second search',
  awaiting_approval: 'waiting for approval', downloading: 'downloading', completed: 'downloaded',
  found_locally: 'in the inventory', failed: 'download failed', skipped: 'skipped', rejected: 'rejected',
  not_found: 'not found'
};

// The furniture list as the research step left it: one line per item with its status.
function ResearchItems({ items }) {
  return (
    <ul className="problem-list small">
      {items.map((item) => (
        <li key={item.key}>
          <span className="badge">{ITEM_STATUS[item.status] ?? item.status}</span>{' '}
          {item.index}. {item.request.description}
          {item.request.essential && <span className="badge warn">essential</span>}
          {item.run > 0 && <span className="muted"> · run {item.run}</span>}
          {item.downloads?.length > 0 && (
            <span className="muted"> · {item.downloads.map((d) => `${d.name} (${d.status})`).join(', ')}</span>
          )}
          {item.local?.length > 0 && <span className="muted"> · {item.local.map((l) => l.name).join(', ')}</span>}
          {item.note && <span className="muted"> · {item.note}</span>}
        </li>
      ))}
    </ul>
  );
}

function ToolCalls({ tools }) {
  return (
    <ol className="stages tools">
      {tools.map((t, i) => (
        <li key={i} className={`status-${t.status}`}>
          <span className="status-dot" />
          <span className="stage-label">
            {TOOL_NAMES[t.tool] ?? t.tool}{' '}
            <span className="muted">{t.input?.query ? `“${t.input.query}”` : t.input?.asset_id}</span>
            {t.resultCount != null && <span className="muted"> → {t.resultCount} {t.resultCount === 1 ? 'result' : 'results'}</span>}
            {t.tool === 'view_preview' && t.note && t.note !== 'ok' && <span className="muted"> ({t.note})</span>}
            {t.error && <span className="error"> {t.error}</span>}
          </span>
          <span className="mono muted">{formatDuration(t.durationMs)}</span>
        </li>
      ))}
    </ol>
  );
}

function Candidates({ candidates }) {
  return (
    <ul className="problem-list small">
      {candidates.map((c) => (
        <li key={c.assetId}><strong>{c.name}</strong> <span className="muted">({c.provider})</span> – {c.reason}</li>
      ))}
    </ul>
  );
}

function Downloads({ result }) {
  return (
    <ul className="problem-list small">
      {result.added.map((a) => (
        <li key={a.assetId}>✓ {a.name}{a.unitScale !== 1 ? ` · unit detected, scale ${a.unitScale}` : ''}{a.warning ? ` · ${a.warning}` : ''}</li>
      ))}
      {result.existing.map((a) => <li key={a.assetId}>= {a.name} (already in the inventory)</li>)}
      {result.failed.map((a) => <li key={a.assetId} className="error">✗ {a.name}: {a.error}</li>)}
    </ul>
  );
}

export default function ExecutionInspector({ execution, labels, now, clockOffset }) {
  if (!execution) return <p className="muted small">Select an entry in the timeline or a node in the graph.</p>;
  const duration = activeDuration(execution, now, clockOffset);
  const waited = waitingDuration(execution, now, clockOffset);
  const label = labels.get(execution.node) ?? execution.node;
  const output = execution.output;

  return (
    <div className="inspector">
      <div className="kv small">
        <span>Node</span><span>{label}</span>
        <span>Execution</span><span className="mono" title={execution.executionId}>{execution.executionId.slice(0, 13)}…</span>
        <span>Status</span><span className={`status-text status-${execution.status}`}>{STATUS_LABELS[execution.status]}</span>
        <span>Start</span><span className="mono">{formatTime(execution.startedAt)}</span>
        <span>End</span><span className="mono">{formatTime(execution.endedAt)}</span>
        <span>Duration (active)</span><span className="mono">{formatDuration(duration)}</span>
        {waited > 0 && (<><span>Waited</span><span className="mono">{formatDuration(waited)}</span></>)}
        {execution.output?.intent && (<><span>Intent</span><span className="badge ok">{execution.output.intent}</span></>)}
        {execution.next && (<><span>Next node</span><span>{labels.get(execution.next) ?? execution.next}</span></>)}
      </div>
      {execution.error && <p className="error small">{execution.error}</p>}
      {execution.output?.message && <p className="small notice-inline">{execution.output.message}</p>}

      <h4>Progress</h4>
      <ol className="stages">
        {execution.stages.map((s, i) => (
          <li key={i} className={`status-${s.status}`}>
            <span className="status-dot" />
            <span className="stage-label">{s.label}</span>
            <span className="mono muted">
              {s.status === 'running' ? formatDuration(liveDuration(s.startedAt, null, null, now, clockOffset)) : s.durationMs ? formatDuration(s.durationMs) : ''}
            </span>
          </li>
        ))}
        {execution.stages.length === 0 && <li className="muted">No steps yet.</li>}
      </ol>

      {execution.status === 'waiting' && execution.request && (
        <p className="small status-text status-waiting">
          Waiting for a choice among {execution.request.candidates?.length ?? 0} suggestions (see above).
        </p>
      )}

      {execution.tools.length > 0 && (<>
        <h4>Tool calls</h4>
        <ToolCalls tools={execution.tools} />
      </>)}

      {output?.designer_action && (<>
        <h4>Designer decision</h4>
        <p className="small">
          Next action: <span className="badge">{output.designer_action}</span>
          {output.finish_when_valid ? ' · finish once valid' : ''}
          {output.layout_version != null ? ` · layout v${output.layout_version}` : ''}
        </p>
        {output.furniture_request?.items?.length > 0 && (
          <p className="small muted">Furniture request: {output.furniture_request.items.map((i) => i.description).join('; ')}</p>
        )}
      </>)}

      {output?.critique && (<>
        <h4>Critique{output.critiqued_layout_version != null ? ` (layout v${output.critiqued_layout_version})` : ''}</h4>
        <CritiqueView critique={output.critique} compact />
      </>)}

      {output?.workflow_log?.length > 0 && (
        <p className="small muted">{output.workflow_log.join(' · ')}</p>
      )}

      {output?.research_items?.length > 0 && (<>
        <h4>Furniture list</h4>
        <ResearchItems items={output.research_items} />
      </>)}

      {output?.approval_item && (() => {
        const item = output.research_items?.find((i) => i.key === output.approval_item);
        return item?.candidates?.length > 0 && (<>
          <h4>Proposed assets for item {item.index}</h4>
          {item.summary && <p className="small">{item.summary}</p>}
          <Candidates candidates={item.candidates} />
        </>);
      })()}

      {output?.approval && (<>
        <h4>Decision</h4>
        <p className="small">
          {ACTION_LABELS[output.approval.action] ?? output.approval.action}
          {output.approval.selected.length > 0 && `: ${output.approval.selected.join(', ')}`}
        </p>
        {output.approval.feedback && <p className="small muted">Feedback: “{output.approval.feedback}”</p>}
      </>)}

      {output?.download_result && (<>
        <h4>Downloads</h4>
        <Downloads result={output.download_result} />
      </>)}

      {execution.llmCalls.length > 0 && (<>
        <h4>LLM</h4>
        {execution.llmCalls.map((call, i) => <LlmCall key={i} call={call} />)}
      </>)}

      {execution.output?.scene_images && (<>
        <h4>Screenshots</h4>
        <div className="screenshots">
          {Object.entries(execution.output.scene_images).map(([view, ref]) => (
            <a key={view} href={`/api/${ref}`} target="_blank" rel="noreferrer" title="Open full size">
              <img src={`/api/${ref}`} alt={VIEW_NAMES[view] ?? view} />
              <span className="small muted">{VIEW_NAMES[view] ?? view}</span>
            </a>
          ))}
        </div>
      </>)}

      {execution.output?.validation && (<>
        <h4>Validation</h4>
        <ValidationDetails validation={execution.output.validation} durationMs={duration} />
      </>)}

      {execution.changes && (<>
        <h4>State changes</h4>
        <LayoutChanges changes={execution.changes} />
      </>)}

      {execution.output?.explanation && (<>
        <h4>Explanation</h4>
        <p className="small">{execution.output.explanation}</p>
      </>)}

      <details className="section">
        <summary>Input (state)</summary>
        <JsonView data={execution.input} />
      </details>
      <details className="section">
        <summary>Output</summary>
        <JsonView data={execution.output} />
      </details>
      {execution.stages.some((s) => s.data && Object.keys(s.data).length) && (
        <details className="section">
          <summary>Step details</summary>
          <JsonView data={Object.fromEntries(execution.stages.filter((s) => s.data).map((s) => [s.stage, s.data]))} />
        </details>
      )}
    </div>
  );
}
