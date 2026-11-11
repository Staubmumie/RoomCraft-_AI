import { useEffect, useState } from 'react';
import { api } from '../api.js';

// Storage keys keep the former product name so saved settings survive.
const MODEL_KEY = 'roomforge.designModel';
const AGENT_MODELS_KEY = 'roomforge.agentModels';
const OPTIONS_KEY = 'roomforge.designOptions';
const RESULT_TITLES = { design: 'Design', research: 'Furniture search', reject: 'Notice' };
const SCORE_LABELS = {
  functionality: 'Function', aesthetics: 'Aesthetics', space_utilization: 'Use of space', preference_fit: 'Fits the request'
};

const DEFAULT_OPTIONS = { allowNewFurniture: false, allowRemoval: false, finalReview: true, maxIterations: 8, maxResearch: 2 };
const MAX_ITERATIONS = 15;
const MAX_RESEARCH = 10;
const clampInt = (value, min, max, fallback) => {
  const n = Math.round(Number(value));
  return Number.isFinite(n) && value !== '' ? Math.min(Math.max(n, min), max) : fallback;
};

function loadSavedModel() {
  try { return localStorage.getItem(MODEL_KEY) || ''; } catch { return ''; }
}

// Per-agent overrides {agent: modelId}; agents without an entry use the backend's default.
function loadSavedAgentModels() {
  try {
    const saved = JSON.parse(localStorage.getItem(AGENT_MODELS_KEY) || '{}');
    return saved && typeof saved === 'object' && !Array.isArray(saved) ? saved : {};
  } catch { return {}; }
}

function loadSavedOptions() {
  try { return { ...DEFAULT_OPTIONS, ...JSON.parse(localStorage.getItem(OPTIONS_KEY) || '{}') }; } catch { return DEFAULT_OPTIONS; }
}

export default function DesignPanel({ furnitureCount, onGenerate, onCancel, cancelling, canUndo, onUndo }) {
  const [request, setRequest] = useState('');
  const [models, setModels] = useState([]);
  const [model, setModel] = useState(loadSavedModel);
  const [agents, setAgents] = useState([]);
  const [agentModels, setAgentModels] = useState(loadSavedAgentModels);
  const [designOptions, setDesignOptions] = useState(loadSavedOptions);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  useEffect(() => {
    api.designModels()
      .then((data) => {
        setModels(data.models);
        setModel((current) => current || data.default);
        // The main model drives the interior designer; the others are configured per agent.
        setAgents((data.agents ?? []).filter((a) => a.id !== 'interior_designer'));
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    try { if (model) localStorage.setItem(MODEL_KEY, model); } catch { /* storage unavailable */ }
  }, [model]);

  useEffect(() => {
    try { localStorage.setItem(AGENT_MODELS_KEY, JSON.stringify(agentModels)); } catch { /* storage unavailable */ }
  }, [agentModels]);

  useEffect(() => {
    try { localStorage.setItem(OPTIONS_KEY, JSON.stringify(designOptions)); } catch { /* storage unavailable */ }
  }, [designOptions]);

  const setOption = (key, value) => setDesignOptions((current) => ({ ...current, [key]: value }));
  // An empty room can still be furnished when inventory furniture may be added.
  const noFurniture = furnitureCount === 0 && !designOptions.allowNewFurniture;

  const runRequest = async (text) => {
    if (!text.trim() || busy) return;
    const maxIterations = clampInt(designOptions.maxIterations, 1, MAX_ITERATIONS, DEFAULT_OPTIONS.maxIterations);
    const maxResearch = clampInt(designOptions.maxResearch, 0, MAX_RESEARCH, DEFAULT_OPTIONS.maxResearch);
    setBusy(true);
    setError('');
    try {
      // Only real overrides of known agents are sent; everything else uses the backend default.
      const models = Object.fromEntries(agents.filter((a) => agentModels[a.id]).map((a) => [a.id, agentModels[a.id]]));
      setResult(await onGenerate({
        request: text.trim(), model, models, options: { ...designOptions, maxIterations, maxResearch }
      }));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const submit = (event) => {
    event.preventDefault();
    runRequest(request);
  };

  // Starts a furniture search for a piece the design could not provide; downloads still need approval.
  const searchMissing = (gap) => runRequest(`Find furniture for my room: ${gap.item} (${gap.reason}).`);

  // Keep the saved model selectable even if the backend no longer lists it.
  const withSaved = (id) => (id && !models.some((m) => m.id === id) ? [...models, { id, label: id }] : models);
  const options = withSaved(model);
  const modelLabel = (id) => models.find((m) => m.id === id)?.label ?? id;
  const overridden = agents.filter((a) => agentModels[a.id]);
  const agentSummary = overridden.length
    ? overridden.map((a) => `${a.label}: ${modelLabel(agentModels[a.id])}`).join(', ')
    : `all on default (${[...new Set(agents.map((a) => modelLabel(a.default)))].join(', ')})`;
  const added = result?.placements?.filter((p) => p.id.startsWith('new-')) ?? [];
  // Missing essentials are shown with a search action; their "Not fulfilled" warnings would repeat them.
  const missing = result?.missing ?? [];
  const otherWarnings = (result?.warnings ?? []).filter((w) => !(missing.length && w.startsWith('Not fulfilled:')));

  return (
    <div className="panel">
      <form className="design-form" onSubmit={submit}>
        <label className="field">
          <span>How should the room be designed?</span>
          <textarea rows={5} value={request} onChange={(e) => setRequest(e.target.value)} disabled={busy}
            placeholder="e.g. Make my living room creative and cozy." />
        </label>
        <label className="field">
          <span>Model (Interior Designer)</span>
          <select value={model} onChange={(e) => setModel(e.target.value)} disabled={busy}>
            {options.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        </label>
        {agents.length > 0 && (
          <details className="agent-models">
            <summary>
              Models of the other agents
              <span className="muted small"> · {agentSummary}</span>
            </summary>
            <fieldset disabled={busy}>
              {agents.map((agent) => (
                <label key={agent.id} className="field inline">
                  <span>{agent.label}</span>
                  <select value={agentModels[agent.id] ?? ''}
                    onChange={(e) => setAgentModels((current) => ({ ...current, [agent.id]: e.target.value }))}>
                    <option value="">Default ({modelLabel(agent.default)})</option>
                    {withSaved(agentModels[agent.id]).map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
                  </select>
                </label>
              ))}
            </fieldset>
          </details>
        )}
        <fieldset className="design-options" disabled={busy}>
          <label className="toggle">
            <input type="checkbox" checked={designOptions.allowNewFurniture}
              onChange={(e) => setOption('allowNewFurniture', e.target.checked)} />
            <span>
              Include more furniture
              <small className="muted">Furniture from the inventory (including newly researched pieces) may be added to the room.</small>
            </span>
          </label>
          <label className="toggle">
            <input type="checkbox" checked={designOptions.allowRemoval}
              onChange={(e) => setOption('allowRemoval', e.target.checked)} />
            <span>
              Allow removing furniture
              <small className="muted">Otherwise all furniture stays in the room. Pieces added during the run can always be removed again.</small>
            </span>
          </label>
          <label className="toggle">
            <input type="checkbox" checked={designOptions.finalReview}
              onChange={(e) => setOption('finalReview', e.target.checked)} />
            <span>
              Final review by the critic
              <small className="muted">When the designer is done, the critic reviews the validated layout once and the designer gets one more step to act on it. If those changes fail, the reviewed layout is kept.</small>
            </span>
          </label>
          <label className="field inline">
            <span className="option-label">
              Max. iterations
              <small className="muted">Designer steps (one LLM call each: plan, place, repair or decide what to do next). Validations, renders and critiques do not count, but the designer needs a step to react to each.</small>
            </span>
            <input type="number" min={1} max={MAX_ITERATIONS} value={designOptions.maxIterations}
              onChange={(e) => setOption('maxIterations', e.target.value)} />
          </label>
          <label className="field inline">
            <span className="option-label">
              Max. furniture searches
              <small className="muted">How often the designer may ask the furniture research agent for a missing piece (one piece per search, up to 3 suggestion rounds each). 0 = no search. Needs “Include more furniture”.</small>
            </span>
            <input type="number" min={0} max={MAX_RESEARCH} value={designOptions.maxResearch}
              disabled={!designOptions.allowNewFurniture}
              onChange={(e) => setOption('maxResearch', e.target.value)} />
          </label>
        </fieldset>
        <div className="card-actions">
          <button className="primary" disabled={busy || !request.trim()}>
            {busy ? <><span className="spinner" /> Designer is working…</> : 'Create layout'}
          </button>
          {busy && (
            <button type="button" className="danger" onClick={onCancel} disabled={cancelling}
              title="Stop the run for good; nothing of it is applied">
              {cancelling ? 'Aborting…' : 'Abort run'}
            </button>
          )}
          {!busy && canUndo && <button type="button" onClick={onUndo}>Previous layout</button>}
        </div>
      </form>

      {noFurniture && (
        <p className="muted small">The room is empty: place furniture or enable “Include more furniture” to get a design. Furniture search and critique work either way.</p>
      )}
      {busy && <p className="muted small">Processing the request ({furnitureCount} pieces in the room). This can take a minute or more.</p>}
      {error && <p className="error">{error}</p>}

      {result?.kind === 'critique' && result.critique && !busy && (
        <section className="design-result">
          <h3>Room critique</h3>
          <p>{result.critique.summary}</p>
          <div className="badges">
            {Object.entries(result.critique.scores).map(([name, score]) => (
              <span key={name} className="badge">{SCORE_LABELS[name] ?? name}: {score}/5</span>
            ))}
          </div>
          {result.critique.issues.length > 0 && (
            <ul className="warnings small">{result.critique.issues.map((i, n) => <li key={n}>{i.description}</li>)}</ul>
          )}
          {result.critique.suggestions.length > 0 && (
            <ul className="small">{result.critique.suggestions.map((s, n) => <li key={n}>{s}</li>)}</ul>
          )}
          <p className="muted small">The room was not changed.</p>
        </section>
      )}

      {result?.message && result.kind !== 'critique' && !busy && (
        <section className="design-result">
          <h3>{RESULT_TITLES[result.kind] ?? 'Notice'}</h3>
          <p>{result.message}</p>
          {result.downloads?.added?.length > 0 && (
            <p className="small result-line">New in the inventory (right sidebar): {result.downloads.added.map((a) => a.name).join(', ')}</p>
          )}
          <p className="muted small">Recognized as “{result.intent}” · the room was not changed.</p>
        </section>
      )}

      {result?.kind === 'design' && !result.message && !busy && (
        <section className="design-result">
          <h3>Design idea</h3>
          <p>{result.explanation}</p>
          {result.validation?.status === 'auto_corrected' && (
            <p className="small notice-inline">
              The validator made {result.validation.corrections.length} small correction(s).
            </p>
          )}
          {result.downloads?.added?.length > 0 && (
            <p className="small result-line">New in the inventory (right sidebar): {result.downloads.added.map((a) => a.name).join(', ')}</p>
          )}
          {missing.length > 0 && (
            <div className="missing-furniture">
              <strong className="small">Missing for your request</strong>
              <ul className="small">
                {missing.map((gap) => (
                  <li key={gap.item}>
                    <span><b>{gap.item}</b> <span className="muted">· {gap.reason}</span></span>
                    <button type="button" onClick={() => searchMissing(gap)} title="Start a furniture search for this piece">
                      Search
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {otherWarnings.length > 0 && (
            <ul className="warnings small">
              {otherWarnings.map((w) => <li key={w}>{w}</li>)}
            </ul>
          )}
          {added.length > 0 && (
            <p className="small result-line">Added: {added.map((p) => p.name).join(', ')}</p>
          )}
          {result.removed?.length > 0 && (
            <p className="small notice-inline">{result.removed.length} piece(s) removed from the room.</p>
          )}
          <p className="muted small">
            {result.placements.length} pieces arranged and validated
            {result.steps > 1 ? ` · ${result.steps} designer steps` : ''} · {result.model}
          </p>
        </section>
      )}
    </div>
  );
}
