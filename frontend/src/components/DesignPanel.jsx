import { useEffect, useState } from 'react';
import { api } from '../api.js';
import CritiqueView from './CritiqueView.jsx';

// Storage keys keep the former product name so saved settings survive.
const MODEL_KEY = 'roomforge.designModel';
const AGENT_MODELS_KEY = 'roomforge.agentModels';
const OPTIONS_KEY = 'roomforge.designOptions';
const RESULT_TITLES = { design: 'Design', research: 'Furniture search', reject: 'Request not processed' };

const DEFAULT_OPTIONS = {
  allowNewFurniture: false, allowRemoval: false, finalReview: true, maxIterations: 8, maxResearch: 2,
  searchOnline: true, reviewSuggestions: true
};
const MAX_ITERATIONS = 15;
const MAX_RESEARCH = 5;
const PIECES_PER_LIST = 6;  // the furniture research cuts longer lists (backend: MAX_ITEMS_PER_REQUEST)
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
  try {
    // Only known options: settings that were removed meanwhile are dropped.
    const saved = JSON.parse(localStorage.getItem(OPTIONS_KEY) || '{}');
    return Object.fromEntries(Object.entries(DEFAULT_OPTIONS).map(([key, value]) => [key, saved?.[key] ?? value]));
  } catch { return DEFAULT_OPTIONS; }
}

// One collapsible settings group; the summary shows the current values while it is closed.
function SettingsGroup({ title, summary, children, open = false }) {
  return (
    <details className="settings-group" open={open}>
      <summary>
        <span className="settings-title">{title}</span>
        <span className="settings-summary muted">{summary}</span>
      </summary>
      <div className="settings-body">{children}</div>
    </details>
  );
}

function Toggle({ checked, onChange, disabled, title, children }) {
  return (
    <label className={`toggle ${disabled ? 'disabled' : ''}`}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span>{title}<small className="muted">{children}</small></span>
    </label>
  );
}

function NumberOption({ label, unit, value, min, max, disabled, onChange, children }) {
  return (
    <label className={`field inline ${disabled ? 'disabled' : ''}`}>
      <span className="option-label">{label}<small className="muted">{children}</small></span>
      <span className="number-input">
        <input type="number" min={min} max={max} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)} />
        {unit && <span className="muted small">{unit}</span>}
      </span>
    </label>
  );
}

function DesignResult({ result, onSearchMissing }) {
  const added = result.placements?.filter((p) => p.id.startsWith('new-')) ?? [];
  // Missing essentials are shown with a search action; their "Not fulfilled" warnings would repeat them.
  const missing = result.missing ?? [];
  const warnings = (result.warnings ?? []).filter((w) => !(missing.length && w.startsWith('Not fulfilled:')));
  const corrections = result.validation?.status === 'auto_corrected' ? result.validation.corrections.length : 0;
  return (<>
    <p className="result-text">{result.explanation}</p>
    <div className="result-facts">
      <span className="badge ok">{result.placements.length} pieces placed</span>
      {added.length > 0 && <span className="badge ok">+{added.length} added</span>}
      {result.removed?.length > 0 && <span className="badge warn">−{result.removed.length} removed</span>}
      {corrections > 0 && <span className="badge">{corrections} auto-correction{corrections === 1 ? '' : 's'}</span>}
      {result.steps > 1 && <span className="badge">{result.steps} designer steps</span>}
    </div>
    {added.length > 0 && (
      <p className="small"><span className="muted">Added:</span> {added.map((p) => p.name).join(', ')}</p>
    )}
    {result.downloads?.added?.length > 0 && (
      <p className="small result-line">New in the inventory: {result.downloads.added.map((a) => a.name).join(', ')}</p>
    )}
    {missing.length > 0 && (
      <div className="missing-furniture">
        <strong className="small">Still missing for your request</strong>
        <ul className="small">
          {missing.map((gap) => (
            <li key={gap.item}>
              <span><b>{gap.item}</b> <span className="muted">· {gap.reason}</span></span>
              <button type="button" onClick={() => onSearchMissing(gap)} title="Start a furniture search for this piece">
                Search
              </button>
            </li>
          ))}
        </ul>
      </div>
    )}
    {warnings.length > 0 && (
      <details className="result-warnings">
        <summary className="small">{warnings.length} note{warnings.length === 1 ? '' : 's'} from the validation</summary>
        <ul className="warnings small">{warnings.map((w) => <li key={w}>{w}</li>)}</ul>
      </details>
    )}
  </>);
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
    // Reviewing suggestions only exists with the online search; without it nothing is downloaded at all.
    const reviewSuggestions = designOptions.reviewSuggestions || !designOptions.searchOnline;
    setBusy(true);
    setError('');
    try {
      // Only real overrides of known agents are sent; everything else uses the backend default.
      const models = Object.fromEntries(agents.filter((a) => agentModels[a.id]).map((a) => [a.id, agentModels[a.id]]));
      setResult(await onGenerate({
        request: text.trim(), model, models,
        options: { ...designOptions, maxIterations, maxResearch, reviewSuggestions }
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

  // Starts a furniture search for a piece the design could not provide (with the search settings below).
  const searchMissing = (gap) => runRequest(`Find furniture for my room: ${gap.item} (${gap.reason}).`);

  // Keep the saved model selectable even if the backend no longer lists it.
  const withSaved = (id) => (id && !models.some((m) => m.id === id) ? [...models, { id, label: id }] : models);
  const modelLabel = (id) => models.find((m) => m.id === id)?.label ?? id;
  const overridden = agents.filter((a) => agentModels[a.id]);
  const agentSummary = overridden.length
    ? overridden.map((a) => `${a.label}: ${modelLabel(agentModels[a.id])}`).join(', ')
    : `default (${[...new Set(agents.map((a) => modelLabel(a.default)))].join(', ')})`;

  const o = designOptions;
  const canResearch = o.allowNewFurniture && o.searchOnline;
  const designerSummary = [
    o.allowNewFurniture ? 'adds furniture' : 'room furniture only',
    o.allowRemoval ? 'may remove' : null,
    o.finalReview ? 'final review' : null,
    `${o.maxIterations || '–'} steps`
  ].filter(Boolean).join(' · ');
  const searchSummary = !o.searchOnline
    ? 'off · your database only'
    : [
      canResearch ? `${o.maxResearch || 0} list${Number(o.maxResearch) === 1 ? '' : 's'} × ${PIECES_PER_LIST}` : null,
      o.reviewSuggestions ? 'you choose' : 'auto-download'
    ].filter(Boolean).join(' · ');

  const showResult = result && !busy;
  const resultTitle = result?.kind === 'critique' ? 'Room critique' : result?.message
    ? (RESULT_TITLES[result.kind] ?? 'Notice') : 'Design applied';

  return (
    <div className="panel design-panel">
      {showResult && (
        <section className={`result-card kind-${result.kind}`}>
          <header className="result-header">
            <h3>{resultTitle}</h3>
            <button type="button" className="icon-button small-icon" onClick={() => setResult(null)}
              title="Dismiss" aria-label="Dismiss result">×</button>
          </header>

          {result.kind === 'critique' && result.critique && (<>
            <CritiqueView critique={result.critique} />
            <p className="result-foot">The room was not changed. Use the suggestions in a design request to apply them.</p>
          </>)}

          {result.message && result.kind !== 'critique' && (<>
            <p className="result-text">{result.message}</p>
            {result.downloads?.added?.length > 0 && (
              <p className="small result-line">New in the inventory: {result.downloads.added.map((a) => a.name).join(', ')}</p>
            )}
            <p className="result-foot">Recognized as “{result.intent}” · the room was not changed.</p>
          </>)}

          {result.kind === 'design' && !result.message && (<>
            <DesignResult result={result} onSearchMissing={searchMissing} />
            <p className="result-foot">Validated layout · {result.model}</p>
          </>)}

          {canUndo && (
            <div className="card-actions">
              <button type="button" onClick={onUndo}>↶ Previous layout</button>
            </div>
          )}
        </section>
      )}

      <form className="design-form" onSubmit={submit}>
        <label className="field">
          <span>What should happen with the room?</span>
          <textarea rows={4} value={request} onChange={(e) => setRequest(e.target.value)} disabled={busy}
            placeholder="e.g. “Make my living room cozy”, “Critique my room” or “Find a green armchair”"
            onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) submit(e); }} />
        </label>
        <div className="card-actions">
          <button className="primary" disabled={busy || !request.trim()} title="Ctrl+Enter">
            {busy ? <><span className="spinner" /> Agents are working…</> : 'Send request'}
          </button>
          {busy && (
            <button type="button" className="danger" onClick={onCancel} disabled={cancelling}
              title="Stop the run for good; nothing of it is applied">
              {cancelling ? 'Aborting…' : 'Abort run'}
            </button>
          )}
          {!busy && canUndo && !showResult && <button type="button" onClick={onUndo}>↶ Previous layout</button>}
          <span className="muted small push-right">{furnitureCount} piece{furnitureCount === 1 ? '' : 's'} in the room</span>
        </div>
        {busy && <p className="muted small">Follow the agents live in the Overview tab. This can take a minute or more.</p>}
        {noFurniture && !busy && (
          <p className="hint-inline small">The room is empty: place furniture or enable “Add furniture from the inventory” to get a design. Critique and furniture search work either way.</p>
        )}
        {error && <p className="error">{error}</p>}

        <h3 className="section-title">Settings</h3>
        <fieldset className="settings" disabled={busy}>
          <SettingsGroup title="Models" summary={`Designer: ${modelLabel(model) || '–'}`}>
            <label className="field inline">
              <span>Interior Designer</span>
              <select value={model} onChange={(e) => setModel(e.target.value)}>
                {withSaved(model).map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
            </label>
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
            {agents.length > 0 && <p className="muted small">Other agents: {agentSummary}</p>}
          </SettingsGroup>

          <SettingsGroup title="Designer" summary={designerSummary}>
            <Toggle checked={o.allowNewFurniture} onChange={(v) => setOption('allowNewFurniture', v)}
              title="Add furniture from the inventory">
              The designer may bring pieces from your inventory into the room, including pieces it has searched for during the run.
            </Toggle>
            <Toggle checked={o.allowRemoval} onChange={(v) => setOption('allowRemoval', v)} title="Allow removing furniture">
              Otherwise every piece that is in the room now stays. Pieces added during the run can always be removed again.
            </Toggle>
            <Toggle checked={o.finalReview} onChange={(v) => setOption('finalReview', v)} title="Final review by the critic">
              When the designer is done, the critic reviews the result once more and the designer gets one last step to act on it.
            </Toggle>
            <NumberOption label="Designer steps" unit="max." value={o.maxIterations} min={1} max={MAX_ITERATIONS}
              onChange={(v) => setOption('maxIterations', v)}>
              Each step is one decision of the designer (plan, place, repair, ask for a critique or for furniture). Validation, screenshots and critiques do not count.
            </NumberOption>
          </SettingsGroup>

          <SettingsGroup title="Furniture search" summary={searchSummary}>
            <Toggle checked={o.searchOnline} onChange={(v) => setOption('searchOnline', v)} title="Search furniture online">
              Missing furniture is looked up in free online 3D libraries and downloaded into your inventory. Off: only your own furniture database is used.
            </Toggle>
            <Toggle checked={o.reviewSuggestions && o.searchOnline} disabled={!o.searchOnline}
              onChange={(v) => setOption('reviewSuggestions', v)} title="Let me choose the downloads">
              You get up to four suggestions per piece and decide what is downloaded; the other pieces keep searching meanwhile. Off: the best match is downloaded automatically.
            </Toggle>
            <NumberOption label="Furniture lists" unit="max." value={o.maxResearch} min={0} max={MAX_RESEARCH}
              disabled={!canResearch} onChange={(v) => setOption('maxResearch', v)}>
              How often the designer may send a list of missing pieces to the furniture search during one design. Each list holds up to {PIECES_PER_LIST} pieces (e.g. sofa, dining chairs, floor lamp), and every piece gets a second search if the first finds nothing suitable. Lists fully answered by your database do not count. 0 = the designer never searches online. Needs “Add furniture from the inventory”.
            </NumberOption>
          </SettingsGroup>
        </fieldset>
      </form>
    </div>
  );
}
