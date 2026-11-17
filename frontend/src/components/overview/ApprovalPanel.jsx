import { useState } from 'react';
import { formatDimensions, LicenseLabel, Thumbnail } from '../common.jsx';

const CHECK_LABELS = { category: 'type', size: 'size', style: 'style', quality: 'model', license: 'licence' };
export const PROGRESS_LABELS = {
  searching_local: 'searching locally', searching_online: 'searching online', retrying: 'searching again',
  pending: 'queued', awaiting_approval: 'ready for you', downloading: 'downloading', completed: 'downloaded',
  found_locally: 'in the inventory', failed: 'download failed', skipped: 'skipped', rejected: 'rejected',
  not_found: 'not found'
};

// Shown while a run waits for the user: ONE item of the furniture list at a time. The other items
// keep being searched and downloaded in the background; nothing is downloaded until "Download" is confirmed.
export default function ApprovalPanel({ approval, onRespond, onAbort, aborting }) {
  const [selected, setSelected] = useState(() => new Set());
  const [feedback, setFeedback] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const candidates = approval.candidates || [];
  const request = approval.request || {};
  const lastRun = approval.run >= (approval.maxRuns || 2);
  const others = Object.entries(approval.progress || {});

  const toggle = (assetId) => setSelected((current) => {
    const next = new Set(current);
    if (next.has(assetId)) next.delete(assetId);
    else next.add(assetId);
    return next;
  });

  const send = async (action) => {
    setBusy(true);
    setError('');
    try {
      await onRespond({
        itemKey: approval.itemKey, run: approval.run, action,
        selected: action === 'approve' ? [...selected] : [], feedback
      });
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  };

  return (
    <section className="approval" aria-label="Approval">
      <div className="approval-header">
        <strong>
          Item {approval.index} of {approval.total}
          {request.essential && <span className="badge warn">Essential</span>}
          {approval.run > 1 && <span className="badge warn">2nd search</span>}
        </strong>
        <span className="small muted">{candidates.length} suggestion{candidates.length === 1 ? '' : 's'}</span>
      </div>
      <div className="approval-request">
        <p><b>{request.description}</b></p>
        {(request.purpose || request.size || request.style || request.colors_materials) && (
          <p className="small muted">
            {[request.purpose, request.size, request.style, request.colors_materials].filter(Boolean).join(' · ')}
          </p>
        )}
        {approval.intent && <p className="small">Research: {approval.intent}</p>}
        {approval.summary && <p className="small muted">{approval.summary}</p>}
      </div>

      <div className="approval-list">
        {candidates.map((c) => (
          <label key={c.assetId} className={`approval-card ${selected.has(c.assetId) ? 'selected' : ''}`}>
            <input type="checkbox" checked={selected.has(c.assetId)} onChange={() => toggle(c.assetId)} disabled={busy} />
            <Thumbnail src={c.thumbnailUrl} alt={c.name} />
            <div className="card-body">
              <h3 title={c.name}>{c.name}</h3>
              <p className="muted small">{c.provider}{c.author ? ` · ${c.author}` : ''}</p>
              {c.description && <p className="small clamp">{c.description}</p>}
              <div className="badges">
                {c.formats.map((f) => <span key={f} className={`badge ${f === 'glb' || f === 'gltf' ? 'ok' : ''}`}>{f}</span>)}
                <span className="badge">{c.dimensions ? formatDimensions(c.dimensions) : 'size unknown'}</span>
                {Object.entries(c.checks || {}).map(([check, value]) => (
                  <span key={check} className={`badge ${value === 'ok' ? 'ok' : value === 'fails' ? 'warn' : ''}`}
                    title={`${CHECK_LABELS[check] ?? check}: ${value}`}>{CHECK_LABELS[check] ?? check} {value}</span>
                ))}
              </div>
              <p className="small"><LicenseLabel license={c.license} /></p>
              <p className="small reason">{c.reason}</p>
              <p className="small muted">
                {c.previewInspected ? `Preview checked${c.visualCheck ? `: ${c.visualCheck}` : ''}` : 'Preview not checked'}
              </p>
              {c.sourceUrl && <a className="link small" href={c.sourceUrl} target="_blank" rel="noreferrer">Source ↗</a>}
            </div>
          </label>
        ))}
      </div>

      <textarea className="approval-feedback" rows={2} value={feedback} disabled={busy}
        onChange={(e) => setFeedback(e.target.value)}
        placeholder={lastRun ? 'Optional feedback' : 'Optional feedback for the second search, e.g. “darker wood” or “smaller”'} />
      <div className="card-actions">
        <button className="primary" disabled={busy || selected.size === 0} onClick={() => send('approve')}>
          {selected.size ? `Download ${selected.size}` : 'Download selection'}
        </button>
        <button disabled={busy} onClick={() => send('reject')}
          title={lastRun ? 'No further search: this item is skipped' : 'Search this item once more with other terms; you continue with the next item meanwhile'}>
          {lastRun ? 'Nothing fits – skip' : 'Nothing fits'}
        </button>
        {!lastRun && <button disabled={busy} onClick={() => send('skip')} title="End this item without another search">Skip</button>}
        {onAbort && (
          <button className="danger" disabled={aborting} onClick={onAbort}
            title="End the whole run; nothing more is downloaded or applied">
            {aborting ? 'Aborting…' : 'Abort run'}
          </button>
        )}
      </div>
      {others.length > 0 && (
        <p className="small muted research-progress">
          Other items: {others.map(([status, n]) => <span key={status} className="badge">{n} {PROGRESS_LABELS[status] ?? status}</span>)}
        </p>
      )}
      {approval.localDone?.length > 0 && (
        <ul className="small muted approval-done">
          {approval.localDone.map((d) => (
            <li key={d.description}>✓ {d.description}: {d.items.map((i) => i.name).join(', ')}</li>
          ))}
        </ul>
      )}
      {busy && <p className="muted small"><span className="spinner" /> Continuing…</p>}
      {error && <p className="error small">{error}</p>}
    </section>
  );
}
