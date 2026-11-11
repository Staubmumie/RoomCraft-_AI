import { useState } from 'react';
import { formatDimensions, LicenseLabel, Thumbnail } from '../common.jsx';

// Shown while a run waits for the user: compare the proposed assets, pick some to download,
// or ask for other suggestions. Nothing is downloaded until "Download" is confirmed.
export default function ApprovalPanel({ approval, onRespond, onAbort, aborting }) {
  const [selected, setSelected] = useState(() => new Set());
  const [feedback, setFeedback] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const candidates = approval.candidates || [];

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
      await onRespond({ action, selected: action === 'approve' ? [...selected] : [], feedback });
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  };

  return (
    <section className="approval" aria-label="Approval">
      <div className="approval-header">
        <strong>Please choose</strong>
        <span className="small muted">Round {approval.round} · {candidates.length} suggestions</span>
      </div>
      {approval.summary && <p className="small">{approval.summary}</p>}

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
                {c.dimensions && <span className="badge">{formatDimensions(c.dimensions)}</span>}
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
        placeholder="Optional feedback, e.g. “darker wood” or “smaller furniture”" />
      <div className="card-actions">
        <button className="primary" disabled={busy || selected.size === 0} onClick={() => send('approve')}>
          {selected.size ? `Download ${selected.size}` : 'Download selection'}
        </button>
        <button disabled={busy} onClick={() => send('retry')}>Other suggestions</button>
        <button className="danger" disabled={busy} onClick={() => send('reject')}>Reject all</button>
        {onAbort && (
          <button className="danger" disabled={aborting} onClick={onAbort}
            title="End the whole run; nothing is downloaded or applied">
            {aborting ? 'Aborting…' : 'Abort run'}
          </button>
        )}
      </div>
      {busy && <p className="muted small"><span className="spinner" /> Resuming…</p>}
      {error && <p className="error small">{error}</p>}
    </section>
  );
}
