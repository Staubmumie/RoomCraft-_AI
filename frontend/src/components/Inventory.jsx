import { formatBytes, formatDimensions, LicenseLabel, Thumbnail } from './common.jsx';

export default function Inventory({ items, loading, error, placingId, placedCounts, onPlace, onDelete, onReload }) {
  if (loading) return <div className="panel"><p className="muted">Loading inventory…</p></div>;

  return (
    <div className="panel">
      {error && (
        <p className="error">{error} <button className="link" onClick={onReload}>Reload</button></p>
      )}
      {!error && items.length === 0 && (
        <p className="muted">Nothing downloaded yet. Search for furniture in the “Search” tab and add it.</p>
      )}
      <div className="card-list">
        {items.map((item) => {
          const placing = placingId === item.id;
          const placed = placedCounts.get(item.id) || 0;
          return (
            <article key={item.id} className={`card ${placing ? 'placing' : ''}`}>
              <Thumbnail src={item.thumbnailUrl} alt={item.name} />
              <div className="card-body">
                <h3 title={item.name}>{item.name}</h3>
                <p className="muted small">{item.provider}{item.author ? ` · ${item.author}` : ''}</p>
                <p className="small"><LicenseLabel license={item.license} /></p>
                <div className="badges">
                  <span className="badge ok">{item.format}</span>
                  {item.sizeBytes > 0 && <span className="badge">{formatBytes(item.sizeBytes)}</span>}
                  {item.dimensions && <span className="badge" title="Width × depth × height">{formatDimensions(item.dimensions)}</span>}
                  {placed > 0 && <span className="badge">{placed}× in the room</span>}
                </div>
                {item.missing && <p className="error small">The model file is missing on disk. Delete the entry and download the asset again.</p>}
                <div className="card-actions">
                  <button className={placing ? '' : 'primary'} disabled={item.missing} onClick={() => onPlace(item)}>
                    {placing ? 'Cancel' : 'Place'}
                  </button>
                  <button className="danger" onClick={() => onDelete(item)}>Delete</button>
                  {item.sourceUrl && <a className="link" href={item.sourceUrl} target="_blank" rel="noreferrer">Source ↗</a>}
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}
