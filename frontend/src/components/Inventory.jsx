import { useMemo, useState } from 'react';
import { formatBytes, formatDimensions, LicenseLabel, Thumbnail } from './common.jsx';

// Every word of the query must appear somewhere in the item's name, source, description or tags.
function matches(item, words) {
  const text = [item.name, item.provider, item.author, item.description, ...(item.tags || [])]
    .filter(Boolean).join(' ').toLowerCase();
  return words.every((word) => text.includes(word));
}

export default function Inventory({ items, loading, error, placingId, placedCounts, onPlace, onDelete, onReload }) {
  const [query, setQuery] = useState('');
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const visible = useMemo(
    () => (words.length ? items.filter((item) => matches(item, words)) : items),
    [items, query]
  );

  if (loading) return <div className="panel"><p className="muted">Loading inventory…</p></div>;

  return (
    <div className="panel">
      {error && (
        <p className="error">{error} <button className="link" onClick={onReload}>Reload</button></p>
      )}
      {!error && items.length === 0 && (
        <p className="muted">Nothing downloaded yet. Search for furniture in the “Search” tab and add it.</p>
      )}
      {items.length > 0 && (
        <div className="search-form">
          <input type="search" value={query} onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Escape') setQuery(''); }}
            placeholder="Filter inventory, e.g. bean bag" aria-label="Filter inventory" />
        </div>
      )}
      {words.length > 0 && (
        <p className="muted small">
          {visible.length === 0 ? 'No inventory item matches.' : `${visible.length} of ${items.length} items`}
        </p>
      )}
      <div className="card-list">
        {visible.map((item) => {
          const placing = placingId === item.id;
          const placed = placedCounts.get(item.id) || 0;
          // Dimensions are stored in model units; unitScale turns them into metres.
          const scale = item.unitScale ?? 1;
          const dims = item.dimensions && {
            width: item.dimensions.width * scale, depth: item.dimensions.depth * scale, height: item.dimensions.height * scale
          };
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
                  {dims && <span className="badge" title="Width × depth × height">{formatDimensions(dims)}</span>}
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
