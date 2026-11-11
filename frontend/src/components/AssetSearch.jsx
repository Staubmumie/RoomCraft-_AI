import { useEffect, useState } from 'react';
import { api } from '../api.js';
import { formatBytes, LicenseLabel, Thumbnail } from './common.jsx';

function AssetDetails({ assetId }) {
  const [state, setState] = useState({ loading: true });

  useEffect(() => {
    let active = true;
    api.details(assetId)
      .then((details) => active && setState({ details }))
      .catch((err) => active && setState({ error: err.message }));
    return () => { active = false; };
  }, [assetId]);

  if (state.loading) return <p className="muted small">Loading details…</p>;
  if (state.error) return <p className="error small">{state.error}</p>;
  const { details } = state;
  return (
    <div className="details small">
      {details.description && <p>{details.description.slice(0, 300)}{details.description.length > 300 ? '…' : ''}</p>}
      {details.polyCount && <p>Polygons: {details.polyCount.toLocaleString('en-US')}</p>}
      {details.tags?.length > 0 && <p className="muted">Tags: {details.tags.slice(0, 10).join(', ')}</p>}
      {details.files?.length > 0 && (
        <ul className="file-list">
          {details.files.slice(0, 8).map((f) => (
            <li key={f.filename}>
              <span>{f.filename}</span>
              <span className="muted">{f.format}{f.sizeBytes ? ` · ${formatBytes(f.sizeBytes)}` : ''}{f.requiresAuth ? ' · login required' : ''}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ResultCard({ asset, inInventory, downloading, error, onDownload }) {
  const [showDetails, setShowDetails] = useState(false);

  let action;
  if (inInventory) action = <button disabled>In inventory ✓</button>;
  else if (downloading) action = <button disabled><span className="spinner" /> Downloading…</button>;
  else if (!asset.loadable) action = <button disabled title="No GLB/glTF available">Format not supported</button>;
  else action = <button className="primary" onClick={() => onDownload(asset)}>Add to inventory</button>;

  return (
    <article className="card">
      <Thumbnail src={asset.thumbnailUrl} alt={asset.name} />
      <div className="card-body">
        <h3 title={asset.name}>{asset.name}</h3>
        <p className="muted small">{asset.provider}{asset.author ? ` · ${asset.author}` : ''}</p>
        <p className="small"><LicenseLabel license={asset.license} /></p>
        <div className="badges">
          {asset.formats.length > 0
            ? asset.formats.map((f) => <span key={f} className={`badge ${f === 'glb' || f === 'gltf' ? 'ok' : ''}`}>{f}</span>)
            : <span className="badge">Unknown format</span>}
        </div>
        <div className="card-actions">
          {action}
          <button className="link" onClick={() => setShowDetails((v) => !v)}>{showDetails ? 'Less' : 'Details'}</button>
          {asset.sourceUrl && <a className="link" href={asset.sourceUrl} target="_blank" rel="noreferrer">Source ↗</a>}
        </div>
        {error && <p className="error small">{error}</p>}
        {showDetails && <AssetDetails assetId={asset.id} />}
      </div>
    </article>
  );
}

export default function AssetSearch({ inventoryAssetIds, downloading, downloadErrors, onDownload }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const search = async (event) => {
    event.preventDefault();
    if (!query.trim() || loading) return;
    setLoading(true);
    setError('');
    try {
      const data = await api.search(query.trim());
      setResults(data.results);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel">
      <form className="search-form" onSubmit={search}>
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. sofa, chair, lamp" aria-label="Search term" />
        <button className="primary" disabled={loading || !query.trim()}>{loading ? <span className="spinner" /> : 'Search'}</button>
      </form>
      <p className="muted small">Only free, directly downloadable models. GLB/glTF models can be placed in the room.</p>
      {error && <p className="error">{error}</p>}
      {loading && <p className="muted">Searching all sources…</p>}
      {!loading && results?.length === 0 && <p className="muted">No results for “{query}”.</p>}
      <div className="card-list">
        {results?.map((asset) => (
          <ResultCard key={asset.id} asset={asset}
            inInventory={inventoryAssetIds.has(asset.id)}
            downloading={downloading.has(asset.id)}
            error={downloadErrors[asset.id]}
            onDownload={onDownload} />
        ))}
      </div>
    </div>
  );
}
